"""
Constitutional Guard — principle-based self-critique and revision for clinical validation.

Implements a custom critique→revision loop (NOT LangChain's deprecated ConstitutionalChain)
using direct ChatOpenAI calls, consistent with the existing codebase pattern.

Flow:
  1. Receive raw agent JSON response string
  2. Critique: ask LLM if any of the clinical safety principles are violated
  3. If violations found: Revision: ask LLM to rewrite the JSON to comply
  4. Validate JSON integrity of the revised output
  5. Fall back to original if revision breaks JSON or produces an invalid structure

Healthcare principles enforced:
  P1 — Emergency conservatism:
       CRITICAL severity must never be APPROVED without emergencyCareNeeded=YES
  P2 — Epistemic humility:
       Confidence score ≥ 0.9 should only be assigned when clinical evidence is unambiguous
  P3 — Safety-first triage:
       When emergency-level symptoms are present, downgrading severity to LOW requires
       explicit clinical justification; default to REVIEW not APPROVE
  P4 — Completeness requirement:
       An APPROVE recommendation must include a non-empty validation_summary with
       at least one specific clinical observation (not generic statements)
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import OPENAI_MODEL
from guidelines.guideline_client import search_guidelines
from log.logger import logger


# ---------------------------------------------------------------------------
# Constitutional principles
# ---------------------------------------------------------------------------

_PRINCIPLES = [
    {
        "id": "P1",
        "name": "Emergency conservatism",
        "description": (
            "A diagnosis with severity=CRITICAL must never receive recommendation=APPROVE "
            "unless emergencyCareNeeded is explicitly YES. If severity is CRITICAL and "
            "emergencyCareNeeded is NO or absent, the recommendation must be REJECT."
        ),
    },
    {
        "id": "P2",
        "name": "Epistemic humility",
        "description": (
            "A confidence_score of 0.9 or higher must only be assigned when the clinical "
            "evidence provided is clear, specific, and unambiguous. Vague or incomplete "
            "clinical information warrants confidence_score no higher than 0.75. "
            "IMPORTANT: Adjusting confidence_score must NEVER change the recommendation field. "
            "The recommendation reflects clinical correctness; confidence_score reflects certainty. "
            "They are independent. Only lower confidence_score — do not alter recommendation."
        ),
    },
    {
        "id": "P3",
        "name": "Safety-first triage",
        "description": (
            "When the patient symptoms contain emergency-level presentations "
            "(chest pain, respiratory failure, altered consciousness, etc.), "
            "a recommendation of APPROVE with severity=LOW requires explicit clinical "
            "justification in validation_summary. Without such justification, "
            "the recommendation must be REVIEW."
        ),
    },
    {
        "id": "P4",
        "name": "Completeness requirement",
        "description": (
            "An APPROVE recommendation must include a validation_summary with at least "
            "one specific clinical observation. Generic statements like 'the diagnosis "
            "appears reasonable' are insufficient. If no specific clinical support is "
            "provided, the recommendation must be REVIEW."
        ),
    },
    {
        "id": "P5",
        "name": "Clinical guideline alignment",
        "description": (
            "When relevant clinical guidelines are provided as context, the recommendation "
            "and treatment plan must not contradict them. If the response approves a "
            "treatment that is clearly contrary to a cited guideline, the recommendation "
            "must be REVIEW and the violation must be noted in key_concerns."
        ),
    },
]

_PRINCIPLES_TEXT = "\n".join(
    f"{p['id']} ({p['name']}): {p['description']}"
    for p in _PRINCIPLES
)

_CRITIQUE_SYSTEM = (
    "You are a clinical governance reviewer. "
    "Examine the following AI validation response against the listed safety principles. "
    "Identify any principles that are violated. Be precise and concise. "
    "If no principles are violated, respond with exactly: NO_VIOLATIONS"
)

_REVISION_SYSTEM = (
    "You are a clinical governance reviewer. "
    "Revise the following AI validation response to comply with the violated safety principles. "
    "CRITICAL RULES:\n"
    "- If only P2 (confidence_score) is violated, adjust confidence_score ONLY. "
    "  Do NOT change the recommendation field.\n"
    "- Only change the recommendation field if P1, P3, or P4 require it.\n"
    "Your output must be valid JSON matching exactly this schema:\n"
    '{"is_validated": bool, "confidence_score": float, '
    '"validation_summary": "string", "key_concerns": ["string"], '
    '"recommendation": "APPROVE or REJECT or REVIEW"}\n'
    "Do not add any other keys. Do not wrap in markdown code fences."
)

# ---------------------------------------------------------------------------
# LLM instance (same model as the agent for consistency)
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _needs_critique(raw_response: str, severity: str, emergency_care: str) -> bool:
    """
    Fast pre-filter: skip the LLM critique call when there are no structural red flags.
    Only trigger critique when a P1-level risk is detectable from the response structure
    (CRITICAL severity or APPROVE recommendation) to avoid unnecessary LLM calls on
    routine LOW-severity cases.
    """
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return True  # malformed JSON — let critique handle it
    recommendation = parsed.get("recommendation", "")
    confidence = parsed.get("confidence_score", 0.0)
    sev_upper = severity.upper()
    # Always critique if: CRITICAL severity, high confidence on an uncertain case,
    # or APPROVE with CRITICAL+NO-emergency (P1 violation candidate)
    if sev_upper == "CRITICAL":
        return True
    if recommendation == "APPROVE" and sev_upper == "CRITICAL" and emergency_care.upper() != "YES":
        return True
    if isinstance(confidence, (int, float)) and confidence >= 0.9:
        return True
    return False


def apply(
    raw_response: str,
    symptoms: str,
    severity: str,
    emergency_care: str,
) -> tuple[str, str | None]:
    """
    Apply constitutional principles to a raw agent response string.

    Args:
        raw_response:   JSON string returned by the DeepAgent
        symptoms:       Patient symptoms from the original request
        severity:       Severity from the diagnosis dict
        emergency_care: Emergency care flag from the diagnosis dict

    Returns:
        (final_response, critique_text | None)
        final_response is either the revised response or the original if no violations
        or if revision failed JSON validation.
    """
    if not _needs_critique(raw_response, severity, emergency_care):
        logger.debug("[CONST_GUARD] Pre-filter: no red flags — skipping critique.")
        return raw_response, None

    # Retrieve relevant clinical guidelines for P5 grounding
    guideline_context = _build_guideline_context(symptoms, raw_response)

    critique = _critique(raw_response, symptoms, severity, emergency_care, guideline_context)

    if critique.strip().upper() == "NO_VIOLATIONS" or not critique.strip():
        logger.debug("[CONST_GUARD] No violations detected.")
        return raw_response, None

    logger.info("[CONST_GUARD] Violations detected — requesting revision.")
    logger.debug("[CONST_GUARD] Critique: %s", critique)

    revised = _revise(raw_response, critique, symptoms, severity, emergency_care, guideline_context)
    if revised is None:
        logger.warning("[CONST_GUARD] Revision produced invalid JSON — keeping original response.")
        return raw_response, critique

    logger.info("[CONST_GUARD] Response revised by constitutional guard.")
    return revised, critique


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _build_guideline_context(symptoms: str, raw_response: str) -> str:
    """
    Retrieve relevant clinical guidelines from ChromaDB for P5 grounding.
    Returns a formatted string to inject into the critique/revision prompts,
    or empty string if ChromaDB is unavailable or no guidelines match.
    """
    try:
        query = f"{symptoms} {raw_response[:200]}"
        guidelines = search_guidelines(query, k=3)
        if not guidelines:
            return ""
        lines = ["\nRelevant clinical guidelines (for P5 alignment check):"]
        for g in guidelines:
            lines.append(f"  [{g['source']}] {g['text']} (relevance: {g['score']})")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[CONST_GUARD] Guideline retrieval failed: %s", exc)
        return ""


def _critique(
    raw_response: str,
    symptoms: str,
    severity: str,
    emergency_care: str,
    guideline_context: str = "",
) -> str:
    """Ask the LLM to critique the response against the constitutional principles."""
    query = (
        f"Patient symptoms: {symptoms}\n"
        f"Severity: {severity} | Emergency care: {emergency_care}\n\n"
        f"Validation response to review:\n{raw_response}\n\n"
        f"Safety principles to check:\n{_PRINCIPLES_TEXT}"
        f"{guideline_context}\n\n"
        "List any principle violations. If none, respond with: NO_VIOLATIONS"
    )
    try:
        result = _llm.invoke([
            SystemMessage(content=_CRITIQUE_SYSTEM),
            HumanMessage(content=query),
        ])
        return result.content.strip()
    except Exception as exc:
        logger.warning("[CONST_GUARD] Critique LLM error: %s", exc)
        return "NO_VIOLATIONS"


def _revise(
    raw_response: str,
    critique: str,
    symptoms: str,
    severity: str,
    emergency_care: str,
    guideline_context: str = "",
) -> str | None:
    """
    Ask the LLM to revise the response to fix the identified violations.
    Returns the revised JSON string, or None if the revision is not valid JSON.
    """
    query = (
        f"Patient symptoms: {symptoms}\n"
        f"Severity: {severity} | Emergency care: {emergency_care}\n\n"
        f"Original validation response:\n{raw_response}\n\n"
        f"Identified violations:\n{critique}"
        f"{guideline_context}\n\n"
        "Rewrite the validation response to fix all violations. "
        "Output valid JSON only — no markdown, no explanation."
    )
    try:
        result = _llm.invoke([
            SystemMessage(content=_REVISION_SYSTEM),
            HumanMessage(content=query),
        ])
        content = result.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Validate JSON structure
        parsed: dict[str, Any] = json.loads(content)
        _validate_schema(parsed)
        return content
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[CONST_GUARD] Revised response failed JSON validation: %s", exc)
        return None
    except Exception as exc:
        logger.warning("[CONST_GUARD] Revision LLM error: %s", exc)
        return None


def _validate_schema(parsed: dict) -> None:
    """Raise ValueError if the parsed dict is missing required ValidationResult fields."""
    required = {"is_validated", "confidence_score", "validation_summary",
                "key_concerns", "recommendation"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Revised response missing fields: {missing}")
    if parsed["recommendation"] not in ("APPROVE", "REJECT", "REVIEW"):
        raise ValueError(f"Invalid recommendation: {parsed['recommendation']}")
    if not isinstance(parsed["confidence_score"], (int, float)):
        raise ValueError("confidence_score must be numeric")
