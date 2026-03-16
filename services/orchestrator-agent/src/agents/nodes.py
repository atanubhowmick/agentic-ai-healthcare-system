"""
LangGraph node implementations for the Healthcare Orchestrator.

Node execution order (happy path):
  chroma_lookup → triage → specialist → [secondary_check → conflict_check]
  → xai_diagnosis_validator → treatment → xai_treatment_validator → finish

ChromaDB integration:
  - chroma_lookup_node  : first node — returns cached treatment if similarity >= threshold
  - xai_diagnosis_validator_node : saves validated diagnosis to ChromaDB (non-blocking)
  - xai_treatment_validator_node : saves validated treatment to ChromaDB (non-blocking)

Retry loops:
  - xai_diagnosis_validator → specialist (max MAX_RETRY times, then human review)
  - xai_treatment_validator → treatment   (max MAX_RETRY times, then human review)

Conflict / human review:
  - conflict_check → finish (if conflict detected)
  - any node can set requires_human_review=True → finish node collects it
"""

import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
from tools.cardiology_client import call_cardiology_api
from tools.neurology_client import call_neurology_api
from tools.pathology_client import call_pathology_api
from tools.cancer_client import call_cancer_api
from tools.treatment_client import call_treatment_api
from tools.xai_client import call_validate_diagnosis, call_validate_treatment
from core.config import MAX_RETRY_COUNT
from agents.triage_router import route_symptoms
from core.mongo_client import save_case
from core.chroma_client import (
    lookup_treatment_recommendation,
    save_diagnosis_outcome,
    save_treatment_outcome,
)
from log.logger import logger

_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# -- Prompts ------------------------------------------------------------------

_CONFLICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical conflict resolution specialist.
Two AI agents have assessed the same patient. Determine whether their findings materially conflict.

A CONFLICT exists when:
- One agent reports LOW/normal severity while another reports HIGH or CRITICAL
- Diagnoses directly contradict each other (e.g. "healthy" vs "at risk")
- Emergency care recommendations significantly disagree

Respond ONLY with valid JSON (no markdown):
{{
    "conflict_detected": false,
    "conflict_reason": "No significant conflict detected between the two assessments.",
    "resolution_needed": false
}}"""),
    ("human", """Primary Assessment ({primary_agent}):
{primary_summary}
Severity: {primary_severity}

Secondary Assessment ({secondary_agent}):
{secondary_summary}
Severity: {secondary_severity}"""),
])

_conflict_chain = _CONFLICT_PROMPT | _llm


# -- Helpers ------------------------------------------------------------------

def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def _diagnosis_summary(diagnosis: dict) -> str:
    return (
        diagnosis.get("diagnosysDetails") or   # cardiology (original typo preserved)
        diagnosis.get("diagnosisDetails") or   # neurology / cancer
        diagnosis.get("analysisDetails") or    # pathology
        str(diagnosis)
    )


# -- Nodes --------------------------------------------------------------------

async def chroma_lookup_node(state: AgentState) -> dict:
    """
    First node in the pipeline (step 2.1).
    Searches ChromaDB treatment_outcomes for a semantically similar prior case.
    On cache hit, builds a pre-filled final_response and short-circuits to finish.
    """
    symptoms = state.get("symptoms", "")
    patient_id = state.get("patient_id", "UNKNOWN")

    logger.info("[CHROMA_LOOKUP] Checking cache | patient: %s", patient_id)

    hit, cached = await lookup_treatment_recommendation(symptoms)

    if hit:
        score = cached.get("similarity_score", 0.0)
        specialist = cached.get("specialist_agent", "Unknown")
        logger.info("[CHROMA_LOOKUP] Cache hit | patient: %s | similarity: %.4f", patient_id, score)

        final_response = {
            "patient_id": patient_id,
            "status": "COMPLETED_FROM_CACHE",
            "specialist_agent": specialist,
            "diagnosis": {
                "summary": cached.get("diagnosis_summary", "Cached result — see treatment"),
                "severity": cached.get("severity", "N/A"),
                "emergency_care_needed": "N/A",
                "hospitalization_needed": "N/A",
                "full_details": {},
            },
            "xai_diagnosis_validation": None,
            # Wrap in the same envelope the treatment agent returns so the UI
            # can render it identically to a non-cached response.
            "treatment": {
                "agent": "Treatment_Care_Agent",
                "agent_id": "TREAT-AGENT-CACHE",
                "treatment": cached.get("treatment"),
            },
            "xai_treatment_validation": None,
            "conflict_detected": False,
            "conflict_reason": "",
            "human_review_reason": None,
            "audit_trail": [
                f"[CHROMA_LOOKUP] Cache hit (similarity={score}) "
                "— returning cached treatment recommendation"
            ],
        }
        return {
            "chroma_cache_hit": True,
            "chroma_cached_result": cached,
            "final_response": final_response,
            "messages": [
                f"[CHROMA_LOOKUP] Cache hit | similarity: {score} — skipping full diagnosis pipeline"
            ],
        }

    logger.info("[CHROMA_LOOKUP] No cache hit | patient: %s — proceeding with full pipeline", patient_id)
    return {
        "chroma_cache_hit": False,
        "chroma_cached_result": None,
        "messages": ["[CHROMA_LOOKUP] No cache hit — proceeding with full diagnosis flow"],
    }


async def triage_node(state: AgentState) -> dict:
    """Hybrid triage: Rule → BioBERT → ClinicalBERT → LLM fallback."""
    symptoms = state.get("symptoms", "")
    logger.info("[TRIAGE] patient: %s | symptoms: %.100s...", state.get("patient_id"), symptoms)

    try:
        specialist, secondary_needed, reasoning = await route_symptoms(symptoms)

        if specialist not in ("cardiology", "neurology", "cancer", "pathology"):
            specialist = "unknown"

        logger.info("[TRIAGE] → %s | secondary: %s | reason: %s", specialist, secondary_needed, reasoning)
        return {
            "assigned_specialist": specialist,
            "secondary_check_needed": secondary_needed,
            "messages": [
                f"[TRIAGE] Routing to '{specialist}'. "
                f"Secondary check: {secondary_needed}. Reason: {reasoning}"
            ],
        }
    except Exception as e:
        logger.error("[TRIAGE] Failed: %s", str(e))
        return {
            "assigned_specialist": "unknown",
            "secondary_check_needed": False,
            "messages": [f"[TRIAGE] Error during triage: {str(e)}"],
        }


async def specialist_node(state: AgentState) -> dict:
    """Dispatch to the appropriate specialist agent via HTTP (step 2.2)."""
    patient_id = state["patient_id"]
    symptoms = state["symptoms"]
    specialist = state["assigned_specialist"]
    retry = state.get("diagnosis_retry_count", 0)
    is_followup = retry > 0   # treat subsequent calls as follow-ups with refined context

    logger.info("[SPECIALIST] patient: %s | specialist: %s | retry: %d", patient_id, specialist, retry)

    try:
        if specialist == "cardiology":
            data = await call_cardiology_api(patient_id, symptoms, is_followup)
            agent_name = "Cardiology_Specialist"
        elif specialist == "neurology":
            data = await call_neurology_api(patient_id, symptoms, is_followup)
            agent_name = "Neurology_Specialist"
        elif specialist == "pathology":
            data = await call_pathology_api(patient_id, symptoms, is_followup)
            agent_name = "Pathology_Specialist"
        elif specialist == "cancer":
            data = await call_cancer_api(patient_id, symptoms, is_followup)
            agent_name = "Cancer_Oncology_Specialist"
        else:
            return {
                "requires_human_review": True,
                "human_review_reason": f"Unknown specialist '{specialist}' — cannot route.",
                "messages": [f"[SPECIALIST] Cannot route to unknown specialist: {specialist}"],
            }

        # Unwrap GenericResponse wrapper
        if not data.get("is_success") or not data.get("payload"):
            return {
                "requires_human_review": True,
                "human_review_reason": f"{specialist} agent returned failure or empty response.",
                "messages": [f"[{agent_name.upper()}] Response failed or empty."],
            }

        diagnosis = data["payload"].get("diagnosis")
        if not diagnosis:
            return {
                "requires_human_review": True,
                "human_review_reason": f"{specialist} returned no diagnosis in payload.",
                "messages": [f"[{agent_name.upper()}] No diagnosis in payload."],
            }

        severity = diagnosis.get("severity", "N/A")
        emergency = diagnosis.get("emergencyCareNeeded", "N/A")
        logger.info("[%s] severity: %s | emergency: %s", agent_name.upper(), severity, emergency)

        return {
            "specialist_diagnosis": diagnosis,
            "specialist_agent": agent_name,
            "messages": [
                f"[{agent_name.upper()}] Severity: {severity} | Emergency: {emergency}"
            ],
        }

    except Exception as e:
        logger.error("[SPECIALIST] Error: %s", str(e))
        return {
            "requires_human_review": True,
            "human_review_reason": f"{specialist} service unreachable: {str(e)}",
            "messages": [f"[SPECIALIST] Service error: {str(e)}"],
        }


async def secondary_check_node(state: AgentState) -> dict:
    """Call the pathology agent as a secondary lab cross-check."""
    patient_id = state["patient_id"]
    symptoms = state["symptoms"]

    logger.info("[SECONDARY_CHECK] Running pathology cross-check | patient: %s", patient_id)

    try:
        data = await call_pathology_api(patient_id, symptoms, is_followup=False)

        secondary_diagnosis = None
        if data.get("is_success") and data.get("payload"):
            secondary_diagnosis = data["payload"].get("diagnosis")

        if secondary_diagnosis:
            severity = secondary_diagnosis.get("severity", "N/A")
            logger.info("[SECONDARY_CHECK] Pathology severity: %s", severity)
            return {
                "secondary_diagnosis": secondary_diagnosis,
                "secondary_agent": "Pathology_Specialist",
                "secondary_check_done": True,
                "messages": [f"[PATHOLOGY_SECONDARY] Severity: {severity}"],
            }
        else:
            logger.warning("[SECONDARY_CHECK] Pathology returned empty — proceeding without secondary data")
            return {
                "secondary_diagnosis": None,
                "secondary_agent": None,
                "secondary_check_done": True,
                "messages": ["[PATHOLOGY_SECONDARY] Empty response — proceeding without secondary data"],
            }

    except Exception as e:
        logger.warning("[SECONDARY_CHECK] Unavailable (non-blocking): %s", str(e))
        return {
            "secondary_diagnosis": None,
            "secondary_agent": None,
            "secondary_check_done": True,
            "messages": [f"[PATHOLOGY_SECONDARY] Service unavailable: {str(e)}"],
        }


async def conflict_check_node(state: AgentState) -> dict:
    """LLM-based conflict detection between primary and secondary diagnoses."""
    primary = state.get("specialist_diagnosis")
    secondary = state.get("secondary_diagnosis")
    primary_agent = state.get("specialist_agent", "Primary Specialist")
    secondary_agent = state.get("secondary_agent", "Pathology Specialist")

    if not secondary:
        return {
            "conflict_detected": False,
            "conflict_reason": "No secondary diagnosis — conflict check skipped.",
            "messages": ["[CONFLICT_CHECK] No secondary data — skipped."],
        }

    primary_summary = _diagnosis_summary(primary or {})
    secondary_summary = _diagnosis_summary(secondary)
    primary_severity = (primary or {}).get("severity", "UNKNOWN")
    secondary_severity = secondary.get("severity", "UNKNOWN")

    logger.info(
        "[CONFLICT_CHECK] primary_severity: %s | secondary_severity: %s",
        primary_severity, secondary_severity,
    )

    try:
        result = _conflict_chain.invoke({
            "primary_agent": primary_agent,
            "primary_summary": primary_summary,
            "primary_severity": primary_severity,
            "secondary_agent": secondary_agent,
            "secondary_summary": secondary_summary,
            "secondary_severity": secondary_severity,
        })
        raw = _parse_json(result.content)
        conflict = bool(raw.get("conflict_detected", False))
        reason = raw.get("conflict_reason", "")

        if conflict:
            logger.warning("[CONFLICT_CHECK] Conflict detected: %s", reason)
            return {
                "conflict_detected": True,
                "conflict_reason": reason,
                "requires_human_review": True,
                "human_review_reason": (
                    f"Conflicting diagnoses between {primary_agent} and {secondary_agent}. "
                    f"Reason: {reason}"
                ),
                "messages": [f"[CONFLICT_CHECK] Conflict detected: {reason}"],
            }

        logger.info("[CONFLICT_CHECK] No conflict.")
        return {
            "conflict_detected": False,
            "conflict_reason": reason,
            "messages": [f"[CONFLICT_CHECK] No conflict. {reason}"],
        }

    except Exception as e:
        logger.error("[CONFLICT_CHECK] Error: %s — proceeding without conflict flag", str(e))
        return {
            "conflict_detected": False,
            "conflict_reason": f"Conflict check error: {str(e)}",
            "messages": [f"[CONFLICT_CHECK] Error: {str(e)} — proceeding"],
        }


async def xai_diagnosis_validator_node(state: AgentState) -> dict:
    """
    Validate specialist diagnosis via XAI service — retry loop #1 (step 2.3 / 2.4).
    On successful validation, saves diagnosis outcome to ChromaDB (step 2.4.1).
    On max retries, triggers human review (step 2.4.3).
    """
    patient_id = state["patient_id"]
    symptoms = state["symptoms"]
    diagnosis = state.get("specialist_diagnosis", {})
    specialist_agent = state.get("specialist_agent", "Unknown_Specialist")
    attempt = state.get("diagnosis_retry_count", 0) + 1  # 1-based display

    logger.info("[XAI_DIAGNOSIS] patient: %s | attempt: %d/%d", patient_id, attempt, MAX_RETRY_COUNT)

    try:
        xai_result = await call_validate_diagnosis(patient_id, symptoms, specialist_agent, diagnosis)
        payload = xai_result.get("payload", {}) if xai_result.get("is_success") else {}
        is_validated = payload.get("result", {}).get("is_validated", False)

        new_retry = attempt  # attempt already = old_count + 1

        if is_validated:
            logger.info("[XAI_DIAGNOSIS] Validated | patient: %s", patient_id)
            # Step 2.4.1: Save validated diagnosis to ChromaDB (fire-and-forget)
            asyncio.create_task(save_diagnosis_outcome(
                patient_id=patient_id,
                symptoms=symptoms,
                specialist_agent=specialist_agent,
                diagnosis=diagnosis,
            ))
            return {
                "diagnosis_validated": True,
                "diagnosis_retry_count": new_retry,
                "diagnosis_xai_result": payload,
                "messages": [
                    f"[XAI_DIAGNOSIS] Validated on attempt {attempt} — diagnosis saved to ChromaDB"
                ],
            }

        # Validation failed
        if new_retry >= MAX_RETRY_COUNT:
            # Step 2.4.3: Max retries — human intervention needed
            logger.warning("[XAI_DIAGNOSIS] Max retries reached | patient: %s", patient_id)
            return {
                "diagnosis_validated": False,
                "diagnosis_retry_count": new_retry,
                "diagnosis_xai_result": payload,
                "requires_human_review": True,
                "human_review_reason": (
                    f"Diagnosis failed XAI validation after {new_retry} attempts — "
                    "human intervention needed. "
                    f"Summary: {payload.get('result', {}).get('validation_summary', 'N/A')}"
                ),
                "messages": [
                    f"[XAI_DIAGNOSIS] Validation failed — max retries ({MAX_RETRY_COUNT}) reached. "
                    "Diagnosis failed, human intervention needed."
                ],
            }

        # Step 2.4.2: Retry specialist
        logger.info("[XAI_DIAGNOSIS] Validation failed | attempt: %d — retrying specialist", attempt)
        return {
            "diagnosis_validated": False,
            "diagnosis_retry_count": new_retry,
            "diagnosis_xai_result": payload,
            "messages": [
                f"[XAI_DIAGNOSIS] Validation failed (attempt {attempt}) — retrying specialist."
            ],
        }

    except Exception as e:
        new_retry = state.get("diagnosis_retry_count", 0) + 1
        logger.error("[XAI_DIAGNOSIS] Service error: %s", str(e))
        if new_retry >= MAX_RETRY_COUNT:
            return {
                "diagnosis_validated": False,
                "diagnosis_retry_count": new_retry,
                "diagnosis_xai_result": None,
                "requires_human_review": True,
                "human_review_reason": (
                    f"XAI diagnosis service error after {new_retry} attempts: {str(e)} — "
                    "human intervention needed."
                ),
                "messages": [
                    "[XAI_DIAGNOSIS] Service error — max retries reached. Human intervention needed."
                ],
            }
        return {
            "diagnosis_validated": False,
            "diagnosis_retry_count": new_retry,
            "diagnosis_xai_result": None,
            "messages": [f"[XAI_DIAGNOSIS] Service error: {str(e)} — retrying."],
        }


async def treatment_node(state: AgentState) -> dict:
    """Invoke the treatment agent to generate a care plan (step 2.4.1 / 2.5)."""
    patient_id = state["patient_id"]
    diagnosis = state.get("specialist_diagnosis", {})
    specialist_agent = state.get("specialist_agent", "Unknown_Specialist")
    retry = state.get("treatment_retry_count", 0)

    summary = _diagnosis_summary(diagnosis)
    severity = diagnosis.get("severity", "UNKNOWN")

    logger.info("[TREATMENT] patient: %s | severity: %s | retry: %d", patient_id, severity, retry)

    try:
        raw = await call_treatment_api(
            patient_id=patient_id,
            diagnosis=f"[{severity}] {summary}",
            specialist_notes=(
                f"Specialist: {specialist_agent}. "
                f"Hospitalization: {diagnosis.get('hospitalizationNeeded', 'N/A')}. "
                f"Emergency: {diagnosis.get('emergencyCareNeeded', 'N/A')}."
            ),
        )
        # Unwrap GenericResponse[TreatmentResponse] wrapper
        treatment_payload = None
        if raw.get("is_success") and raw.get("payload"):
            treatment_payload = raw["payload"]  # {agent, agent_id, treatment: {...}}
        else:
            treatment_payload = raw  # fallback: store as-is

        logger.info("[TREATMENT] Recommendation received | patient: %s", patient_id)
        urgency = (treatment_payload or {}).get("treatment", {}).get("urgency", "N/A")
        return {
            "treatment_recommendation": treatment_payload,
            "messages": [f"[TREATMENT] Care plan generated for patient {patient_id} | urgency: {urgency}"],
        }

    except Exception as e:
        logger.error("[TREATMENT] Error: %s", str(e))
        return {
            "treatment_recommendation": None,
            "messages": [f"[TREATMENT] Error: {str(e)}"],
        }


async def xai_treatment_validator_node(state: AgentState) -> dict:
    """
    Validate treatment recommendation via XAI service — retry loop #2 (step 2.5 / 2.6).
    On successful validation, saves treatment outcome to ChromaDB (step 2.6.1).
    On max retries, triggers human review (step 2.6.2).
    """
    patient_id = state["patient_id"]
    symptoms = state["symptoms"]
    diagnosis = state.get("specialist_diagnosis", {})
    treatment = state.get("treatment_recommendation") or {}
    attempt = state.get("treatment_retry_count", 0) + 1

    summary = _diagnosis_summary(diagnosis)
    severity = diagnosis.get("severity", "UNKNOWN")
    # Extract treatment plan text from the unwrapped TreatmentResponse payload
    treatment_obj = treatment.get("treatment", {})
    treatment_rec = treatment_obj.get("treatmentPlan", treatment.get("recommendation", ""))

    logger.info("[XAI_TREATMENT] patient: %s | attempt: %d/%d", patient_id, attempt, MAX_RETRY_COUNT)

    try:
        xai_result = await call_validate_treatment(
            patient_id=patient_id,
            specialist_agent=state.get("specialist_agent", "Unknown_Specialist"),
            diagnosis_summary=summary,
            severity=severity,
            treatment_recommendation=treatment_rec,
        )
        payload = xai_result.get("payload", {}) if xai_result.get("is_success") else {}
        is_validated = payload.get("result", {}).get("is_validated", False)

        new_retry = attempt

        if is_validated:
            logger.info("[XAI_TREATMENT] Validated | patient: %s", patient_id)
            # Step 2.6.1: Save validated treatment to ChromaDB (fire-and-forget)
            asyncio.create_task(save_treatment_outcome(
                patient_id=patient_id,
                symptoms=symptoms,
                specialist_agent=state.get("specialist_agent", "Unknown_Specialist"),
                diagnosis=diagnosis,
                treatment=treatment_obj,
            ))
            return {
                "treatment_validated": True,
                "treatment_retry_count": new_retry,
                "treatment_xai_result": payload,
                "messages": [
                    f"[XAI_TREATMENT] Validated on attempt {attempt} — treatment saved to ChromaDB"
                ],
            }

        if new_retry >= MAX_RETRY_COUNT:
            # Step 2.6.2: Max retries — human intervention needed
            logger.warning("[XAI_TREATMENT] Max retries reached | patient: %s", patient_id)
            return {
                "treatment_validated": False,
                "treatment_retry_count": new_retry,
                "treatment_xai_result": payload,
                "requires_human_review": True,
                "human_review_reason": (
                    f"Treatment recommendation failed XAI validation after {new_retry} attempts — "
                    "human intervention needed. "
                    f"Summary: {payload.get('result', {}).get('validation_summary', 'N/A')}"
                ),
                "messages": [
                    f"[XAI_TREATMENT] Validation failed — max retries ({MAX_RETRY_COUNT}) reached. "
                    "Treatment recommendation failed, human intervention needed."
                ],
            }

        # Step 2.6.2: Retry treatment agent
        logger.info("[XAI_TREATMENT] Validation failed | attempt: %d — retrying treatment", attempt)
        return {
            "treatment_validated": False,
            "treatment_retry_count": new_retry,
            "treatment_xai_result": payload,
            "messages": [
                f"[XAI_TREATMENT] Validation failed (attempt {attempt}) — retrying treatment."
            ],
        }

    except Exception as e:
        new_retry = state.get("treatment_retry_count", 0) + 1
        logger.error("[XAI_TREATMENT] Service error: %s", str(e))
        if new_retry >= MAX_RETRY_COUNT:
            return {
                "treatment_validated": False,
                "treatment_retry_count": new_retry,
                "treatment_xai_result": None,
                "requires_human_review": True,
                "human_review_reason": (
                    f"XAI treatment service error after {new_retry} attempts: {str(e)} — "
                    "human intervention needed."
                ),
                "messages": [
                    "[XAI_TREATMENT] Service error — max retries reached. Human intervention needed."
                ],
            }
        return {
            "treatment_validated": False,
            "treatment_retry_count": new_retry,
            "treatment_xai_result": None,
            "messages": [f"[XAI_TREATMENT] Service error: {str(e)} — retrying."],
        }


async def finish_node(state: AgentState) -> dict:
    """
    Assemble the final response and persist to MongoDB.
    Terminal node — reached on success, cache hit, or human-review.
    When chroma_cache_hit=True the final_response is already pre-filled.
    """
    patient_id = state.get("patient_id", "UNKNOWN")

    # Cache-hit path: final_response already built by chroma_lookup_node
    if state.get("chroma_cache_hit"):
        final_response = state.get("final_response") or {}
        logger.info(
            "[FINISH] Cache-hit path | patient: %s | status: %s",
            patient_id, final_response.get("status"),
        )
        asyncio.create_task(save_case(final_response))
        return {"final_response": final_response}

    requires_review = state.get("requires_human_review", False)
    diagnosis = state.get("specialist_diagnosis") or {}

    final_response = {
        "patient_id": patient_id,
        "status": "HUMAN_REVIEW_REQUIRED" if requires_review else "COMPLETED",
        "specialist_agent": state.get("specialist_agent"),
        "diagnosis": {
            "summary": _diagnosis_summary(diagnosis),
            "severity": diagnosis.get("severity", "UNKNOWN"),
            "emergency_care_needed": diagnosis.get("emergencyCareNeeded", "UNKNOWN"),
            "hospitalization_needed": diagnosis.get("hospitalizationNeeded", "UNKNOWN"),
            "full_details": diagnosis,
        } if diagnosis else None,
        "xai_diagnosis_validation": state.get("diagnosis_xai_result"),
        "treatment": state.get("treatment_recommendation"),
        "xai_treatment_validation": state.get("treatment_xai_result"),
        "conflict_detected": state.get("conflict_detected", False),
        "conflict_reason": state.get("conflict_reason", ""),
        "human_review_reason": state.get("human_review_reason", "") if requires_review else None,
        "audit_trail": state.get("messages", []),
    }

    status_label = "HUMAN_REVIEW_REQUIRED" if requires_review else "COMPLETED"
    logger.info("[FINISH] patient: %s | status: %s", patient_id, status_label)

    # Persist to MongoDB asynchronously (non-blocking)
    asyncio.create_task(save_case(final_response))

    return {"final_response": final_response}
