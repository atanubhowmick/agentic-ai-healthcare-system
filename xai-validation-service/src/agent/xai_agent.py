"""
XAI Validation Agent - DeepAgent-based implementation.

Architecture:
  - Uses deepagents.create_deep_agent (built on LangGraph) as the executor.
  - @tool decorator exposes validation schema, medical rule checks, and explainability.
  - The agent autonomously validates diagnoses and treatment recommendations.
  - Session history and message construction are handled by validator_service.py.

Public interface (used by validator_service.py):
  xai_executor  - the raw DeepAgent instance
  BASE_SYSTEM   - system prompt (used by service to build messages)
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from explainers.shap_provider import DiagnosisExplainer
from explainers import context as explanation_context
from log.logger import logger


# -- JSON response schema -------------------------------------------------------

_JSON_SCHEMA = """
{
    "is_validated": true or false,
    "confidence_score": 0.0 to 1.0,
    "validation_summary": "brief clinical safety assessment within 100 words",
    "key_concerns": ["clinical concern 1", "clinical concern 2"],
    "recommendation": "APPROVE or REJECT or REVIEW"
}"""

BASE_SYSTEM = (
    "You are a Clinical Safety Validator AI Agent. Your goal is to validate specialist AI "
    "diagnoses and treatment recommendations for clinical safety, consistency, and ethical soundness. "

    # Severity scale context
    "IMPORTANT — Severity scale used in this system is derived from MIMIC clinical admission patterns: "
    "LOW = routine or elective discharge (patient stable, no emergency admission required); "
    "HIGH = hospital admission required; "
    "CRITICAL = ICU or emergency intervention required. "
    "This applies to ALL specialist diagnoses (cardiology, neurology, oncology, pathology). "
    "LOW severity means the condition does not require emergency admission — it does NOT mean "
    "the underlying condition is clinically insignificant. Do NOT reject a diagnosis solely "
    "because serious chronic or oncological symptoms are paired with LOW severity — this is "
    "clinically valid for outpatient or elective cases. "

    # Fix 1 — broadened undertriage exception (principle-based, not symptom-list-based)
    "UNDERTRIAGE RULE: If the diagnosis itself describes an acute condition — including but not "
    "limited to STEMI, ACS, stroke, TIA, sepsis, septic shock, respiratory failure, pulmonary "
    "embolism, aortic dissection, meningitis, status epilepticus, haemodynamic instability, "
    "altered consciousness, or any condition the specialist labels as requiring immediate "
    "intervention — AND the diagnosis assigns severity=LOW with emergencyCareNeeded=NO, "
    "this is an undertriage error regardless of the MIMIC severity label. Flag as REJECT. "
    "Apply this rule based on the clinical content of the diagnosis, not just symptom keywords. "

    # Fix 3 — anchor decision on structured fields, not symptom phrasing
    "DECISION ANCHORING: Base your recommendation primarily on the structured diagnosis fields "
    "(severity, emergencyCareNeeded, hospitalizationNeeded) and the clinical content of the "
    "diagnosisDetails. Treat the free-text symptom description as secondary context only. "
    "Two descriptions of the same clinical condition must produce the same recommendation. "

    # Fix 4 — readability: target both length and vocabulary
    "SUMMARY STYLE: Write validation_summary in plain language a nurse or junior doctor can read quickly. "
    "Maximum 2 short sentences. Use common English words — replace long Latin or Greek medical terms "
    "with plain alternatives where possible (e.g. 'heart attack' not 'myocardial infarction', "
    "'low oxygen' not 'hypoxia', 'fits' not 'seizures'). State the key finding and the safety conclusion only. "

    "For diagnosis validation: check that the emergency care decision is appropriate, "
    "no dangerous oversights or contradictions exist, and the diagnosis is clinically plausible. "
    "Always call check_emergency_consistency and explain_diagnosis_factors tools when validating a diagnosis. "
    "For treatment validation: check the treatment is proportional to the diagnosis, medications are safe, "
    "urgency matches severity, and the plan is evidence-based. "
    "Always call check_severity_validity when validating a treatment recommendation. "
    "recommendation must be one of: APPROVE, REJECT, REVIEW. "
    "confidence_score must be a float between 0.0 and 1.0. "
    "Provide the response strictly in the following JSON format:" + _JSON_SCHEMA
)


# -- Tools ----------------------------------------------------------------------

@tool
def get_validation_response_schema() -> str:
    """Return the required JSON response schema for validation output.
    Call this tool whenever you need a reminder of the exact JSON format expected."""
    return _JSON_SCHEMA


@tool
def check_emergency_consistency(symptoms: str, severity: str, emergency_care: str) -> str:
    """Check that the emergency care decision is consistent with the patient symptoms and severity.
    Returns a consistency verdict and explanation. Always call this tool for diagnosis validation.

    Args:
        symptoms: Patient symptoms or clinical description text.
        severity: Severity level from the specialist diagnosis (LOW/HIGH/CRITICAL).
        emergency_care: Emergency care flag from the specialist diagnosis (YES/NO).
    """
    from validators.medical_rules import check_emergency_consistency as _check
    is_ok, message = _check(symptoms, severity, emergency_care)
    status = "CONSISTENT" if is_ok else "INCONSISTENT"
    logger.debug("[XAI_TOOL] Emergency consistency: %s | %s", status, message)
    return f"Status: {status}\nMessage: {message}"


@tool
def check_severity_validity(severity: str) -> str:
    """Validate that the severity value is a recognised clinical level (LOW/HIGH/CRITICAL).
    Returns a validity verdict. Always call this tool for treatment validation.

    Args:
        severity: Severity level string to validate.
    """
    from validators.medical_rules import check_severity_validity as _check
    is_ok, message = _check(severity)
    status = "VALID" if is_ok else "INVALID"
    logger.debug("[XAI_TOOL] Severity validity: %s | %s", status, message)
    return f"Status: {status}\nMessage: {message}"


@tool
def explain_diagnosis_factors(symptoms: str, diagnosis_summary: str) -> str:
    """Identify the top contributing clinical factors for a diagnosis decision.
    Returns a ranked list of factors with importance scores and direction.
    Always call this tool when validating a diagnosis to provide explainability context.

    Args:
        symptoms: Patient symptoms or clinical presentation text.
        diagnosis_summary: Summary of the specialist diagnosis.
    """
    explainer = DiagnosisExplainer()
    factors = explainer.explain_diagnosis(symptoms, diagnosis_summary)
    if not factors:
        explanation_context.set_factors([])
        explanation_context.set_method("LLM_FALLBACK")
        return "No explainability factors could be determined."
    explanation_context.set_factors(factors)
    explanation_context.set_method(explainer.last_method)
    lines = [
        f"{i}. {f.get('factor', 'Unknown')} | importance: {f.get('importance', 0):.2f} | {f.get('direction', 'neutral')}"
        for i, f in enumerate(factors, start=1)
    ]
    logger.info("[XAI_TOOL] Explainability: %d factor(s) identified", len(factors))
    return "\n".join(lines)


# -- LLM ------------------------------------------------------------------------

logger.debug("Initializing XAI Validation LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, seed=42)


# -- DeepAgent ------------------------------------------------------------------

logger.debug("Building XAI Validation DeepAgent")
xai_executor = create_deep_agent(
    model=_llm,
    tools=[
        get_validation_response_schema,
        check_emergency_consistency,
        check_severity_validity,
        explain_diagnosis_factors,
    ],
    system_prompt=BASE_SYSTEM,
)
logger.debug("XAI Validation DeepAgent ready")
