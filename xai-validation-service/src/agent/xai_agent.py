from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from explainers.shap_provider import DiagnosisExplainer
from explainers import context as explanation_context
from log.logger import logger


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

    "IMPORTANT — Severity scale used in this system is derived from MIMIC clinical admission patterns: "
    "LOW = routine or elective discharge (patient stable, no emergency admission required); "
    "HIGH = hospital admission required; "
    "CRITICAL = ICU or emergency intervention required. "
    "This applies to ALL specialist diagnoses (cardiology, neurology, oncology, pathology). "
    "LOW severity means the condition does not require emergency admission — it does NOT mean "
    "the underlying condition is clinically insignificant. Do NOT reject a diagnosis solely "
    "because serious chronic or oncological symptoms are paired with LOW severity — this is "
    "clinically valid for outpatient or elective cases. "

    "UNDERTRIAGE RULE: If the diagnosis itself describes an acute condition — including but not "
    "limited to STEMI, ACS, stroke, TIA, sepsis, septic shock, respiratory failure, pulmonary "
    "embolism, aortic dissection, meningitis, status epilepticus, haemodynamic instability, "
    "altered consciousness, or any condition the specialist labels as requiring immediate "
    "intervention — AND the diagnosis assigns severity=LOW with emergencyCareNeeded=NO, "
    "this is an undertriage error regardless of the MIMIC severity label. Flag as REJECT. "
    "Apply this rule based on the clinical content of the diagnosis, not just symptom keywords. "
    "ADDITIONAL SIGNAL: If the diagnosisDetails states 'Original admission type: EMERGENCY', "
    "this means the patient was admitted as an emergency. A severity=LOW with emergencyCareNeeded=NO "
    "assignment for an emergency admission is a strong undertriage signal — flag as REJECT or REVIEW. "

    "DECISION ANCHORING: Base your recommendation primarily on the structured diagnosis fields "
    "(severity, emergencyCareNeeded, hospitalizationNeeded) and the clinical content of the "
    "diagnosisDetails. Treat the free-text symptom description as secondary context only. "
    "Two descriptions of the same clinical condition must produce the same recommendation. "

    "SUMMARY STYLE: Write validation_summary in plain language a nurse or junior doctor can read quickly. "
    "Maximum 2 short sentences. Use common English words — replace long Latin or Greek medical terms "
    "with plain alternatives where possible (e.g. 'heart attack' not 'myocardial infarction', "
    "'low oxygen' not 'hypoxia', 'fits' not 'seizures'). State the key finding and the safety conclusion only. "

    "KEY CONCERNS: Always populate key_concerns with 1–2 specific clinical observations. "
    "For APPROVE: state what was checked and found safe (e.g. 'Emergency care correctly flagged', "
    "'Severity consistent with outpatient oncology presentation'). "
    "For REJECT or REVIEW: state the specific clinical concern. "
    "Never leave key_concerns empty. "

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


@tool
def get_validation_response_schema() -> str:
    """Returns the expected JSON response schema for a validation output."""
    return _JSON_SCHEMA


@tool
def check_emergency_consistency(symptoms: str, severity: str, emergency_care: str) -> str:
    """Check that the emergency care decision is consistent with the patient symptoms and severity.
    Returns a consistency verdict and explanation. Always call this for diagnosis validation.

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
    Always call this for treatment validation.

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
    """Identify the top contributing clinical factors for a diagnosis decision using SHAP or LLM.
    Always call this when validating a diagnosis to provide explainability context.

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


logger.debug("Initializing XAI Validation LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, seed=42)

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
