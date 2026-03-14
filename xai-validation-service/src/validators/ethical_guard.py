import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from log.logger import logger

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# -- Diagnosis Validation ----------------------------------------------------

_DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Clinical Safety Validator with expertise in medical diagnosis review.
Evaluate whether the specialist AI diagnosis is clinically sound and safe.

Check:
1. Does the stated severity match the described symptoms?
2. Is the emergency care decision appropriate for the symptoms and severity?
3. Are there dangerous oversights, missed diagnoses, or contradictions?
4. Is the diagnosis clinically plausible given the patient's symptoms?

Respond ONLY with valid JSON (no markdown fences):
{{
    "is_validated": true,
    "confidence_score": 0.85,
    "validation_summary": "brief summary within 100 words",
    "key_concerns": ["concern1"],
    "recommendation": "APPROVE"
}}
recommendation must be one of: APPROVE, REJECT, REVIEW"""),
    ("human", """Specialist: {specialist_agent}
Patient Symptoms: {symptoms}
Diagnosis Summary: {diagnosis_summary}
Severity: {severity}
Emergency Care Needed: {emergency_care_needed}
Hospitalization Needed: {hospitalization_needed}"""),
])

_diagnosis_chain = _DIAGNOSIS_PROMPT | llm

# -- Treatment Validation ----------------------------------------------------

_TREATMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Clinical Safety Validator with expertise in treatment review.
Evaluate whether the treatment recommendation is clinically appropriate and safe.

Check:
1. Is the treatment proportional and appropriate for the diagnosis?
2. Are there dangerous or inappropriate medication recommendations?
3. Does the urgency of care match the stated severity?
4. Is the treatment plan clinically sound and evidence-based?

Respond ONLY with valid JSON (no markdown fences):
{{
    "is_validated": true,
    "confidence_score": 0.85,
    "validation_summary": "brief summary within 100 words",
    "key_concerns": ["concern1"],
    "recommendation": "APPROVE"
}}
recommendation must be one of: APPROVE, REJECT, REVIEW"""),
    ("human", """Diagnosis Summary: {diagnosis_summary}
Severity: {severity}
Treatment Recommendation: {treatment_recommendation}"""),
])

_treatment_chain = _TREATMENT_PROMPT | llm


# -- Public helpers ----------------------------------------------------------

def validate_diagnosis(specialist_agent: str, symptoms: str, diagnosis: dict) -> dict:
    """LLM-based clinical safety validation of a specialist diagnosis."""
    diagnosis_summary = (
        diagnosis.get("diagnosysDetails") or   # cardiology (typo preserved)
        diagnosis.get("diagnosisDetails") or   # neurology
        diagnosis.get("analysisDetails") or    # pathology
        str(diagnosis)
    )
    severity = diagnosis.get("severity", "UNKNOWN")
    emergency = diagnosis.get("emergencyCareNeeded", "UNKNOWN")
    hospitalization = diagnosis.get("hospitalizationNeeded", "UNKNOWN")

    logger.debug("LLM diagnosis validation | specialist: %s | severity: %s", specialist_agent, severity)

    result = _diagnosis_chain.invoke({
        "specialist_agent": specialist_agent,
        "symptoms": symptoms,
        "diagnosis_summary": diagnosis_summary,
        "severity": severity,
        "emergency_care_needed": emergency,
        "hospitalization_needed": hospitalization,
    })
    raw = _parse_json(result.content)
    logger.debug("LLM diagnosis verdict: %s | confidence: %s", raw.get("recommendation"), raw.get("confidence_score"))
    return raw


def validate_treatment(diagnosis_summary: str, severity: str, treatment_recommendation: str) -> dict:
    """LLM-based clinical safety validation of a treatment recommendation."""
    logger.debug("LLM treatment validation | severity: %s", severity)

    result = _treatment_chain.invoke({
        "diagnosis_summary": diagnosis_summary,
        "severity": severity,
        "treatment_recommendation": treatment_recommendation,
    })
    raw = _parse_json(result.content)
    logger.debug("LLM treatment verdict: %s | confidence: %s", raw.get("recommendation"), raw.get("confidence_score"))
    return raw


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())
