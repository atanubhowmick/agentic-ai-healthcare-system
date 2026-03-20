import httpx
from langchain.tools import tool
from core.config import XAI_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


@tool
async def call_validate_diagnosis(
    patient_id: str,
    symptoms: str,
    specialist_agent: str,
    diagnosis: dict,
) -> dict:
    """Call the XAI Validator service to validate a specialist diagnosis for clinical safety.
    POST /xai-validator/validate-diagnosis and return the parsed response dict.

    Args:
        patient_id: Unique patient identifier.
        symptoms: Original patient symptom description.
        specialist_agent: Name of the specialist agent that produced the diagnosis.
        diagnosis: Full diagnosis dict returned by the specialist agent.
    """
    payload = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "specialist_agent": specialist_agent,
        "diagnosis": diagnosis,
    }
    url = f"{XAI_SERVICE_URL}/validate-diagnosis"
    logger.debug("[xai_client] POST %s | patient: %s | specialist: %s", url, patient_id, specialist_agent)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@tool
async def call_validate_treatment(
    patient_id: str,
    specialist_agent: str,
    diagnosis_summary: str,
    severity: str,
    treatment_recommendation: str,
) -> dict:
    """Call the XAI Validator service to validate a treatment recommendation for clinical safety.
    POST /xai-validator/validate-treatment and return the parsed response dict.

    Args:
        patient_id: Unique patient identifier.
        specialist_agent: Name of the specialist agent that produced the diagnosis.
        diagnosis_summary: Text summary of the specialist diagnosis.
        severity: Severity level from the diagnosis (LOW/HIGH/CRITICAL).
        treatment_recommendation: Treatment plan text to be validated.
    """
    payload = {
        "patient_id": patient_id,
        "specialist_agent": specialist_agent,
        "diagnosis_summary": diagnosis_summary,
        "severity": severity,
        "treatment_recommendation": treatment_recommendation,
    }
    url = f"{XAI_SERVICE_URL}/validate-treatment"
    logger.debug("[xai_client] POST %s | patient: %s", url, patient_id)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
