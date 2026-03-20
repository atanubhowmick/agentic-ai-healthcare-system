import httpx
from langchain.tools import tool
from core.config import CANCER_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


@tool
async def call_cancer_api(patient_id: str, symptoms: str, is_followup: bool = False) -> dict:
    """Call the Cancer Oncology specialist agent to diagnose oncological symptoms.
    POST /cancer-agent/diagnose and return the parsed response dict.

    Args:
        patient_id: Unique patient identifier.
        symptoms: Patient symptom description text.
        is_followup: True if this is a follow-up query for an existing session.
    """
    payload = {"patient_id": patient_id, "symptoms": symptoms, "is_followup": is_followup}
    url = f"{CANCER_SERVICE_URL}/diagnose"
    logger.debug("[cancer_client] POST %s | patient: %s | followup: %s", url, patient_id, is_followup)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
