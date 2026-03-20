import httpx
from langchain.tools import tool
from core.config import CARDIOLOGY_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


@tool
async def call_cardiology_api(patient_id: str, symptoms: str, is_followup: bool = False) -> dict:
    """Call the Cardiology specialist agent to diagnose cardiac symptoms.
    POST /cardiology-agent/diagnose and return the parsed response dict.

    Args:
        patient_id: Unique patient identifier.
        symptoms: Patient symptom description text.
        is_followup: True if this is a follow-up query for an existing session.
    """
    payload = {"patient_id": patient_id, "symptoms": symptoms, "is_followup": is_followup}
    url = f"{CARDIOLOGY_SERVICE_URL}/diagnose"
    logger.debug("[cardiology_client] POST %s | patient: %s | followup: %s", url, patient_id, is_followup)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
