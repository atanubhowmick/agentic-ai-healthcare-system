import httpx
from langchain.tools import tool
from core.config import NEUROLOGY_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


@tool
async def call_neurology_api(patient_id: str, symptoms: str, is_followup: bool = False) -> dict:
    """Call the Neurology specialist agent to diagnose neurological symptoms.
    POST /neurology-agent/diagnose and return the parsed response dict.

    Args:
        patient_id: Unique patient identifier.
        symptoms: Patient symptom description text.
        is_followup: True if this is a follow-up query for an existing session.
    """
    payload = {"patient_id": patient_id, "symptoms": symptoms, "is_followup": is_followup}
    url = f"{NEUROLOGY_SERVICE_URL}/diagnose"
    logger.debug("[neurology_client] POST %s | patient: %s | followup: %s", url, patient_id, is_followup)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
