import httpx
from core.config import NEUROLOGY_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


async def call_neurology_api(patient_id: str, symptoms: str, is_followup: bool = False) -> dict:
    """POST /neurology-agent/diagnose and return the parsed response dict."""
    payload = {"patient_id": patient_id, "symptoms": symptoms, "is_followup": is_followup}
    url = f"{NEUROLOGY_SERVICE_URL}/diagnose"
    logger.debug("[neurology_client] POST %s | patient: %s | followup: %s", url, patient_id, is_followup)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
