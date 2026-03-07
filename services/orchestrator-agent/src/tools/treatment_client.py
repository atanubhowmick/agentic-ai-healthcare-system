import httpx
from core.config import TREATMENT_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


async def call_treatment_api(patient_id: str, diagnosis: str, specialist_notes: str) -> dict:
    """POST /recommend to the treatment agent and return the response dict."""
    payload = {
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "specialist_notes": specialist_notes,
    }
    url = f"{TREATMENT_SERVICE_URL}/treatment-agent/recommend"
    logger.debug("[treatment_client] POST %s | patient: %s", url, patient_id)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
