import httpx
from langchain.tools import tool
from core.config import TREATMENT_SERVICE_URL, HTTP_TIMEOUT
from log.logger import logger


@tool
async def call_treatment_api(patient_id: str, diagnosis: str, specialist_notes: str) -> dict:
    """Call the Treatment & Patient Care agent to generate a comprehensive care plan.
    POST /treatment-agent/recommend and return the response dict.

    Args:
        patient_id: Unique patient identifier.
        diagnosis: Formatted diagnosis string including severity and summary.
        specialist_notes: Additional notes from the specialist agent.
    """
    payload = {
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "specialist_notes": specialist_notes,
    }
    url = f"{TREATMENT_SERVICE_URL}/recommend"
    logger.debug("[treatment_client] POST %s | patient: %s", url, patient_id)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
