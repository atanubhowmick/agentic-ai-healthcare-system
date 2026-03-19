"""
Pathology Service - orchestrates diagnosis via the DeepAgent.

Diagnosis flow:
  1. Build the patient query string.
  2. Invoke the DeepAgent directly using SystemMessage / HumanMessage.
     - Per-session chat history is maintained here in _session_store.
  3. Parse the JSON response and return DiagnosisResponse.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.pathology_agent import pathology_executor, BASE_SYSTEM
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import PathologySvcException
from constant.constants import PATHOLOGY_AGENT_ID
from log.logger import logger


# -- Per-session chat history --------------------------------------------------

_session_store: dict[str, list] = {}


# -- Helpers -------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


# -- Main entry point ----------------------------------------------------------

def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    patient_id = request.patient_id

    if request.is_followup:
        query = request.symptoms
        logger.debug("Follow-up query | patient: %s", patient_id)
    else:
        query = (
            f"Analyze lab results for patient {patient_id}. "
            f"Details: {request.symptoms}. "
            "Identify any abnormalities in biomarkers and respond strictly in the requested JSON format."
        )
        logger.debug("Initial query | patient: %s", patient_id)

    # -- Build messages with session history -----------------------------------
    history: list = _session_store.setdefault(patient_id, [])
    history.append(HumanMessage(content=query))

    logger.debug(
        "[PATHO_SVC] Invoking DeepAgent | patient: %s | history_turns: %d",
        patient_id, len(history),
    )

    # -- Invoke DeepAgent ------------------------------------------------------
    try:
        result = pathology_executor.invoke({
            "messages": [SystemMessage(content=BASE_SYSTEM)] + history
        })
    except Exception as e:
        raise PathologySvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {patient_id}: {e}",
        )

    last_msg = result["messages"][-1]
    _session_store[patient_id] = result["messages"]

    logger.debug(
        "[PATHO_SVC] Agent response | patient: %s | length: %d chars",
        patient_id, len(last_msg.content),
    )

    # -- Parse response --------------------------------------------------------
    try:
        raw = _parse_llm_json(last_msg.content)
        diagnosis = DiagnosisResult(**raw)
        logger.debug(
            "[PATHO_SVC] Parsed | patient: %s | severity: %s",
            patient_id, diagnosis.severity,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise PathologySvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )

    return DiagnosisResponse(
        agent="Pathology_Specialist",
        agent_id=PATHOLOGY_AGENT_ID,
        diagnosis=diagnosis,
    )
