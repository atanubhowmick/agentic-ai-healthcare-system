"""
Treatment Service - orchestrates care plan generation via the DeepAgent.

Treatment flow:
  1. Build the patient query string.
  2. Invoke the DeepAgent directly using SystemMessage / HumanMessage.
     - Per-session chat history is maintained here in _session_store.
  3. Parse the JSON response and return TreatmentResponse.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.treatment_agent import treatment_executor, BASE_SYSTEM
from datamodel.models import TreatmentRequest, TreatmentResult, TreatmentResponse
from exception.exceptions import TreatmentSvcException
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

def recommend(request: TreatmentRequest) -> TreatmentResponse:
    patient_id = request.patient_id

    if request.is_followup:
        query = request.diagnosis   # follow-up message reuses the diagnosis field as free-text input
        logger.debug("Follow-up query | patient: %s", patient_id)
    else:
        query = (
            f"Patient {patient_id} has received the following diagnosis: {request.diagnosis}. "
            f"Specialist notes: {request.specialist_notes}. "
            "Create a comprehensive treatment and patient care plan. "
            "Respond strictly in the requested JSON format."
        )
        logger.debug("Initial treatment query | patient: %s", patient_id)

    # -- Build messages with session history -----------------------------------
    history: list = _session_store.setdefault(patient_id, [])
    history.append(HumanMessage(content=query))

    logger.debug(
        "[TREATMENT_SVC] Invoking DeepAgent | patient: %s | history_turns: %d",
        patient_id, len(history),
    )

    # -- Invoke DeepAgent ------------------------------------------------------
    try:
        result = treatment_executor.invoke({
            "messages": [SystemMessage(content=BASE_SYSTEM)] + history
        })
    except Exception as e:
        raise TreatmentSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {patient_id}: {e}",
        )

    last_msg = result["messages"][-1]
    _session_store[patient_id] = result["messages"]

    logger.debug(
        "[TREATMENT_SVC] Agent response | patient: %s | length: %d chars",
        patient_id, len(last_msg.content),
    )

    # -- Parse response --------------------------------------------------------
    try:
        raw = _parse_llm_json(last_msg.content)
        treatment = TreatmentResult(**raw)
        logger.debug(
            "[TREATMENT_SVC] Parsed | patient: %s | urgency: %s",
            patient_id, treatment.urgency,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise TreatmentSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )

    return TreatmentResponse(
        agent="Treatment_Care_Agent",
        agent_id="TREAT-AGENT-1004",
        treatment=treatment,
    )
