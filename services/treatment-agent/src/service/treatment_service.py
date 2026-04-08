import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.treatment_agent import treatment_executor, BASE_SYSTEM
from core.config import AGENT_NAME, AGENT_ID
from datamodel.models import TreatmentRequest, TreatmentResult, TreatmentResponse
from exception.exceptions import TreatmentSvcException
from log.logger import logger


# In-memory session history keyed by patient_id
_session_store: dict[str, list] = {}


def _build_query(request: TreatmentRequest) -> str:
    if request.is_followup:
        # follow-up reuses the diagnosis field as free-text input
        logger.debug("Follow-up query | patient: %s", request.patient_id)
        return request.diagnosis
    logger.debug("Initial treatment query | patient: %s", request.patient_id)
    return (
        f"Patient {request.patient_id} has received the following diagnosis: {request.diagnosis}. "
        f"Specialist notes: {request.specialist_notes}. "
        "Create a comprehensive treatment and patient care plan. "
        "Respond strictly in the requested JSON format."
    )


def _invoke_agent(patient_id: str, history: list) -> str:
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
    logger.debug("Agent response | patient: %s | length: %d chars", patient_id, len(last_msg.content))
    return last_msg.content


def _parse_treatment(patient_id: str, content: str) -> TreatmentResult:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        raw = json.loads(content.strip())
        treatment = TreatmentResult(**raw)
        logger.debug("Parsed treatment | patient: %s | urgency: %s", patient_id, treatment.urgency)
        return treatment
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise TreatmentSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )


async def recommend(request: TreatmentRequest) -> TreatmentResponse:
    patient_id = request.patient_id
    query = _build_query(request)

    history = _session_store.setdefault(patient_id, [])
    history.append(HumanMessage(content=query))
    logger.debug("Invoking DeepAgent | patient: %s | history_turns: %d", patient_id, len(history))

    content = _invoke_agent(patient_id, history)
    treatment = _parse_treatment(patient_id, content)

    return TreatmentResponse(agent=AGENT_NAME, agent_id=AGENT_ID, treatment=treatment)
