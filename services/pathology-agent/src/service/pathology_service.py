import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.pathology_agent import pathology_executor, BASE_SYSTEM
from core.config import AGENT_NAME, AGENT_ID
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import PathologySvcException
from log.logger import logger


# In-memory session history keyed by patient_id
_session_store: dict[str, list] = {}


def _build_query(request: DiagnosisRequest) -> str:
    if request.is_followup:
        logger.debug("Follow-up query | patient: %s", request.patient_id)
        return request.symptoms
    logger.debug("Initial query | patient: %s", request.patient_id)
    return (
        f"Analyze lab results for patient {request.patient_id}. "
        f"Details: {request.symptoms}. "
        "Identify any abnormalities in biomarkers and respond strictly in the requested JSON format."
    )


def _invoke_agent(patient_id: str, history: list) -> str:
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
    logger.debug("Agent response | patient: %s | length: %d chars", patient_id, len(last_msg.content))
    return last_msg.content


def _parse_diagnosis(patient_id: str, content: str) -> DiagnosisResult:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        raw = json.loads(content.strip())
        diagnosis = DiagnosisResult(**raw)
        logger.debug("Parsed diagnosis | patient: %s | severity: %s", patient_id, diagnosis.severity)
        return diagnosis
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise PathologySvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )


async def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    patient_id = request.patient_id
    query = _build_query(request)

    history = _session_store.setdefault(patient_id, [])
    history.append(HumanMessage(content=query))
    logger.debug("Invoking DeepAgent | patient: %s | history_turns: %d", patient_id, len(history))

    content = _invoke_agent(patient_id, history)
    diagnosis = _parse_diagnosis(patient_id, content)

    return DiagnosisResponse(agent=AGENT_NAME, agent_id=AGENT_ID, diagnosis=diagnosis)
