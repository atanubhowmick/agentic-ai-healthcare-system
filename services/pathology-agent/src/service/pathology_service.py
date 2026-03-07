import json
from agent.pathology_agent import pathology_executor
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import LLMInvocationException, LLMResponseParseException
from constant.constants import PATHOLOGY_AGENT_ID
from log.logger import logger


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    if request.is_followup:
        query = request.symptoms
        logger.debug("Follow-up query for patient %s: %s", request.patient_id, query)
    else:
        query = (
            f"Analyze lab results for patient {request.patient_id}. "
            f"Details: {request.symptoms}. "
            "Identify any abnormalities in biomarkers and respond strictly in the requested JSON format."
        )
        logger.debug("Initial query for patient %s: %s", request.patient_id, query)

    try:
        logger.debug("Invoking pathology executor for patient: %s", request.patient_id)
        result = pathology_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": request.patient_id}}
        )
        logger.debug("LLM response received for patient %s | content length: %d chars",
                     request.patient_id, len(result.content))
    except Exception as e:
        raise LLMInvocationException(message=f"LLM call failed for patient {request.patient_id}: {e}")

    try:
        raw = _parse_llm_json(result.content)
        diagnosis = DiagnosisResult(**raw)
        logger.debug("LLM response parsed successfully for patient %s | severity: %s",
                     request.patient_id, diagnosis.severity)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise LLMResponseParseException(message=f"Failed to parse LLM response for patient {request.patient_id}: {e}")

    return DiagnosisResponse(
        agent="Pathology_Specialist",
        agent_id=PATHOLOGY_AGENT_ID,
        diagnosis=diagnosis
    )
