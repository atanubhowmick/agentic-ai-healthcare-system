import json
from agent.cancer_agent import cancer_executor
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import CancerSvcException
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
            f"Analyze patient {request.patient_id} with symptoms: {request.symptoms}. "
            "Evaluate for oncological conditions and respond strictly in the requested JSON format."
        )
        logger.debug("Initial query for patient %s: %s", request.patient_id, query)

    try:
        logger.debug("Invoking cancer executor for patient: %s", request.patient_id)
        result = cancer_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": request.patient_id}},
        )
        logger.debug(
            "LLM response received for patient %s | content length: %d chars",
            request.patient_id, len(result.content),
        )
    except Exception as e:
        raise CancerSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"LLM call failed for patient {request.patient_id}: {e}",
        )

    try:
        raw = _parse_llm_json(result.content)
        diagnosis = DiagnosisResult(**raw)
        logger.debug(
            "LLM response parsed for patient %s | severity: %s | cancer type: %s",
            request.patient_id, diagnosis.severity, diagnosis.suspectedCancerType,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise CancerSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse LLM response for patient {request.patient_id}: {e}",
        )

    return DiagnosisResponse(
        agent="Cancer_Oncology_Specialist",
        agent_id="CANCER-AGENT-1004",
        diagnosis=diagnosis,
    )
