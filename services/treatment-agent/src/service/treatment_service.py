import json
from agent.treatment_agent import treatment_executor
from datamodel.models import TreatmentRequest, TreatmentResult, TreatmentResponse
from exception.exceptions import TreatmentSvcException
from log.logger import logger


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def recommend(request: TreatmentRequest) -> TreatmentResponse:
    if request.is_followup:
        query = request.diagnosis   # follow-up message reuses the diagnosis field as free-text input
        logger.debug("Follow-up query for patient %s: %s", request.patient_id, query)
    else:
        query = (
            f"Patient {request.patient_id} has received the following diagnosis: {request.diagnosis}. "
            f"Specialist notes: {request.specialist_notes}. "
            "Create a comprehensive treatment and patient care plan. "
            "Respond strictly in the requested JSON format."
        )
        logger.debug("Initial treatment query for patient %s", request.patient_id)

    try:
        logger.debug("Invoking treatment executor for patient: %s", request.patient_id)
        result = treatment_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": request.patient_id}}
        )
        logger.debug(
            "LLM response received for patient %s | content length: %d chars",
            request.patient_id, len(result.content),
        )
    except Exception as e:
        raise TreatmentSvcException(
            error_code="LLM_INVOCATION_ERROR", message=f"LLM call failed for patient {request.patient_id}: {e}"
        )

    try:
        raw = _parse_llm_json(result.content)
        treatment = TreatmentResult(**raw)
        logger.debug(
            "LLM response parsed successfully for patient %s | urgency: %s",
            request.patient_id, treatment.urgency,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise TreatmentSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR", message=f"Failed to parse LLM response for patient {request.patient_id}: {e}"
        )

    return TreatmentResponse(
        agent="Treatment_Care_Agent",
        agent_id="TREAT-AGENT-1004",
        treatment=treatment,
    )
