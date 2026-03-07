import json
from agent.neurology_agent import neurology_executor
from datamodel.models import NeurologyRequest, NeurologyResult, NeurologyResponse
from exception.exceptions import NeurologySvcException
from log.logger import logger


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def diagnose(request: NeurologyRequest) -> NeurologyResponse:
    if request.is_followup:
        query = request.symptoms
        logger.debug("Follow-up query for patient %s: %s", request.patient_id, query)
    else:
        query = (
            f"Analyze patient {request.patient_id} with neurological symptoms: {request.symptoms}. "
            "Assess for neurological anomalies and respond strictly in the requested JSON format."
        )
        logger.debug("Initial query for patient %s: %s", request.patient_id, query)

    try:
        logger.debug("Invoking neurology executor for patient: %s", request.patient_id)
        result = neurology_executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": request.patient_id}}
        )
        logger.debug("LLM response received for patient %s | content length: %d chars",
                     request.patient_id, len(result.content))
    except Exception as e:
        raise NeurologySvcException(error_code="LLM_INVOCATION_ERROR", message=f"LLM call failed for patient {request.patient_id}: {e}")

    try:
        raw = _parse_llm_json(result.content)
        diagnosis = NeurologyResult(**raw)
        logger.debug("LLM response parsed successfully for patient %s | severity: %s",
                     request.patient_id, diagnosis.severity)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise NeurologySvcException(error_code="LLM_RESPONSE_PARSE_ERROR", message=f"Failed to parse LLM response for patient {request.patient_id}: {e}")

    return NeurologyResponse(
        agent="Neurology_Specialist",
        agent_id="NEURO-AGENT-1002",
        diagnosis=diagnosis
    )
