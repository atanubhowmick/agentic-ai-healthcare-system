"""
XAI Validator Service - orchestrates validation via the DeepAgent.

Validation flow:
  1. Run deterministic rule checks (fast pre-filter, no LLM required).
  2. Build the patient query string with all clinical context.
  3. Invoke the DeepAgent using SystemMessage / HumanMessage.
     - Each validation is stateless; no session history is maintained.
  4. Parse the JSON response and return ValidationResponse.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.xai_agent import xai_executor, BASE_SYSTEM
from datamodel.models import (
    DiagnosisValidationRequest, TreatmentValidationRequest,
    ValidationResult, ValidationResponse, GenericResponse,
)
from validators.medical_rules import check_emergency_consistency, check_severity_validity
from exception.exceptions import ValidationSvcException
from core.config import XAI_AGENT_ID
from log.logger import logger


# -- Helpers -------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def _make_rejection(patient_id: str, validation_type: str, reason: str) -> GenericResponse:
    result = ValidationResult(
        is_validated=False,
        confidence_score=0.0,
        validation_summary=reason,
        key_concerns=[reason],
        recommendation="REJECT",
    )
    response = ValidationResponse(
        agent="XAI_Validator",
        agent_id=XAI_AGENT_ID,
        patient_id=patient_id,
        validation_type=validation_type,
        result=result,
    )
    return GenericResponse.success(response)


def _invoke_agent(query: str) -> str:
    """Invoke the DeepAgent with a single-turn message and return the last message content."""
    result = xai_executor.invoke({
        "messages": [SystemMessage(content=BASE_SYSTEM), HumanMessage(content=query)]
    })
    return result["messages"][-1].content


# -- Main entry points ---------------------------------------------------------

def run_diagnosis_validation(request: DiagnosisValidationRequest) -> GenericResponse:
    logger.debug(
        "[XAI_SVC] validate-diagnosis | patient: %s | specialist: %s",
        request.patient_id, request.specialist_agent,
    )

    diagnosis = request.diagnosis
    severity = diagnosis.get("severity", "UNKNOWN")
    emergency = diagnosis.get("emergencyCareNeeded", "UNKNOWN")

    # Fast deterministic pre-filter before invoking the agent
    rule_ok, rule_msg = check_emergency_consistency(request.symptoms, severity, emergency)
    if not rule_ok:
        logger.warning("[XAI_SVC] Rule check failed | patient: %s | %s", request.patient_id, rule_msg)
        return _make_rejection(request.patient_id, "DIAGNOSIS", rule_msg)

    diagnosis_summary = (
        diagnosis.get("diagnosysDetails") or   # cardiology (typo preserved)
        diagnosis.get("diagnosisDetails") or   # oncology / neurology
        diagnosis.get("analysisDetails") or    # pathology
        str(diagnosis)
    )

    query = (
        f"Validate the following specialist diagnosis for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Patient Symptoms: {request.symptoms}\n"
        f"Diagnosis Summary: {diagnosis_summary}\n"
        f"Severity: {severity}\n"
        f"Emergency Care Needed: {emergency}\n"
        f"Hospitalization Needed: {diagnosis.get('hospitalizationNeeded', 'UNKNOWN')}\n"
        "Respond strictly in the requested JSON format."
    )

    try:
        content = _invoke_agent(query)
        logger.debug(
            "[XAI_SVC] Agent response | patient: %s | length: %d chars",
            request.patient_id, len(content),
        )
    except Exception as e:
        raise ValidationSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {request.patient_id}: {e}",
        )

    try:
        raw = _parse_llm_json(content)
        result = ValidationResult(**raw)
        logger.debug(
            "[XAI_SVC] Parsed | patient: %s | recommendation: %s",
            request.patient_id, result.recommendation,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValidationSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {request.patient_id}: {e}",
        )

    response = ValidationResponse(
        agent="XAI_Validator",
        agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id,
        validation_type="DIAGNOSIS",
        result=result,
    )
    logger.debug(
        "[XAI_SVC] validate-diagnosis complete | patient: %s | validated: %s",
        request.patient_id, result.is_validated,
    )
    return GenericResponse.success(response)


def run_treatment_validation(request: TreatmentValidationRequest) -> GenericResponse:
    logger.debug("[XAI_SVC] validate-treatment | patient: %s", request.patient_id)

    # Fast deterministic pre-filter before invoking the agent
    sev_ok, sev_msg = check_severity_validity(request.severity)
    if not sev_ok:
        return _make_rejection(request.patient_id, "TREATMENT", sev_msg)

    query = (
        f"Validate the following treatment recommendation for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Diagnosis Summary: {request.diagnosis_summary}\n"
        f"Severity: {request.severity}\n"
        f"Treatment Recommendation: {request.treatment_recommendation}\n"
        "Respond strictly in the requested JSON format."
    )

    try:
        content = _invoke_agent(query)
        logger.debug(
            "[XAI_SVC] Agent response | patient: %s | length: %d chars",
            request.patient_id, len(content),
        )
    except Exception as e:
        raise ValidationSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {request.patient_id}: {e}",
        )

    try:
        raw = _parse_llm_json(content)
        result = ValidationResult(**raw)
        logger.debug(
            "[XAI_SVC] Parsed | patient: %s | recommendation: %s",
            request.patient_id, result.recommendation,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValidationSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {request.patient_id}: {e}",
        )

    response = ValidationResponse(
        agent="XAI_Validator",
        agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id,
        validation_type="TREATMENT",
        result=result,
    )
    logger.debug(
        "[XAI_SVC] validate-treatment complete | patient: %s | validated: %s",
        request.patient_id, result.is_validated,
    )
    return GenericResponse.success(response)
