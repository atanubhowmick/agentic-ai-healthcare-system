import json
from datamodel.models import (
    DiagnosisValidationRequest, TreatmentValidationRequest,
    ValidationResult, ValidationResponse, GenericResponse,
)
from validators.medical_rules import check_emergency_consistency, check_severity_validity
from validators.ethical_guard import validate_diagnosis, validate_treatment
from explainers.shap_provider import DiagnosisExplainer
from exception.exceptions import ValidationSvcException
from constants import XAI_AGENT_ID
from log.logger import logger

_explainer = DiagnosisExplainer()


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


def run_diagnosis_validation(request: DiagnosisValidationRequest) -> GenericResponse:
    logger.debug(
        "validate-diagnosis | patient: %s | specialist: %s",
        request.patient_id, request.specialist_agent,
    )

    diagnosis = request.diagnosis
    severity = diagnosis.get("severity", "UNKNOWN")
    emergency = diagnosis.get("emergencyCareNeeded", "UNKNOWN")

    rule_ok, rule_msg = check_emergency_consistency(request.symptoms, severity, emergency)
    if not rule_ok:
        logger.warning("Rule check failed | patient: %s | %s", request.patient_id, rule_msg)
        return _make_rejection(request.patient_id, "DIAGNOSIS", rule_msg)

    try:
        raw = validate_diagnosis(request.specialist_agent, request.symptoms, diagnosis)
    except json.JSONDecodeError as e:
        raise ValidationSvcException(error_code="VALIDATION_PARSE_ERROR", message=f"LLM response parse error: {e}")
    except Exception as e:
        raise ValidationSvcException(error_code="VALIDATION_LLM_ERROR", message=f"LLM validation failed: {e}")

    diagnosis_summary = (
        diagnosis.get("diagnosysDetails") or
        diagnosis.get("diagnosisDetails") or
        diagnosis.get("analysisDetails") or
        str(diagnosis)
    )
    top_factors = _explainer.explain_diagnosis(request.symptoms, diagnosis_summary)
    logger.debug("Top factors for patient %s: %s", request.patient_id, top_factors)

    result = ValidationResult(
        is_validated=raw["is_validated"],
        confidence_score=raw.get("confidence_score", 0.0),
        validation_summary=raw.get("validation_summary", ""),
        key_concerns=raw.get("key_concerns", []),
        recommendation=raw.get("recommendation", "REVIEW"),
    )
    response = ValidationResponse(
        agent="XAI_Validator",
        agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id,
        validation_type="DIAGNOSIS",
        result=result,
    )
    logger.debug(
        "validate-diagnosis complete | patient: %s | validated: %s",
        request.patient_id, result.is_validated,
    )
    return GenericResponse.success(response)


def run_treatment_validation(request: TreatmentValidationRequest) -> GenericResponse:
    logger.debug("validate-treatment | patient: %s", request.patient_id)

    sev_ok, sev_msg = check_severity_validity(request.severity)
    if not sev_ok:
        return _make_rejection(request.patient_id, "TREATMENT", sev_msg)

    try:
        raw = validate_treatment(
            request.diagnosis_summary, request.severity, request.treatment_recommendation
        )
    except json.JSONDecodeError as e:
        raise ValidationSvcException(error_code="VALIDATION_PARSE_ERROR", message=f"LLM response parse error: {e}")
    except Exception as e:
        raise ValidationSvcException(error_code="VALIDATION_LLM_ERROR", message=f"LLM validation failed: {e}")

    result = ValidationResult(
        is_validated=raw["is_validated"],
        confidence_score=raw.get("confidence_score", 0.0),
        validation_summary=raw.get("validation_summary", ""),
        key_concerns=raw.get("key_concerns", []),
        recommendation=raw.get("recommendation", "REVIEW"),
    )
    response = ValidationResponse(
        agent="XAI_Validator",
        agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id,
        validation_type="TREATMENT",
        result=result,
    )
    logger.debug(
        "validate-treatment complete | patient: %s | validated: %s",
        request.patient_id, result.is_validated,
    )
    return GenericResponse.success(response)
