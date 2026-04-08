import json
import time
from langchain_core.messages import SystemMessage, HumanMessage

from agent.xai_agent import xai_executor, BASE_SYSTEM
from core.config import XAI_AGENT_ID, OPENAI_MODEL
from datamodel.models import (
    DiagnosisValidationRequest, TreatmentValidationRequest,
    ValidationResult, ValidationResponse, GenericResponse,
)
from validators.medical_rules import check_emergency_consistency, check_severity_validity
from rules.rule_engine import evaluate as rule_evaluate
from explainers.shap_provider import DiagnosisExplainer
from explainers import context as explanation_context
from guardrails import constitutional_guard
from exception.exceptions import ValidationSvcException
from log.logger import logger


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(content.strip())


def _make_rejection(
    patient_id: str,
    validation_type: str,
    reason: str,
    rules_triggered: list[str] | None = None,
) -> GenericResponse:
    result = ValidationResult(
        is_validated=False,
        confidence_score=0.0,
        validation_summary=reason,
        key_concerns=[reason],
        recommendation="REJECT",
        model_used=OPENAI_MODEL,
        rules_triggered=rules_triggered or [],
    )
    response = ValidationResponse(
        agent="XAI_Validator", agent_id=XAI_AGENT_ID,
        patient_id=patient_id, validation_type=validation_type, result=result,
    )
    return GenericResponse.success(response)


def _invoke_agent(query: str) -> str:
    """Invoke the DeepAgent with a single-turn message and return the last message content."""
    result = xai_executor.invoke({
        "messages": [SystemMessage(content=BASE_SYSTEM), HumanMessage(content=query)]
    })
    return result["messages"][-1].content


def _format_factors_for_prompt(factors: list[dict]) -> str:
    if not factors:
        return "No explainability factors could be determined."
    return "\n".join(
        f"{i}. {f.get('factor', 'Unknown')} | importance: {f.get('importance', 0):.2f} | {f.get('direction', 'neutral')}"
        for i, f in enumerate(factors, start=1)
    )


def _run_diagnosis_rule_checks(
    request: DiagnosisValidationRequest,
    rules_triggered: list[str],
    severity: str,
    emergency: str,
    diagnosis_summary: str,
) -> tuple[GenericResponse | None, str | None]:
    """Steps 1-2: medical rules pre-filter + rule engine. Returns (early_rejection, review_warning)."""
    rule_ok, rule_msg = check_emergency_consistency(request.symptoms, severity, emergency)
    if not rule_ok:
        rules_triggered.append("check_emergency_consistency")
        logger.warning("Rule check failed | patient: %s | %s", request.patient_id, rule_msg)
        return _make_rejection(request.patient_id, "DIAGNOSIS", rule_msg, rules_triggered), None

    rule_passed, rule_action, rule_reason, triggered_ids = rule_evaluate(
        symptoms=request.symptoms, severity=severity, emergency_care=emergency,
        diagnosis_text=diagnosis_summary, treatment_text="",
    )
    if not rule_passed:
        rules_triggered.extend(triggered_ids)
        if rule_action == "REJECT":
            logger.warning("Rule engine REJECT | patient: %s | rules: %s", request.patient_id, triggered_ids)
            return _make_rejection(request.patient_id, "DIAGNOSIS", rule_reason, rules_triggered), None
        review_warning = f"Rule {triggered_ids[0] if triggered_ids else 'unknown'}: {rule_reason}"
        logger.info("Rule engine REVIEW | patient: %s | %s", request.patient_id, review_warning)
        return None, review_warning

    return None, None


def _pre_compute_explainability(symptoms: str, diagnosis_summary: str) -> tuple[list, str]:
    """Run SHAP/LLM explainability before the agent call. Returns (factors, method)."""
    explanation_context.clear()
    try:
        explainer = DiagnosisExplainer()
        factors = explainer.explain_diagnosis(symptoms, diagnosis_summary)
        explanation_context.set_factors(factors)
        explanation_context.set_method(explainer.last_method)
        return factors, explainer.last_method
    except Exception as exc:
        logger.warning("Explainability pre-computation failed: %s", exc)
        return [], ""


def _parse_diagnosis_result(
    patient_id: str,
    content: str,
    critique: str | None,
    latency_ms: float,
    rules_triggered: list[str],
) -> ValidationResult:
    """Parse LLM JSON, attach SHAP factors and validation metadata."""
    try:
        raw = _parse_llm_json(content)
        if critique:
            raw.setdefault("key_concerns", [])
            raw["key_concerns"] = [f"[Constitutional revision] {critique[:120]}"] + raw["key_concerns"]
        result = ValidationResult(
            **{k: v for k, v in raw.items() if k in ValidationResult.model_fields},
            explanation_factors=explanation_context.get_factors(),
            validator_latency_ms=round(latency_ms, 1),
            model_used=OPENAI_MODEL,
            explainability_method=explanation_context.get_method(),
            rules_triggered=rules_triggered,
            constitutional_revised=critique is not None,
        )
        logger.debug("Parsed | patient: %s | recommendation: %s", patient_id, result.recommendation)
        return result
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValidationSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )


def _run_treatment_rule_checks(
    request: TreatmentValidationRequest,
    rules_triggered: list[str],
) -> tuple[GenericResponse | None, str | None]:
    """Steps 1-2: severity validity + rule engine. Returns (early_rejection, review_warning)."""
    sev_ok, sev_msg = check_severity_validity(request.severity)
    if not sev_ok:
        rules_triggered.append("check_severity_validity")
        return _make_rejection(request.patient_id, "TREATMENT", sev_msg, rules_triggered), None

    rule_passed, rule_action, rule_reason, triggered_ids = rule_evaluate(
        symptoms=request.diagnosis_summary, severity=request.severity,
        emergency_care="NO",  # treatment requests don't carry an emergency_care field
        diagnosis_text=request.diagnosis_summary,
        treatment_text=request.treatment_recommendation,
    )
    if not rule_passed:
        rules_triggered.extend(triggered_ids)
        if rule_action == "REJECT":
            logger.warning("Rule engine REJECT (treatment) | patient: %s | rules: %s", request.patient_id, triggered_ids)
            return _make_rejection(request.patient_id, "TREATMENT", rule_reason, rules_triggered), None
        review_warning = f"Rule {triggered_ids[0] if triggered_ids else 'unknown'}: {rule_reason}"
        logger.info("Rule engine REVIEW (treatment) | patient: %s | %s", request.patient_id, review_warning)
        return None, review_warning

    return None, None


def _parse_treatment_result(
    patient_id: str,
    content: str,
    latency_ms: float,
    rules_triggered: list[str],
) -> ValidationResult:
    """Parse LLM JSON and attach validation metadata."""
    try:
        raw = _parse_llm_json(content)
        result = ValidationResult(
            **{k: v for k, v in raw.items() if k in ValidationResult.model_fields},
            validator_latency_ms=round(latency_ms, 1),
            model_used=OPENAI_MODEL,
            rules_triggered=rules_triggered,
        )
        logger.debug("Parsed | patient: %s | recommendation: %s", patient_id, result.recommendation)
        return result
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValidationSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )


async def run_diagnosis_validation(request: DiagnosisValidationRequest) -> GenericResponse:
    logger.debug("validate-diagnosis | patient: %s | specialist: %s", request.patient_id, request.specialist_agent)

    diagnosis = request.diagnosis
    severity = diagnosis.get("severity", "UNKNOWN")
    emergency = diagnosis.get("emergencyCareNeeded", "UNKNOWN")
    rules_triggered: list[str] = []

    diagnosis_summary = (
        diagnosis.get("diagnosisDetails") or
        diagnosis.get("analysisDetails") or
        str(diagnosis)
    )

    rejection, review_warning = _run_diagnosis_rule_checks(
        request, rules_triggered, severity, emergency, diagnosis_summary
    )
    if rejection:
        return rejection

    factors, method = _pre_compute_explainability(request.symptoms, diagnosis_summary)
    logger.debug("Pre-computed %d explainability factor(s) via %s | patient: %s", len(factors), method, request.patient_id)

    query = (
        f"Validate the following specialist diagnosis for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Patient Symptoms: {request.symptoms}\n"
        f"Diagnosis Summary: {diagnosis_summary}\n"
        f"Severity: {severity}\n"
        f"Emergency Care Needed: {emergency}\n"
        f"Hospitalization Needed: {diagnosis.get('hospitalizationNeeded', 'UNKNOWN')}\n"
        f"Rule Check: PASSED — emergency consistency verified.\n"
    )
    if review_warning:
        query += f"⚠ Rule Engine Safety Concern: {review_warning}\n"
    query += (
        f"Explainability Factors (pre-computed):\n{_format_factors_for_prompt(factors)}\n"
        "Respond strictly in the requested JSON format."
    )

    t_start = time.perf_counter()
    try:
        content = _invoke_agent(query)
        logger.debug("Agent response | patient: %s | length: %d chars", request.patient_id, len(content))
    except Exception as e:
        raise ValidationSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {request.patient_id}: {e}",
        )
    latency_ms = (time.perf_counter() - t_start) * 1000

    content, critique = constitutional_guard.apply(
        raw_response=content, symptoms=request.symptoms,
        severity=severity, emergency_care=emergency,
    )
    if critique:
        logger.info("Constitutional guard revised response | patient: %s", request.patient_id)

    result = _parse_diagnosis_result(request.patient_id, content, critique, latency_ms, rules_triggered)

    response = ValidationResponse(
        agent="XAI_Validator", agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id, validation_type="DIAGNOSIS", result=result,
    )
    logger.debug(
        "validate-diagnosis complete | patient: %s | validated: %s | latency: %.0fms",
        request.patient_id, result.is_validated, latency_ms,
    )
    return GenericResponse.success(response)


async def run_treatment_validation(request: TreatmentValidationRequest) -> GenericResponse:
    logger.debug("validate-treatment | patient: %s", request.patient_id)
    rules_triggered: list[str] = []

    rejection, review_warning = _run_treatment_rule_checks(request, rules_triggered)
    if rejection:
        return rejection

    query = (
        f"Validate the following treatment recommendation for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Diagnosis Summary: {request.diagnosis_summary}\n"
        f"Severity: {request.severity}\n"
        f"Treatment Recommendation: {request.treatment_recommendation}\n"
        f"Rule Check: PASSED — severity validity verified.\n"
    )
    if review_warning:
        query += f"⚠ Rule Engine Safety Concern: {review_warning}\n"
    query += "Respond strictly in the requested JSON format."

    t_start = time.perf_counter()
    try:
        content = _invoke_agent(query)
        logger.debug("Agent response | patient: %s | length: %d chars", request.patient_id, len(content))
    except Exception as e:
        raise ValidationSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {request.patient_id}: {e}",
        )
    latency_ms = (time.perf_counter() - t_start) * 1000

    result = _parse_treatment_result(request.patient_id, content, latency_ms, rules_triggered)

    response = ValidationResponse(
        agent="XAI_Validator", agent_id=XAI_AGENT_ID,
        patient_id=request.patient_id, validation_type="TREATMENT", result=result,
    )
    logger.debug(
        "validate-treatment complete | patient: %s | validated: %s | latency: %.0fms",
        request.patient_id, result.is_validated, latency_ms,
    )
    return GenericResponse.success(response)
