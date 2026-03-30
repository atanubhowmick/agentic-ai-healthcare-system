"""
XAI Validator Service - orchestrates validation via the DeepAgent.

Validation flow:
  1. Run deterministic rule checks (fast pre-filter, no LLM required).
  2. Pre-execute explainability tool directly — guaranteed execution, results
     stored in ContextVar for collection after agent call.
  3. Build the patient query string with pre-computed tool results injected.
  4. Invoke the DeepAgent for final reasoning (single LLM call, no tool-call overhead).
  5. Apply constitutional guard (critique + optional revision, with guideline RAG).
  6. Parse JSON, attach SHAP factors and validation metadata, return response.
"""

import json
import time
from langchain_core.messages import SystemMessage, HumanMessage

from agent.xai_agent import xai_executor, BASE_SYSTEM
from datamodel.models import (
    DiagnosisValidationRequest, TreatmentValidationRequest,
    ValidationResult, ValidationResponse, GenericResponse,
)
from validators.medical_rules import check_emergency_consistency, check_severity_validity
from explainers.shap_provider import DiagnosisExplainer
from explainers import context as explanation_context
from guardrails import constitutional_guard
from exception.exceptions import ValidationSvcException
from core.config import XAI_AGENT_ID, OPENAI_MODEL
from log.logger import logger


# -- Helpers -------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
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


def _format_factors_for_prompt(factors: list[dict]) -> str:
    """Format SHAP/LLM factors as a readable string for prompt injection."""
    if not factors:
        return "No explainability factors could be determined."
    lines = [
        f"{i}. {f.get('factor', 'Unknown')} | importance: {f.get('importance', 0):.2f} | {f.get('direction', 'neutral')}"
        for i, f in enumerate(factors, start=1)
    ]
    return "\n".join(lines)


# -- Main entry points ---------------------------------------------------------

def run_diagnosis_validation(request: DiagnosisValidationRequest) -> GenericResponse:
    logger.debug(
        "[XAI_SVC] validate-diagnosis | patient: %s | specialist: %s",
        request.patient_id, request.specialist_agent,
    )

    diagnosis = request.diagnosis
    severity = diagnosis.get("severity", "UNKNOWN")
    emergency = diagnosis.get("emergencyCareNeeded", "UNKNOWN")
    rules_triggered: list[str] = []

    # Step 1: Deterministic rule pre-filter
    rule_ok, rule_msg = check_emergency_consistency(request.symptoms, severity, emergency)
    if not rule_ok:
        rules_triggered.append("check_emergency_consistency")
        logger.warning("[XAI_SVC] Rule check failed | patient: %s | %s", request.patient_id, rule_msg)
        return _make_rejection(request.patient_id, "DIAGNOSIS", rule_msg, rules_triggered)

    diagnosis_summary = (
        diagnosis.get("diagnosysDetails") or   # cardiology (typo preserved)
        diagnosis.get("diagnosisDetails") or   # oncology / neurology
        diagnosis.get("analysisDetails") or    # pathology
        str(diagnosis)
    )

    # Step 2: Pre-execute explainability — guaranteed before LLM reasoning
    explanation_context.clear()
    factors: list[dict] = []
    explainability_method = ""
    try:
        explainer = DiagnosisExplainer()
        factors = explainer.explain_diagnosis(request.symptoms, diagnosis_summary)
        explainability_method = explainer.last_method
        explanation_context.set_factors(factors)
        explanation_context.set_method(explainability_method)
        logger.debug(
            "[XAI_SVC] Pre-computed %d explainability factor(s) via %s | patient: %s",
            len(factors), explainability_method, request.patient_id,
        )
    except Exception as exc:
        logger.warning("[XAI_SVC] Explainability pre-computation failed: %s", exc)

    # Step 3: Build query with pre-computed results injected
    query = (
        f"Validate the following specialist diagnosis for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Patient Symptoms: {request.symptoms}\n"
        f"Diagnosis Summary: {diagnosis_summary}\n"
        f"Severity: {severity}\n"
        f"Emergency Care Needed: {emergency}\n"
        f"Hospitalization Needed: {diagnosis.get('hospitalizationNeeded', 'UNKNOWN')}\n"
        f"Rule Check: PASSED — emergency consistency verified.\n"
        f"Explainability Factors (pre-computed):\n{_format_factors_for_prompt(factors)}\n"
        "Respond strictly in the requested JSON format."
    )

    # Step 4: Invoke agent for final reasoning
    t_start = time.perf_counter()
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
    latency_ms = (time.perf_counter() - t_start) * 1000

    # Step 5: Constitutional guard (includes guideline RAG for P5)
    content, critique = constitutional_guard.apply(
        raw_response=content,
        symptoms=request.symptoms,
        severity=severity,
        emergency_care=emergency,
    )
    if critique:
        logger.info("[XAI_SVC] Constitutional guard revised response | patient: %s", request.patient_id)

    # Step 6: Parse, attach metadata, return
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
        "[XAI_SVC] validate-diagnosis complete | patient: %s | validated: %s | latency: %.0fms",
        request.patient_id, result.is_validated, latency_ms,
    )
    return GenericResponse.success(response)


def run_treatment_validation(request: TreatmentValidationRequest) -> GenericResponse:
    logger.debug("[XAI_SVC] validate-treatment | patient: %s", request.patient_id)

    rules_triggered: list[str] = []

    # Step 1: Deterministic rule pre-filter
    sev_ok, sev_msg = check_severity_validity(request.severity)
    if not sev_ok:
        rules_triggered.append("check_severity_validity")
        return _make_rejection(request.patient_id, "TREATMENT", sev_msg, rules_triggered)

    # Step 2: Build query
    query = (
        f"Validate the following treatment recommendation for patient {request.patient_id}.\n"
        f"Specialist Agent: {request.specialist_agent}\n"
        f"Diagnosis Summary: {request.diagnosis_summary}\n"
        f"Severity: {request.severity}\n"
        f"Treatment Recommendation: {request.treatment_recommendation}\n"
        f"Rule Check: PASSED — severity validity verified.\n"
        "Respond strictly in the requested JSON format."
    )

    # Step 3: Invoke agent
    t_start = time.perf_counter()
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
    latency_ms = (time.perf_counter() - t_start) * 1000

    # Step 4: Parse, attach metadata, return
    try:
        raw = _parse_llm_json(content)
        result = ValidationResult(
            **{k: v for k, v in raw.items() if k in ValidationResult.model_fields},
            validator_latency_ms=round(latency_ms, 1),
            model_used=OPENAI_MODEL,
            rules_triggered=rules_triggered,
        )
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
        "[XAI_SVC] validate-treatment complete | patient: %s | validated: %s | latency: %.0fms",
        request.patient_id, result.is_validated, latency_ms,
    )
    return GenericResponse.success(response)
