from fastapi import APIRouter

from datamodel.models import (
    TfidfBaselineRequest, XaiEvaluationRequest, EvaluationStatusResponse, GenericResponse,
)
from exception.exceptions import EvaluationSvcException
from log.logger import logger
from service import evaluation_service

router = APIRouter(prefix="/evaluation-service")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "evaluation-service"}


@router.post("/evaluate/cancer-agent", response_model=GenericResponse[dict])
async def trigger_tfidf_evaluation(request: TfidfBaselineRequest) -> GenericResponse[dict]:
    """
    Trigger a Cancer Agent evaluation (80:20 train/test split by default).
    Runs in a background thread; poll /evaluate/cancer-agent/status for progress.
    """
    logger.debug("[API] POST /evaluate/cancer-agent | max_cases=%s | test_size=%.0f%%",
                 request.max_cases or "all", request.test_size * 100)
    evaluation_service.start_cancer_agent_evaluation(max_cases=request.max_cases, test_size=request.test_size)
    return GenericResponse.success({
        "status":    "started",
        "max_cases": request.max_cases or "all",
        "test_size": request.test_size,
    })


@router.get("/evaluate/cancer-agent/status", response_model=GenericResponse[EvaluationStatusResponse])
async def get_tfidf_status() -> GenericResponse[EvaluationStatusResponse]:
    """Returns whether a Cancer Agent evaluation is running and whether a report is available."""
    status = evaluation_service.get_cancer_agent_report_status()
    return GenericResponse.success(EvaluationStatusResponse(**status))


@router.get("/evaluate/cancer-agent/report", response_model=GenericResponse[dict])
async def get_tfidf_report() -> GenericResponse[dict]:
    """Return the most recent Cancer Agent evaluation report."""
    logger.debug("[API] GET /evaluate/cancer-agent/report")
    report = evaluation_service.get_cancer_agent_report()
    if report is None:
        raise EvaluationSvcException("CANCER_AGENT_REPORT_NOT_FOUND", "No Cancer Agent evaluation report available yet.")
    return GenericResponse.success(report)



@router.post("/evaluate/xai", response_model=GenericResponse[dict])
async def trigger_xai_evaluation(request: XaiEvaluationRequest) -> GenericResponse[dict]:
    """
    Trigger an XAI evaluation measuring decision accuracy, safety net effectiveness,
    rule coverage, over-rejection rate, and XAI quality metrics (stability, fidelity,
    consistency, sparsity, interpretability).
    Runs in a background thread; poll /evaluate/xai/status for progress.
    """
    logger.debug(
        "[API] POST /evaluate/xai | max_cases=%s | max_undertriage=%d",
        request.max_cases or "all", request.max_undertriage_cases,
    )
    evaluation_service.start_xai_evaluation(
        max_cases=request.max_cases,
        max_correct_cases=request.max_correct_cases,
        max_undertriage_cases=request.max_undertriage_cases,
        max_stability_cases=request.max_stability_cases,
        max_fidelity_cases=request.max_fidelity_cases,
        max_consistency_cases=request.max_consistency_cases,
    )
    return GenericResponse.success({
        "status":                 "started",
        "max_cases":              request.max_cases or "all",
        "max_correct_cases":      request.max_correct_cases,
        "max_undertriage_cases":  request.max_undertriage_cases,
        "max_stability_cases":    request.max_stability_cases,
        "max_fidelity_cases":     request.max_fidelity_cases,
        "max_consistency_cases":  request.max_consistency_cases,
    })


@router.get("/evaluate/xai/status", response_model=GenericResponse[EvaluationStatusResponse])
async def get_xai_status() -> GenericResponse[EvaluationStatusResponse]:
    """Returns whether an XAI evaluation is running and whether a report is available."""
    status = evaluation_service.get_xai_status()
    return GenericResponse.success(EvaluationStatusResponse(**status))


@router.get("/evaluate/xai/report", response_model=GenericResponse[dict])
async def get_xai_report() -> GenericResponse[dict]:
    """Return the most recent XAI evaluation report."""
    logger.debug("[API] GET /evaluate/xai/report")
    report = evaluation_service.get_xai_report()
    if report is None:
        raise EvaluationSvcException("XAI_REPORT_NOT_FOUND", "No XAI evaluation report available yet.")
    return GenericResponse.success(report)
