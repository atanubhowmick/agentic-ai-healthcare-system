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


# ---------------------------------------------------------------------------
# TF-IDF baseline endpoints
# ---------------------------------------------------------------------------

@router.post("/evaluate/tfidf-baseline", response_model=GenericResponse[dict])
async def trigger_tfidf_evaluation(request: TfidfBaselineRequest) -> GenericResponse[dict]:
    """
    Trigger a TF-IDF baseline evaluation.
    Trains TF-IDF + classifier on (1-test_size) fraction of MIMIC eval cases and
    reports metrics on the held-out test_size fraction (default 80:20 split).
    Runs in a background thread; poll /evaluate/tfidf-baseline/status for progress.
    """
    logger.debug("[API] POST /evaluate/tfidf-baseline | max_cases=%s | test_size=%.0f%%",
                 request.max_cases or "all", request.test_size * 100)
    evaluation_service.start_tfidf_evaluation(max_cases=request.max_cases, test_size=request.test_size)
    return GenericResponse.success({
        "status":    "started",
        "max_cases": request.max_cases or "all",
        "test_size": request.test_size,
    })


@router.get("/evaluate/tfidf-baseline/status", response_model=GenericResponse[EvaluationStatusResponse])
async def get_tfidf_status() -> GenericResponse[EvaluationStatusResponse]:
    """Returns whether a TF-IDF evaluation is running and whether a report is available."""
    status = evaluation_service.get_tfidf_status()
    return GenericResponse.success(EvaluationStatusResponse(**status))


@router.get("/evaluate/tfidf-baseline/report", response_model=GenericResponse[dict])
async def get_tfidf_report() -> GenericResponse[dict]:
    """Return the most recent TF-IDF baseline evaluation report."""
    logger.debug("[API] GET /evaluate/tfidf-baseline/report")
    report = evaluation_service.get_tfidf_report()
    if report is None:
        raise EvaluationSvcException("TFIDF_REPORT_NOT_FOUND", "No TF-IDF baseline report available yet.")
    return GenericResponse.success(report)


# ---------------------------------------------------------------------------
# XAI evaluation endpoints
# ---------------------------------------------------------------------------

@router.post("/evaluate/xai", response_model=GenericResponse[dict])
async def trigger_xai_evaluation(request: XaiEvaluationRequest) -> GenericResponse[dict]:
    """
    Trigger an XAI validation service evaluation.

    Builds correct and under-triage diagnosis payloads from MIMIC evaluation cases
    and calls the XAI service to measure:
      - Option 1: Validation decision accuracy (approval rate for correct diagnoses)
      - Option 2: Safety net effectiveness (under-triage detection sensitivity)
      - Option 4: Rule engine coverage (% handled by deterministic rules vs LLM)
      - Option 6: Over-rejection rate (false positive REJECT rate for correct diagnoses)

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
