from fastapi import APIRouter

from datamodel.models import (
    TfidfBaselineRequest, EvaluationStatusResponse, GenericResponse,
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
