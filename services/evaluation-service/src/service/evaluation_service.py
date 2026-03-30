import threading

from core.mongo_client import (
    has_tfidf_report, load_latest_tfidf_report,
    has_xai_report, load_latest_xai_report,
)
from evaluators.tfidf_baseline_evaluator import TfidfBaselineEvaluator
from evaluators.xai_evaluator import XaiEvaluator
from exception.exceptions import EvaluationSvcException
from log.logger import logger

_tfidf_lock    = threading.Lock()
_tfidf_running = False

_xai_lock    = threading.Lock()
_xai_running = False


def start_tfidf_evaluation(max_cases: int, test_size: float = 0.20) -> None:
    """
    Launch a TF-IDF baseline evaluation in a background thread.
    Raises EvaluationSvcException if a run is already in progress.
    """
    global _tfidf_running

    with _tfidf_lock:
        if _tfidf_running:
            raise EvaluationSvcException(
                "TFIDF_EVAL_ALREADY_RUNNING",
                "A TF-IDF baseline evaluation is already in progress.",
            )
        _tfidf_running = True

    thread = threading.Thread(
        target=_run_tfidf_evaluation,
        args=(max_cases, test_size),
        daemon=True,
    )
    thread.start()
    logger.info("[TFIDF SERVICE] Background evaluation thread started | max_cases=%s | test_size=%.0f%%",
                max_cases or "all", test_size * 100)


def get_tfidf_status() -> dict:
    """Return TF-IDF run state and whether a report exists."""
    return {
        "running":          _tfidf_running,
        "report_available": has_tfidf_report(),
    }


def get_tfidf_report() -> dict | None:
    """Return the most recent TF-IDF baseline report from MongoDB."""
    return load_latest_tfidf_report()


def _run_tfidf_evaluation(max_cases: int, test_size: float = 0.20) -> None:
    global _tfidf_running
    try:
        evaluator = TfidfBaselineEvaluator(test_size=test_size)
        evaluator.run_evaluation(max_cases=max_cases)
        logger.info("[TFIDF SERVICE] Evaluation complete.")
    except Exception as exc:
        logger.error("[TFIDF SERVICE] Evaluation failed: %s", exc)
    finally:
        with _tfidf_lock:
            _tfidf_running = False


# ---------------------------------------------------------------------------
# XAI evaluation
# ---------------------------------------------------------------------------

def start_xai_evaluation(
    max_cases: int,
    max_correct_cases: int = 150,
    max_undertriage_cases: int = 50,
    max_stability_cases: int = 30,
    max_fidelity_cases: int = 30,
    max_consistency_cases: int = 30,
) -> None:
    """
    Launch an XAI validation service evaluation in a background thread.
    Raises EvaluationSvcException if a run is already in progress.
    """
    global _xai_running

    with _xai_lock:
        if _xai_running:
            raise EvaluationSvcException(
                "XAI_EVAL_ALREADY_RUNNING",
                "An XAI evaluation is already in progress.",
            )
        _xai_running = True

    thread = threading.Thread(
        target=_run_xai_evaluation,
        args=(max_cases, max_correct_cases, max_undertriage_cases,
              max_stability_cases, max_fidelity_cases, max_consistency_cases),
        daemon=True,
    )
    thread.start()
    logger.info(
        "[XAI SERVICE] Background evaluation thread started | max_cases=%s | "
        "correct=%d undertriage=%d stability=%d fidelity=%d consistency=%d",
        max_cases or "all", max_correct_cases, max_undertriage_cases,
        max_stability_cases, max_fidelity_cases, max_consistency_cases,
    )


def get_xai_status() -> dict:
    """Return XAI run state and whether a report exists."""
    return {
        "running":          _xai_running,
        "report_available": has_xai_report(),
    }


def get_xai_report() -> dict | None:
    """Return the most recent XAI evaluation report from MongoDB."""
    return load_latest_xai_report()


def _run_xai_evaluation(
    max_cases: int,
    max_correct_cases: int,
    max_undertriage_cases: int,
    max_stability_cases: int,
    max_fidelity_cases: int,
    max_consistency_cases: int,
) -> None:
    global _xai_running
    try:
        evaluator = XaiEvaluator(
            max_cases=max_cases,
            max_correct_cases=max_correct_cases,
            max_undertriage_cases=max_undertriage_cases,
            max_stability_cases=max_stability_cases,
            max_fidelity_cases=max_fidelity_cases,
            max_consistency_cases=max_consistency_cases,
        )
        evaluator.run_evaluation()
        logger.info("[XAI SERVICE] Evaluation complete.")
    except Exception as exc:
        logger.error("[XAI SERVICE] Evaluation failed: %s", exc)
    finally:
        with _xai_lock:
            _xai_running = False
