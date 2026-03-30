import threading

from core.mongo_client import has_tfidf_report, load_latest_tfidf_report
from evaluators.tfidf_baseline_evaluator import TfidfBaselineEvaluator
from exception.exceptions import EvaluationSvcException
from log.logger import logger

_tfidf_lock    = threading.Lock()
_tfidf_running = False


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
