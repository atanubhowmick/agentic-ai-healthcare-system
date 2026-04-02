import argparse
import os
import sys
from datetime import datetime

# Path setup — must precede local imports so the script works when run directly.
# When imported as a module, src/ is already on sys.path; this is a no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from core.config import CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR  # noqa: E402
from core.mongo_client import load_latest_tfidf_report  # noqa: E402
from log.logger import logger  # noqa: E402

_DPI = 300
_PANEL_LABEL_FONTSIZE = 11
_PANEL_LABEL_WEIGHT = "bold"
_BORDER_COLOR = "#444444"
_BORDER_WIDTH = 1.5


def _add_border(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(_BORDER_WIDTH)
        spine.set_color(_BORDER_COLOR)


def _extract_arrays(per_case_rows: list[dict]):
    """Return (y_true, y_score, y_pred) numpy arrays from stored per-case rows."""
    y_true  = np.array([r["true_label"]      for r in per_case_rows], dtype=int)
    y_score = np.array([r["score"]            for r in per_case_rows], dtype=float)
    y_pred  = np.array([r["predicted_label"]  for r in per_case_rows], dtype=int)
    return y_true, y_score, y_pred


# ---------------------------------------------------------------------------
# Individual panel draw helpers
# ---------------------------------------------------------------------------

def _draw_confusion_matrix(ax: plt.Axes, cm_data: dict) -> None:
    """(a) 2×2 confusion matrix heatmap, light→dark blue."""
    tn = cm_data.get("tn", 0)
    fp = cm_data.get("fp", 0)
    fn = cm_data.get("fn", 0)
    tp = cm_data.get("tp", 0)

    cm = np.array([[tn, fp], [fn, tp]])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        linewidths=0.5,
        linecolor="gray",
        cbar=False,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("(a) Confusion Matrix", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    _add_border(ax)


def _draw_roc_curve(ax: plt.Axes, per_case_rows: list[dict]) -> None:
    """(b) ROC curve with AUC annotation."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true, y_score, _ = _extract_arrays(per_case_rows)

    if len(set(y_true.tolist())) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety\nfor ROC curve",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val     = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(b) ROC Curve", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    _add_border(ax)


def _draw_pr_curve(ax: plt.Axes, per_case_rows: list[dict]) -> None:
    """(c) Precision–Recall curve with PR-AUC annotation."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true, y_score, _ = _extract_arrays(per_case_rows)

    if len(set(y_true.tolist())) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety\nfor PR curve",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color="#d62728", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(c) Precision–Recall Curve", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    _add_border(ax)


# ---------------------------------------------------------------------------
# Collage builder (1×3 horizontal)
# ---------------------------------------------------------------------------

def _build_collage(
    cm_data: dict,
    per_case_rows: list[dict],
    title: str,
    filename: str,
    run_dir: str,
) -> None:
    """
    Build a 1×3 collage: (a) Confusion Matrix | (b) ROC Curve | (c) PR Curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=_DPI)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    sns.set_theme(style="whitegrid")

    _draw_confusion_matrix(axes[0], cm_data)
    _draw_roc_curve(axes[1], per_case_rows)
    _draw_pr_curve(axes[2], per_case_rows)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_STAT_GRAPH] %s saved", filename)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cancer_agent_statistical_report_graphs(
    output_dir: str = CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR,
) -> None:
    """
    Load the latest TF-IDF baseline evaluation report from MongoDB and produce
    two 1×3 collage PNGs saved in a timestamped subfolder of *output_dir*.

    Figures:
        figure1_emergency_statistical.png       — CM | ROC | PR  (Emergency Care)
        figure2_hospitalization_statistical.png — CM | ROC | PR  (Hospitalization)

    Requires:
        - confusion_matrix (tn/fp/fn/tp) in each binary task's metrics
        - per_case_rows in each binary task's metrics (from TfidfBaselineEvaluator)
    """
    report = load_latest_tfidf_report()
    if report is None:
        logger.error("[CANCER_STAT_GRAPH] No TF-IDF report found in MongoDB — cannot generate charts.")
        return

    metrics = report.get("metrics", {})
    if not metrics:
        logger.error("[CANCER_STAT_GRAPH] Report has no 'metrics' field.")
        return

    ts      = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # -- Figure 1: Emergency --------------------------------------------------
    emerg = metrics.get("emergency_care_needed", {})
    emerg_cm   = emerg.get("confusion_matrix", {})
    emerg_rows = emerg.get("per_case_rows", [])

    if not emerg_cm:
        logger.warning("[CANCER_STAT_GRAPH] Emergency confusion_matrix missing — skipping Figure 1.")
    elif not emerg_rows:
        logger.warning("[CANCER_STAT_GRAPH] Emergency per_case_rows missing — skipping Figure 1.")
    else:
        _build_collage(
            cm_data=emerg_cm,
            per_case_rows=emerg_rows,
            title="Emergency Care - Statistical Evaluation",
            filename="figure1_emergency_statistical.png",
            run_dir=run_dir,
        )

    # -- Figure 2: Hospitalization --------------------------------------------
    hosp = metrics.get("hospitalization_needed", {})
    hosp_cm   = hosp.get("confusion_matrix", {})
    hosp_rows = hosp.get("per_case_rows", [])

    if not hosp_cm:
        logger.warning("[CANCER_STAT_GRAPH] Hospitalization confusion_matrix missing — skipping Figure 2.")
    elif not hosp_rows:
        logger.warning("[CANCER_STAT_GRAPH] Hospitalization per_case_rows missing — skipping Figure 2.")
    else:
        _build_collage(
            cm_data=hosp_cm,
            per_case_rows=hosp_rows,
            title="Hospitalization - Statistical Evaluation",
            filename="figure2_hospitalization_statistical.png",
            run_dir=run_dir,
        )

    logger.info("[CANCER_STAT_GRAPH] Statistical collages saved to '%s'", run_dir)
    print(f"Cancer agent statistical charts saved in folder: {run_dir}")


# ---------------------------------------------------------------------------
# Standalone usage:
#   python graph/cancer_agent_statistical_report_graph_generator.py
#   python graph/cancer_agent_statistical_report_graph_generator.py --output-dir my_plots
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cancer agent statistical collages from the latest MongoDB report."
    )
    parser.add_argument(
        "--output-dir",
        default=CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR,
        help=f"Root directory for output (default: {CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR})",
    )
    args = parser.parse_args()
    generate_cancer_agent_statistical_report_graphs(output_dir=args.output_dir)
