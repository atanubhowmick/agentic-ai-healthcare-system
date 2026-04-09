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
_TITLE_FONTSIZE = 15
_TITLE_WEIGHT = "bold"
_AXIS_LABEL_FONTSIZE = 12
_TICK_FONTSIZE = 11
_LEGEND_FONTSIZE = 11
_ANNOT_FONTSIZE = 10
_BORDER_COLOR = "#444444"
_BORDER_WIDTH = 1.5

# Apply globally so all axes inherit consistent font sizes
plt.rcParams.update({
    "axes.labelsize":  _AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": _TICK_FONTSIZE,
    "ytick.labelsize": _TICK_FONTSIZE,
    "legend.fontsize": _LEGEND_FONTSIZE,
})

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

def _generate_confusion_matrix(cm_data: dict, title: str, filename: str, run_dir: str) -> None:
    tn = cm_data.get("tn", 0)
    fp = cm_data.get("fp", 0)
    fn = cm_data.get("fn", 0)
    tp = cm_data.get("tp", 0)

    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
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
    ax.set_title(title, fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_STAT_GRAPH] %s saved", filename)

def _generate_roc_curve(per_case_rows: list[dict], title: str, filename: str, run_dir: str) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true, y_score, _ = _extract_arrays(per_case_rows)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    if len(set(y_true.tolist())) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety\nfor ROC curve",
                ha="center", va="center", transform=ax.transAxes, fontsize=_ANNOT_FONTSIZE)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val     = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=_LEGEND_FONTSIZE)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_STAT_GRAPH] %s saved", filename)

def _generate_pr_curve(per_case_rows: list[dict], title: str, filename: str, run_dir: str) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true, y_score, _ = _extract_arrays(per_case_rows)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    if len(set(y_true.tolist())) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety\nfor PR curve",
                ha="center", va="center", transform=ax.transAxes, fontsize=_ANNOT_FONTSIZE)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color="#d62728", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=_LEGEND_FONTSIZE)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_STAT_GRAPH] %s saved", filename)

def generate_cancer_agent_statistical_report_graphs(
    output_dir: str = CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR,
) -> None:
    """
    Load the latest TF-IDF baseline evaluation report from MongoDB and produce
    6 individual PNGs saved in a timestamped subfolder of *output_dir*.

    Files:
        emergency_confusion_matrix.png       — Emergency Care: 2×2 CM heatmap
        emergency_roc_curve.png              — Emergency Care: ROC curve with AUC
        emergency_pr_curve.png               — Emergency Care: Precision–Recall curve
        hospitalization_confusion_matrix.png — Hospitalization: 2×2 CM heatmap
        hospitalization_roc_curve.png        — Hospitalization: ROC curve with AUC
        hospitalization_pr_curve.png         — Hospitalization: Precision–Recall curve

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

    # -- Emergency Care -------------------------------------------------------
    emerg      = metrics.get("emergency_care_needed", {})
    emerg_cm   = emerg.get("confusion_matrix", {})
    emerg_rows = emerg.get("per_case_rows", [])

    if not emerg_cm:
        logger.warning("[CANCER_STAT_GRAPH] Emergency confusion_matrix missing — skipping.")
    else:
        _generate_confusion_matrix(
            emerg_cm,
            title="Emergency Care — Confusion Matrix",
            filename="emergency_confusion_matrix.png",
            run_dir=run_dir,
        )

    if not emerg_rows:
        logger.warning("[CANCER_STAT_GRAPH] Emergency per_case_rows missing — skipping ROC/PR.")
    else:
        _generate_roc_curve(
            emerg_rows,
            title="Emergency Care — ROC Curve",
            filename="emergency_roc_curve.png",
            run_dir=run_dir,
        )
        _generate_pr_curve(
            emerg_rows,
            title="Emergency Care — Precision–Recall Curve",
            filename="emergency_pr_curve.png",
            run_dir=run_dir,
        )

    # -- Hospitalization -------------------------------------------------------
    hosp      = metrics.get("hospitalization_needed", {})
    hosp_cm   = hosp.get("confusion_matrix", {})
    hosp_rows = hosp.get("per_case_rows", [])

    if not hosp_cm:
        logger.warning("[CANCER_STAT_GRAPH] Hospitalization confusion_matrix missing — skipping.")
    else:
        _generate_confusion_matrix(
            hosp_cm,
            title="Hospitalization — Confusion Matrix",
            filename="hospitalization_confusion_matrix.png",
            run_dir=run_dir,
        )

    if not hosp_rows:
        logger.warning("[CANCER_STAT_GRAPH] Hospitalization per_case_rows missing — skipping ROC/PR.")
    else:
        _generate_roc_curve(
            hosp_rows,
            title="Hospitalization — ROC Curve",
            filename="hospitalization_roc_curve.png",
            run_dir=run_dir,
        )
        _generate_pr_curve(
            hosp_rows,
            title="Hospitalization — Precision–Recall Curve",
            filename="hospitalization_pr_curve.png",
            run_dir=run_dir,
        )

    logger.info("[CANCER_STAT_GRAPH] Statistical charts saved to '%s'", run_dir)
    print(f"Cancer agent statistical charts saved in folder: {run_dir}")

# Standalone usage:
#   python graph/cancer_agent_statistical_report_graph_generator.py
#   python graph/cancer_agent_statistical_report_graph_generator.py --output-dir my_plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cancer agent statistical charts from the latest MongoDB report."
    )
    parser.add_argument(
        "--output-dir",
        default=CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR,
        help=f"Root directory for output (default: {CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR})",
    )
    args = parser.parse_args()
    generate_cancer_agent_statistical_report_graphs(output_dir=args.output_dir)
