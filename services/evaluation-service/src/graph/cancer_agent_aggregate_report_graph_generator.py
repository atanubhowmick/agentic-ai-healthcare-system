import argparse
import os
import sys
from datetime import datetime

# Path setup — must precede local imports so the script works when run directly.
# When imported as a module, src/ is already on sys.path; this is a no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.config import CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR
from core.mongo_client import load_latest_tfidf_report
from log.logger import logger

_DPI = 300
_TITLE_FONTSIZE = 15
_TITLE_WEIGHT = "bold"
_AXIS_LABEL_FONTSIZE = 12
_TICK_FONTSIZE = 11
_LEGEND_FONTSIZE = 11
_BAR_LABEL_FONTSIZE = 10
_BORDER_COLOR = "#444444"
_BORDER_WIDTH = 1.5

_SEVERITY_ORDER = ["LOW", "HIGH", "CRITICAL"]

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


def _label_bars(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    """Annotate each bar with its numeric value."""
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                fmt.format(h),
                ha="center", va="bottom", fontsize=_BAR_LABEL_FONTSIZE,
            )


# ---------------------------------------------------------------------------
# Plot-1: Accuracy Comparison
# ---------------------------------------------------------------------------

def generate_accuracy_comparison(metrics: dict, run_dir: str) -> None:
    emerg  = metrics.get("emergency_care_needed", {})
    hosp   = metrics.get("hospitalization_needed", {})
    sev    = metrics.get("severity", {})
    cancer = metrics.get("cancer_type", {})

    labels = ["Emergency\nCare", "Hospitalization", "Severity", "Cancer\nType"]
    values = [
        emerg.get("accuracy", 0),
        hosp.get("accuracy", 0),
        sev.get("accuracy", 0),
        cancer.get("match_accuracy", 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_ylim(0, 1.15)
    _label_bars(ax)
    ax.set_title("Accuracy Comparison", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("Accuracy")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot1_accuracy_comparison.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot1_accuracy_comparison.png saved")


# ---------------------------------------------------------------------------
# Plot-2: ROC-AUC Comparison
# ---------------------------------------------------------------------------

def generate_roc_auc_comparison(metrics: dict, run_dir: str) -> None:
    emerg  = metrics.get("emergency_care_needed", {})
    hosp   = metrics.get("hospitalization_needed", {})
    sev    = metrics.get("severity", {})
    cancer = metrics.get("cancer_type", {})

    labels = [
        "Emergency\nCare",
        "Hospitalization",
        "Severity",
        "Cancer Type",
    ]
    values = [
        emerg.get("roc_auc", 0),
        hosp.get("roc_auc", 0),
        sev.get("roc_auc_ovr_weighted") or 0,
        cancer.get("roc_auc_ovr_weighted") or 0,
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_ylim(0, 1.15)
    _label_bars(ax)
    ax.set_title("ROC-AUC Comparison", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("ROC-AUC Score")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot2_roc_auc_comparison.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot2_roc_auc_comparison.png saved")


# ---------------------------------------------------------------------------
# Plot-3: F1 Comparison (grouped: Emergency / Hospitalization / Severity / Cancer)
# ---------------------------------------------------------------------------

def _precision_from_cm(task: dict) -> float:
    """Derive precision from stored confusion matrix: tp / (tp + fp)."""
    cm = task.get("confusion_matrix", {})
    tp, fp = cm.get("tp", 0), cm.get("fp", 0)
    return round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0


def _macro_pr_from_per_class(task: dict) -> tuple[float, float]:
    """Macro-average precision and recall across all stored classes."""
    per_class = task.get("per_class", {})
    classes = list(per_class.values())
    if not classes:
        return 0.0, 0.0
    prec = round(sum(c.get("precision", 0) for c in classes) / len(classes), 4)
    rec  = round(sum(c.get("recall",    0) for c in classes) / len(classes), 4)
    return prec, rec



def generate_f1_comparison(metrics: dict, run_dir: str) -> None:
    emerg  = metrics.get("emergency_care_needed", {})
    hosp   = metrics.get("hospitalization_needed", {})
    sev    = metrics.get("severity", {})
    cancer = metrics.get("cancer_type", {})

    # Binary tasks — precision from confusion matrix, recall = sensitivity_recall
    emerg_prec, emerg_rec = _precision_from_cm(emerg), emerg.get("sensitivity_recall", 0)
    hosp_prec,  hosp_rec  = _precision_from_cm(hosp),  hosp.get("sensitivity_recall",  0)

    # Severity and Cancer type — macro P/R/F1 across classes
    sev_prec,    sev_rec    = _macro_pr_from_per_class(sev)
    cancer_prec, cancer_rec = _macro_pr_from_per_class(cancer)

    task_labels = [
        "Emergency\nCare",
        "Hospitalization",
        "Severity\n(Macro)",
        "Cancer Type\n(Macro)",
    ]
    rows = []
    for label, prec, rec, f1 in [
        (task_labels[0], emerg_prec,  emerg_rec,  emerg.get("f1_score",  0)),
        (task_labels[1], hosp_prec,   hosp_rec,   hosp.get("f1_score",   0)),
        (task_labels[2], sev_prec,    sev_rec,    sev.get("f1_macro",    0)),
        (task_labels[3], cancer_prec, cancer_rec, cancer.get("f1_macro", 0)),
    ]:
        rows.append({"Task": label, "Metric": "Precision", "Value": prec})
        rows.append({"Task": label, "Metric": "Recall",    "Value": rec})
        rows.append({"Task": label, "Metric": "F1 Score",  "Value": f1})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.barplot(x="Task", y="Value", hue="Metric", data=df, ax=ax)
    ax.set_ylim(0, 1.2)
    _label_bars(ax, fmt="{:.3f}")
    ax.set_title("Precision / Recall / F1 Score", fontweight=_TITLE_WEIGHT,
                 fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot3_f1_comparison.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot3_f1_comparison.png saved")


# ---------------------------------------------------------------------------
# Plot-4: Sensitivity vs Specificity (grouped bar)
# ---------------------------------------------------------------------------

def generate_sensitivity_specificity(metrics: dict, run_dir: str) -> None:
    emerg = metrics.get("emergency_care_needed", {})
    hosp  = metrics.get("hospitalization_needed", {})

    rows = [
        {"Task": "Emergency\nCare",  "Metric": "Sensitivity", "Value": emerg.get("sensitivity_recall", 0)},
        {"Task": "Emergency\nCare",  "Metric": "Specificity",  "Value": emerg.get("specificity", 0)},
        {"Task": "Hospitalization",  "Metric": "Sensitivity", "Value": hosp.get("sensitivity_recall", 0)},
        {"Task": "Hospitalization",  "Metric": "Specificity",  "Value": hosp.get("specificity", 0)},
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.barplot(x="Task", y="Value", hue="Metric", data=df, ax=ax)
    ax.set_ylim(0, 1.2)
    ax.set_title("Sensitivity vs Specificity", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot4_sensitivity_specificity.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot4_sensitivity_specificity.png saved")


# ---------------------------------------------------------------------------
# Plot-5: Severity Class Performance (Precision / Recall / F1 per class)
# ---------------------------------------------------------------------------

def generate_severity_class_performance(metrics: dict, run_dir: str) -> None:
    per_class = metrics.get("severity", {}).get("per_class", {})
    if not per_class:
        logger.warning("[CANCER_AGG_GRAPH] No severity per_class data — skipping plot5.")
        return

    rows = []
    for cls in _SEVERITY_ORDER:
        if cls in per_class:
            rows.append({"Class": cls, "Metric": "Precision", "Value": per_class[cls].get("precision", 0)})
            rows.append({"Class": cls, "Metric": "Recall",    "Value": per_class[cls].get("recall",    0)})
            rows.append({"Class": cls, "Metric": "F1 Score",  "Value": per_class[cls].get("f1_score",  0)})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.barplot(x="Class", y="Value", hue="Metric", data=df, ax=ax)
    ax.set_ylim(0, 1.2)
    ax.set_title("Severity Class Performance", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("Score")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot5_severity_class_performance.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot5_severity_class_performance.png saved")


# ---------------------------------------------------------------------------
# Plot-6: Cancer Type Distribution (support per class)
# ---------------------------------------------------------------------------

def generate_cancer_type_distribution(metrics: dict, run_dir: str) -> None:
    per_class = metrics.get("cancer_type", {}).get("per_class", {})
    if not per_class:
        logger.warning("[CANCER_AGG_GRAPH] No cancer_type per_class data — skipping plot6.")
        return

    items  = sorted(per_class.items(), key=lambda x: x[1].get("support", 0), reverse=True)
    labels = [k for k, _ in items]
    values = [v.get("support", 0) for _, v in items]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color="#4C9BE8")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Cancer Type Distribution", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.set_ylabel("Support (Test Cases)")
    ax.set_xlabel("Cancer Type")
    _add_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plot6_cancer_type_distribution.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[CANCER_AGG_GRAPH] plot6_cancer_type_distribution.png saved")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cancer_agent_aggregate_report_graphs(
    output_dir: str = CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR,
) -> None:
    """
    Load the latest TF-IDF baseline evaluation report from MongoDB and produce
    six individual PNGs saved in a timestamped subfolder of *output_dir*.

    Plots:
        plot1_accuracy_comparison.png         — Bar: accuracy per task
        plot2_roc_auc_comparison.png          — Bar: ROC-AUC (binary tasks)
        plot3_f1_comparison.png               — Bar: F1 per task
        plot4_sensitivity_specificity.png     — Grouped bar: sensitivity vs specificity
        plot5_severity_class_performance.png  — Grouped bar: P/R/F1 per severity class
        plot6_cancer_type_distribution.png    — Bar: test-set support per cancer type
    """
    report = load_latest_tfidf_report()
    if report is None:
        logger.error("[CANCER_AGG_GRAPH] No TF-IDF report found in MongoDB — cannot generate graphs.")
        return

    metrics = report.get("metrics", {})
    if not metrics:
        logger.error("[CANCER_AGG_GRAPH] Report has no 'metrics' field.")
        return

    ts      = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    generate_accuracy_comparison(metrics, run_dir)
    generate_roc_auc_comparison(metrics, run_dir)
    generate_f1_comparison(metrics, run_dir)
    generate_sensitivity_specificity(metrics, run_dir)
    generate_severity_class_performance(metrics, run_dir)
    generate_cancer_type_distribution(metrics, run_dir)

    logger.info("[CANCER_AGG_GRAPH] 6 aggregate charts saved to '%s'", run_dir)
    print(f"Cancer agent aggregate graphs saved in folder: {run_dir}")


# ---------------------------------------------------------------------------
# Standalone usage:
#   python graph/cancer_agent_aggregate_report_graph_generator.py
#   python graph/cancer_agent_aggregate_report_graph_generator.py --output-dir my_plots
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cancer agent aggregate evaluation charts from the latest MongoDB report."
    )
    parser.add_argument(
        "--output-dir",
        default=CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR,
        help=f"Root directory for output (default: {CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR})",
    )
    args = parser.parse_args()
    generate_cancer_agent_aggregate_report_graphs(output_dir=args.output_dir)
