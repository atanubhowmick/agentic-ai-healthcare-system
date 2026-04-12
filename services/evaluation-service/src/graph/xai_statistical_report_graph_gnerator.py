import os
import sys
from datetime import datetime

# Path setup — must precede local imports so the script works when run directly.
# When imported as a module, src/ is already on sys.path; this is a no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core.config import XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR
from core.mongo_client import load_latest_xai_report
from log.logger import logger

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

def _add_panel_border(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(_BORDER_WIDTH)
        spine.set_color(_BORDER_COLOR)

def _prepare_classification_data(payload: dict):
    """
    Extract (y_true, y_score, decisions) from per_case_decisions.

    y_true   : 1 = unsafe, 0 = safe
    y_score  : risk_score (probability of being unsafe)
    decisions: list of raw validator decision strings
    """
    rows      = payload.get("per_case_decisions", [])
    y_true    = [1 if r["true_label"] == "unsafe" else 0 for r in rows]
    y_score   = [r["risk_score"] for r in rows]
    decisions = [r["validator_decision"] for r in rows]
    return y_true, y_score, decisions

def generate_confusion_matrix(payload: dict, run_dir: str) -> None:
    y_true, _, decisions = _prepare_classification_data(payload)
    y_pred = [0 if d == "APPROVE" else 1 for d in decisions]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Approve", "Reject/Review"],
        yticklabels=["Safe", "Unsafe"],
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Validator Decision")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix Analysis", fontweight=_TITLE_WEIGHT,
                 fontsize=_TITLE_FONTSIZE)
    _add_panel_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[STAT_GRAPH] confusion_matrix.png saved")

def generate_roc_curve(payload: dict, run_dir: str) -> None:
    y_true, y_score, _ = _prepare_classification_data(payload)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    if len(set(y_true)) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety for ROC curve",
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
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    _add_panel_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "roc_curve.png"), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[STAT_GRAPH] roc_curve.png saved")

def generate_pr_curve(payload: dict, run_dir: str) -> None:

    y_true, y_score, _ = _prepare_classification_data(payload)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    if len(set(y_true)) < 2:
        ax.text(0.5, 0.5, "Insufficient class variety for PR curve",
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
    ax.set_title("Precision–Recall Curve", fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    _add_panel_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "precision_recall_curve.png"), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[STAT_GRAPH] precision_recall_curve.png saved")

def generate_confidence_distribution(payload: dict, run_dir: str) -> None:
    rows = payload.get("per_case_decisions", [])

    correct_conf = [
        r["confidence_score"] for r in rows
        if (r["validator_decision"] == "APPROVE" and r["true_label"] == "safe")
        or (r["validator_decision"] in ("REJECT", "REVIEW") and r["true_label"] == "unsafe")
    ]
    wrong_conf = [
        r["confidence_score"] for r in rows
        if (r["validator_decision"] == "APPROVE" and r["true_label"] == "unsafe")
        or (r["validator_decision"] in ("REJECT", "REVIEW") and r["true_label"] == "safe")
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    if correct_conf:
        sns.histplot(correct_conf, color="green", label="Correct", kde=True, ax=ax,
                     alpha=0.5, bins=20)
    if wrong_conf:
        sns.histplot(wrong_conf, color="red", label="Incorrect", kde=True, ax=ax,
                     alpha=0.5, bins=20)

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence Distribution for Validator Decisions",
                 fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE)
    _add_panel_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "confidence_distribution.png"), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[STAT_GRAPH] confidence_distribution.png saved")

def generate_risk_vs_confidence(payload: dict, run_dir: str) -> None:
    rows = payload.get("per_case_decisions", [])

    _COLORS = {"APPROVE": "green", "REJECT": "red", "REVIEW": "orange"}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    sns.set_theme(style="whitegrid")

    for decision, color in _COLORS.items():
        subset = [r for r in rows if r["validator_decision"] == decision]
        if subset:
            ax.scatter(
                [r["risk_score"] for r in subset],
                [r["confidence_score"] for r in subset],
                color=color,
                alpha=0.6,
                label=decision.capitalize(),
                s=30,
            )

    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Risk vs Confidence Decision Boundary",
                 fontweight=_TITLE_WEIGHT, fontsize=_TITLE_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE)
    _add_panel_border(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "risk_vs_confidence.png"), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[STAT_GRAPH] risk_vs_confidence.png saved")

def generate_xai_statistical_report_graphs(output_dir: str = XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR) -> None:
    """
    Load the latest XAI evaluation report from MongoDB and produce five
    high-resolution statistical charts saved in a timestamped subfolder of *output_dir*.

    Charts:
        confusion_matrix.png        — Confusion Matrix heatmap
        roc_curve.png               — ROC Curve with AUC
        precision_recall_curve.png  — Precision–Recall Curve with PR-AUC
        confidence_distribution.png — KDE histogram: correct vs incorrect confidence
        risk_vs_confidence.png      — Scatter: risk score vs confidence, coloured by decision

    Requires per_case_decisions to be present in the report (populated by XaiEvaluator).
    """
    report = load_latest_xai_report()
    if report is None:
        logger.error("[STAT_GRAPH] No XAI report found in MongoDB — cannot generate charts.")
        return

    if not report.get("per_case_decisions"):
        logger.error("[STAT_GRAPH] per_case_decisions missing — run a fresh XAI evaluation first.")
        return

    ts      = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    generate_confusion_matrix(report, run_dir)
    generate_roc_curve(report, run_dir)
    generate_pr_curve(report, run_dir)
    generate_confidence_distribution(report, run_dir)
    generate_risk_vs_confidence(report, run_dir)

    logger.info("[STAT_GRAPH] 5 statistical charts saved to '%s'", run_dir)
    print(f"Statistical charts saved in folder: {run_dir}")

# Standalone usage:
#   python graph/xai_statistical_report_graph_gnerator.py
#   python graph/xai_statistical_report_graph_gnerator.py --output-dir my_plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate XAI statistical classification charts from the latest MongoDB report."
    )
    parser.add_argument(
        "--output-dir",
        default=XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR,
        help=f"Root directory for output (default: {XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR}, "
             "override with XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR env var)",
    )
    args = parser.parse_args()
    generate_xai_statistical_report_graphs(output_dir=args.output_dir)
