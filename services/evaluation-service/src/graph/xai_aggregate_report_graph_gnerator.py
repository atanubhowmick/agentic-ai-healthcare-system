import argparse
import os
import sys
from datetime import datetime

# Path setup — must precede local imports so the script works when run directly.
# When imported as a module, src/ is already on sys.path; this is a no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for use outside the main thread
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

from core.config import XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR
from core.mongo_client import load_latest_xai_report
from log.logger import logger 

_DPI = 300
_SUPTITLE_FONTSIZE = 15
_PANEL_LABEL_FONTSIZE = 14
_PANEL_LABEL_WEIGHT = "bold"
_AXIS_LABEL_FONTSIZE = 12
_TICK_FONTSIZE = 11
_LEGEND_FONTSIZE = 11
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
    """Draw a visible border around a single panel (all four spines)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(_BORDER_WIDTH)
        spine.set_color(_BORDER_COLOR)

# Individual panel draw helpers
# Each helper draws into a supplied Axes object (no plt.figure() calls).

def _draw_decision_performance(ax: plt.Axes, payload: dict) -> None:
    approval_accuracy = payload["decision_accuracy"]["approval_accuracy"]
    specificity       = payload["over_rejection_rate"]["specificity"]
    over_rejection    = payload["over_rejection_rate"]["over_rejection_rate"]
    sns.barplot(
        x=["Approval\nAccuracy", "Specificity", "Over-Rejection\nRate"],
        y=[approval_accuracy, specificity, over_rejection],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("(A) Decision Performance", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Score")
    _add_panel_border(ax)

def _draw_safety_detection(ax: plt.Axes, payload: dict) -> None:
    sensitivity = payload["safety_net_effectiveness"]["sensitivity"]
    miss_rate   = payload["safety_net_effectiveness"]["miss_rate"]
    sns.barplot(x=["Sensitivity", "Miss Rate"], y=[sensitivity, miss_rate], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("(B) Safety Net Effectiveness", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Score")
    _add_panel_border(ax)

def _draw_rule_vs_llm_pie(ax: plt.Axes, payload: dict) -> None:
    rule_rate = payload["rule_engine_coverage"]["rule_coverage_rate"]
    llm_rate  = payload["rule_engine_coverage"]["llm_path_rate"]
    ax.pie(
        [rule_rate, llm_rate],
        labels=["Rule Engine", "LLM Path"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("(C) Rule Engine vs LLM Distribution",
                 fontweight=_PANEL_LABEL_WEIGHT, fontsize=_PANEL_LABEL_FONTSIZE)
    _add_panel_border(ax)

def _draw_rule_hits_vs_llm(ax: plt.Axes, payload: dict) -> None:
    rule_hits = payload["rule_engine_coverage"]["rule_engine_hit"]
    llm_cases = payload["rule_engine_coverage"]["rule_engine_miss_llm_path"]
    sns.barplot(x=["Rule Engine Hits", "LLM Path"], y=[rule_hits, llm_cases], ax=ax)
    ax.set_title("(D) Rule Engine Impact", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Cases")
    _add_panel_border(ax)

def _draw_explanation_complexity(ax: plt.Axes, payload: dict) -> None:
    dist   = payload["xai_sparsity"]["key_concerns_distribution"]
    labels = list(dist.keys())
    values = list(dist.values())
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_title("(A) Explanation Complexity", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Number of Key Concerns")
    _add_panel_border(ax)

def _draw_readability_distribution(ax: plt.Axes, payload: dict) -> None:
    read_dist = payload["xai_interpretability"]["score_distribution"]
    labels    = list(read_dist.keys())
    values    = list(read_dist.values())
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("(B) Readability Distribution", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency")
    _add_panel_border(ax)

def _draw_fidelity_comparison(ax: plt.Axes, payload: dict) -> None:
    decision_sensitivity = payload["xai_fidelity"]["decision_sensitivity_rate"]
    explanation_fidelity = payload["xai_fidelity"]["explanation_fidelity_rate"]
    sns.barplot(
        x=["Decision\nSensitivity", "Explanation\nFidelity"],
        y=[decision_sensitivity, explanation_fidelity],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title("(C) Decision vs Explanation Fidelity",
                 fontweight=_PANEL_LABEL_WEIGHT, fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Score")
    _add_panel_border(ax)

def _draw_reliability_radar(ax: plt.Axes, payload: dict) -> None:
    metrics = {
        "Safety":        payload["safety_net_effectiveness"]["sensitivity"],
        "Stability":     payload["xai_stability"]["stability_rate"],
        "Consistency":   payload["xai_consistency"]["consistency_rate"],
        "Fidelity":      payload["xai_fidelity"]["explanation_fidelity_rate"],
        "Rule\nCoverage": payload["rule_engine_coverage"]["rule_coverage_rate"],
    }
    m_labels = list(metrics.keys())
    m_values = list(metrics.values()) + [list(metrics.values())[0]]
    angles   = np.linspace(0, 2 * np.pi, len(m_labels), endpoint=False).tolist()
    angles  += angles[:1]

    ax.plot(angles, m_values, linewidth=1.5)
    ax.fill(angles, m_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(m_labels, fontsize=_TICK_FONTSIZE)
    ax.set_title("(D) Overall Reliability Radar",
                 fontweight=_PANEL_LABEL_WEIGHT, fontsize=_PANEL_LABEL_FONTSIZE, pad=15)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_linewidth(_BORDER_WIDTH)
    ax.spines["polar"].set_color(_BORDER_COLOR)

def _draw_stability_analysis(ax: plt.Axes, payload: dict) -> None:
    stable   = payload["xai_stability"]["stable_cases"]
    unstable = payload["xai_stability"]["unstable_cases"]
    sns.barplot(x=["Stable", "Unstable"], y=[stable, unstable], ax=ax)
    ax.set_title("(A) Stability Analysis", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel("Cases")
    _add_panel_border(ax)

def _draw_consistency_pie(ax: plt.Axes, payload: dict) -> None:
    consistent   = payload["xai_consistency"]["consistent"]
    inconsistent = payload["xai_consistency"]["inconsistent"]
    ax.pie(
        [consistent, inconsistent],
        labels=["Consistent", "Inconsistent"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("(B) Consistency Analysis", fontweight=_PANEL_LABEL_WEIGHT,
                 fontsize=_PANEL_LABEL_FONTSIZE)
    _add_panel_border(ax)

def _build_collage_a(payload: dict, run_dir: str) -> None:
    """Decision & Safety Performance — 2×2 grid of 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=_DPI)
    fig.suptitle("Decision & Safety Performance", fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=1.01)
    sns.set_theme(style="whitegrid")

    _draw_decision_performance(axes[0, 0], payload)
    _draw_safety_detection(axes[0, 1], payload)
    _draw_rule_vs_llm_pie(axes[1, 0], payload)
    _draw_rule_hits_vs_llm(axes[1, 1], payload)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "A_decision_safety_performance.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

def _build_collage_b(payload: dict, run_dir: str) -> None:
    """Explainability & Interpretation — 2x2 grid, B4 uses polar axes."""
    fig = plt.figure(figsize=(16, 12), dpi=_DPI)
    fig.suptitle("Explainability & Interpretation", fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=1.01)
    sns.set_theme(style="whitegrid")

    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_b1 = fig.add_subplot(gs[0, 0])
    ax_b2 = fig.add_subplot(gs[0, 1])
    ax_b3 = fig.add_subplot(gs[1, 0])
    ax_b4 = fig.add_subplot(gs[1, 1], polar=True)

    _draw_explanation_complexity(ax_b1, payload)
    _draw_readability_distribution(ax_b2, payload)
    _draw_fidelity_comparison(ax_b3, payload)
    _draw_reliability_radar(ax_b4, payload)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "B_explainability_interpretation.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

def _build_collage_c(payload: dict, run_dir: str) -> None:
    """Robustness & Stability — 1x2 grid of 2 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=_DPI)
    fig.suptitle("Robustness & Stability", fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=1.02)
    sns.set_theme(style="whitegrid")

    _draw_stability_analysis(axes[0], payload)
    _draw_consistency_pie(axes[1], payload)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "C_robustness_stability.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

def generate_xai_aggregate_report_graphs(output_dir: str = XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR) -> None:
    """
    Load the latest XAI evaluation report from MongoDB and produce three
    high-resolution collage PNGs saved in a timestamped subfolder of *output_dir*.

    Collages:
        A_decision_safety_performance.png    — 4 panels (2x2)
        B_explainability_interpretation.png  — 4 panels (2x2)
        C_robustness_stability.png           — 2 panels (1x2)
    """
    report = load_latest_xai_report()
    if report is None:
        logger.error("[GRAPH] No XAI report found in MongoDB — cannot generate graphs.")
        return

    payload = report  # MongoDB doc has fields at the top level (no 'payload' wrapper)

    ts      = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    _build_collage_a(payload, run_dir)
    _build_collage_b(payload, run_dir)
    _build_collage_c(payload, run_dir)

    logger.info("[GRAPH] 3 collage charts saved to '%s'", run_dir)
    print(f"Graphs saved in folder: {run_dir}")

# Standalone usage:
#   python graph/xai_aggregate_report_graph_gnerator.py
#   python graph/xai_aggregate_report_graph_gnerator.py --output-dir my_plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate XAI evaluation report collages from the latest MongoDB report."
    )
    parser.add_argument(
        "--output-dir",
        default=XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR,
        help=f"Root directory for output (default: {XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR}, override with XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR env var)",
    )
    args = parser.parse_args()
    generate_xai_aggregate_report_graphs(output_dir=args.output_dir)
