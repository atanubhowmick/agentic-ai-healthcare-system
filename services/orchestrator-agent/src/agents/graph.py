"""
LangGraph workflow for the Healthcare Orchestrator.

Full flow:
  triage → specialist → [secondary_check → conflict_check] →
  xai_diagnosis_validator → treatment → xai_treatment_validator → finish

Retry loops and human-review gates are driven by flags set inside the nodes.
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    triage_node,
    specialist_node,
    secondary_check_node,
    conflict_check_node,
    xai_diagnosis_validator_node,
    treatment_node,
    xai_treatment_validator_node,
    finish_node,
)


# ── Routing functions ────────────────────────────────────────────────────────

def _route_after_triage(state: AgentState) -> str:
    specialist = state.get("assigned_specialist", "unknown")
    if specialist in ("cardiology", "neurology", "pathology"):
        return "specialist"
    return "finish"   # unknown → human review


def _route_after_specialist(state: AgentState) -> str:
    if state.get("requires_human_review"):
        return "finish"
    # Run secondary pathology cross-check only once (skip on retry loops)
    if (
        state.get("secondary_check_needed")
        and state.get("assigned_specialist") != "pathology"
        and not state.get("secondary_check_done", False)
    ):
        return "secondary_check"
    return "xai_diagnosis_validator"


def _route_after_conflict(state: AgentState) -> str:
    if state.get("conflict_detected"):
        return "finish"   # conflict → human review (flag already set in node)
    return "xai_diagnosis_validator"


def _route_after_xai_diagnosis(state: AgentState) -> str:
    if state.get("requires_human_review"):
        return "finish"
    if state.get("diagnosis_validated"):
        return "treatment"
    # retry_count was already incremented inside the node
    return "specialist"


def _route_after_treatment(state: AgentState) -> str:
    if state.get("requires_human_review"):
        return "finish"
    if not state.get("treatment_recommendation"):
        # treatment failed — let xai_treatment_validator handle retry counting
        return "xai_treatment_validator"
    return "xai_treatment_validator"


def _route_after_xai_treatment(state: AgentState) -> str:
    if state.get("requires_human_review"):
        return "finish"
    if state.get("treatment_validated"):
        return "finish"
    # retry_count was already incremented inside the node
    return "treatment"


# ── Graph builder ────────────────────────────────────────────────────────────

def create_orchestrator_graph():
    """
    Build and compile the full LangGraph healthcare orchestration workflow.

    Nodes  : triage, specialist, secondary_check, conflict_check,
             xai_diagnosis_validator, treatment, xai_treatment_validator, finish
    Loops  : diagnosis retry (specialist ↔ xai_diagnosis_validator, max 3×)
             treatment retry (treatment  ↔ xai_treatment_validator,  max 3×)
    Gates  : conflict_check, human-review flags (requires_human_review)
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("triage",                   triage_node)
    workflow.add_node("specialist",               specialist_node)
    workflow.add_node("secondary_check",          secondary_check_node)
    workflow.add_node("conflict_check",           conflict_check_node)
    workflow.add_node("xai_diagnosis_validator",  xai_diagnosis_validator_node)
    workflow.add_node("treatment",                treatment_node)
    workflow.add_node("xai_treatment_validator",  xai_treatment_validator_node)
    workflow.add_node("finish",                   finish_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("triage")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "triage", _route_after_triage,
        {"specialist": "specialist", "finish": "finish"},
    )

    workflow.add_conditional_edges(
        "specialist", _route_after_specialist,
        {
            "secondary_check": "secondary_check",
            "xai_diagnosis_validator": "xai_diagnosis_validator",
            "finish": "finish",
        },
    )

    # secondary_check always goes to conflict_check
    workflow.add_edge("secondary_check", "conflict_check")

    workflow.add_conditional_edges(
        "conflict_check", _route_after_conflict,
        {"finish": "finish", "xai_diagnosis_validator": "xai_diagnosis_validator"},
    )

    # Retry loop #1: xai_diagnosis_validator ↔ specialist
    workflow.add_conditional_edges(
        "xai_diagnosis_validator", _route_after_xai_diagnosis,
        {"treatment": "treatment", "specialist": "specialist", "finish": "finish"},
    )

    workflow.add_conditional_edges(
        "treatment", _route_after_treatment,
        {"xai_treatment_validator": "xai_treatment_validator", "finish": "finish"},
    )

    # Retry loop #2: xai_treatment_validator ↔ treatment
    workflow.add_conditional_edges(
        "xai_treatment_validator", _route_after_xai_treatment,
        {"finish": "finish", "treatment": "treatment"},
    )

    workflow.add_edge("finish", END)

    return workflow.compile()
