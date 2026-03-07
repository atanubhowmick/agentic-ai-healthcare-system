from typing import Annotated, List
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """
    Shared state passed through every node in the LangGraph workflow.
    All fields must be initialised before the graph is invoked.
    """

    # ── Patient context ──────────────────────────────────────────────────────
    patient_id: str
    symptoms: str

    # Append-only audit trail — every node adds its own log entry
    messages: Annotated[List[str], operator.add]

    # ── Triage ───────────────────────────────────────────────────────────────
    assigned_specialist: str        # "cardiology" | "neurology" | "pathology" | "unknown"
    secondary_check_needed: bool    # True when lab cross-check is warranted
    secondary_check_done: bool      # Prevents re-running secondary on retry loops

    # ── Primary specialist result ─────────────────────────────────────────────
    specialist_diagnosis: dict | None   # Diagnosis dict from the specialist agent
    specialist_agent: str | None        # e.g. "Cardiology_Specialist"

    # ── Secondary specialist (pathology cross-check) ──────────────────────────
    secondary_diagnosis: dict | None
    secondary_agent: str | None

    # ── Conflict detection ────────────────────────────────────────────────────
    conflict_detected: bool
    conflict_reason: str

    # ── XAI diagnosis validation ──────────────────────────────────────────────
    diagnosis_validated: bool
    diagnosis_retry_count: int          # Incremented inside xai_diagnosis_validator_node
    diagnosis_xai_result: dict | None

    # ── Treatment ─────────────────────────────────────────────────────────────
    treatment_recommendation: dict | None

    # ── XAI treatment validation ──────────────────────────────────────────────
    treatment_validated: bool
    treatment_retry_count: int          # Incremented inside xai_treatment_validator_node
    treatment_xai_result: dict | None

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    requires_human_review: bool
    human_review_reason: str

    # ── Final assembled response ──────────────────────────────────────────────
    final_response: dict | None
