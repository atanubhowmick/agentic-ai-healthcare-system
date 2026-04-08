from typing import Annotated, List
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """Shared state threaded through every LangGraph node. All fields initialised before graph invoke."""

    patient_id: str
    symptoms: str
    messages: Annotated[List[str], operator.add]   # append-only audit trail

    chroma_cache_hit: bool
    chroma_cached_result: dict | None

    assigned_specialist: str        # "cardiology" | "neurology" | "pathology" | "cancer" | "unknown"
    secondary_check_needed: bool
    secondary_check_done: bool      # prevents re-running secondary on retry loops

    specialist_diagnosis: dict | None
    specialist_agent: str | None

    secondary_diagnosis: dict | None
    secondary_agent: str | None

    conflict_detected: bool
    conflict_reason: str

    diagnosis_validated: bool
    diagnosis_retry_count: int
    diagnosis_xai_result: dict | None

    treatment_recommendation: dict | None

    treatment_validated: bool
    treatment_retry_count: int
    treatment_xai_result: dict | None

    requires_human_review: bool
    human_review_reason: str

    final_response: dict | None
