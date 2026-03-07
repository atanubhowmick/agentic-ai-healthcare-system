from fastapi import APIRouter
from schemas.request import OrchestratorRequest
from schemas.response import OrchestratorResponse, DiagnosisDetail, GenericResponse
from agents.graph import create_orchestrator_graph
from core.exceptions import GraphExecutionException
from constants import ORCHESTRATOR_AGENT_ID
from log.logger import logger

router = APIRouter(prefix="/orchestrator")

# Compile the LangGraph graph once at startup
_graph = create_orchestrator_graph()


def _build_initial_state(patient_id: str, symptoms: str) -> dict:
    """Return a fully-initialised AgentState dict for a new case."""
    return {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "messages": [],
        "assigned_specialist": "",
        "secondary_check_needed": False,
        "secondary_check_done": False,
        "specialist_diagnosis": None,
        "specialist_agent": None,
        "secondary_diagnosis": None,
        "secondary_agent": None,
        "conflict_detected": False,
        "conflict_reason": "",
        "diagnosis_validated": False,
        "diagnosis_retry_count": 0,
        "diagnosis_xai_result": None,
        "treatment_recommendation": None,
        "treatment_validated": False,
        "treatment_retry_count": 0,
        "treatment_xai_result": None,
        "requires_human_review": False,
        "human_review_reason": "",
        "final_response": None,
    }


@router.post("/diagnose", response_model=GenericResponse[OrchestratorResponse])
async def orchestrate_diagnosis(
    request: OrchestratorRequest,
) -> GenericResponse[OrchestratorResponse]:
    logger.info(
        "[API] /orchestrator/diagnose | patient: %s | symptoms: %.80s...",
        request.patient_id, request.symptoms,
    )

    initial_state = _build_initial_state(request.patient_id, request.symptoms)

    try:
        final_state = await _graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("[API] Graph execution error: %s", str(e))
        raise GraphExecutionException(message=f"Orchestration workflow failed: {str(e)}")

    fr = final_state.get("final_response") or {}

    # Build typed response
    diag_raw = fr.get("diagnosis")
    diagnosis = None
    if diag_raw:
        diagnosis = DiagnosisDetail(
            summary=diag_raw.get("summary", ""),
            severity=diag_raw.get("severity", "UNKNOWN"),
            emergency_care_needed=diag_raw.get("emergency_care_needed", "UNKNOWN"),
            hospitalization_needed=diag_raw.get("hospitalization_needed", "UNKNOWN"),
            full_details=diag_raw.get("full_details", {}),
        )

    response = OrchestratorResponse(
        patient_id=fr.get("patient_id", request.patient_id),
        agent_id=ORCHESTRATOR_AGENT_ID,
        status=fr.get("status", "UNKNOWN"),
        specialist_agent=fr.get("specialist_agent"),
        diagnosis=diagnosis,
        xai_diagnosis_validation=fr.get("xai_diagnosis_validation"),
        treatment=fr.get("treatment"),
        xai_treatment_validation=fr.get("xai_treatment_validation"),
        conflict_detected=fr.get("conflict_detected", False),
        conflict_reason=fr.get("conflict_reason", ""),
        human_review_reason=fr.get("human_review_reason"),
        audit_trail=fr.get("audit_trail", []),
    )

    logger.info(
        "[API] Orchestration complete | patient: %s | status: %s",
        response.patient_id, response.status,
    )
    return GenericResponse.success(response)
