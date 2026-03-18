"""
Cancer Service - orchestrates diagnosis via the DeepAgent.

Diagnosis flow:
  1. Build the patient query string.
  2. Invoke the DeepAgent directly using SystemMessage / HumanMessage.
     - Per-session chat history is maintained here in _session_store.
     - The agent autonomously calls search_mimic_cases tool when appropriate.
  3. Parse the JSON response and return DiagnosisResponse.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.cancer_agent import cancer_executor, BASE_SYSTEM
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import CancerSvcException
from log.logger import logger


# -- Per-session chat history --------------------------------------------------

_session_store: dict[str, list] = {}


# -- Helpers -------------------------------------------------------------------

def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


# -- Main entry point ----------------------------------------------------------

def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    patient_id = request.patient_id
    symptoms   = request.symptoms

    if request.is_followup:
        query = symptoms
        logger.debug("Follow-up query | patient: %s", patient_id)
    else:
        query = (
            f"Analyze patient {patient_id} with symptoms: {symptoms}. "
            "Evaluate for oncological conditions and respond strictly in the requested JSON format."
        )
        logger.debug("Initial query | patient: %s", patient_id)

    # -- Build messages with session history -----------------------------------
    history: list = _session_store.setdefault(patient_id, [])
    history.append(HumanMessage(content=query))

    logger.debug(
        "[CANCER_SVC] Invoking DeepAgent | patient: %s | history_turns: %d",
        patient_id, len(history),
    )

    # -- Invoke DeepAgent ------------------------------------------------------
    try:
        result = cancer_executor.invoke({
            "messages": [SystemMessage(content=BASE_SYSTEM)] + history
        })
    except Exception as e:
        raise CancerSvcException(
            error_code="LLM_INVOCATION_ERROR",
            message=f"Agent call failed for patient {patient_id}: {e}",
        )

    last_msg = result["messages"][-1]
    _session_store[patient_id] = result["messages"]

    logger.debug(
        "[CANCER_SVC] Agent response | patient: %s | length: %d chars",
        patient_id, len(last_msg.content),
    )

    # -- Parse response --------------------------------------------------------
    try:
        raw = _parse_llm_json(last_msg.content)
        diagnosis = DiagnosisResult(**raw)
        logger.debug(
            "[CANCER_SVC] Parsed | patient: %s | severity: %s | type: %s",
            patient_id, diagnosis.severity, diagnosis.suspectedCancerType,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise CancerSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse agent response for patient {patient_id}: {e}",
        )

    return DiagnosisResponse(
        agent="Cancer_Oncology_Specialist",
        agent_id="CANCER-AGENT-1004",
        diagnosis=diagnosis,
    )
