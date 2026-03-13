"""
Cancer Service — orchestrates MIMIC-IV RAG + LLM fallback.

Diagnosis flow:
  1. Search MIMIC-IV ChromaDB for semantically similar oncology cases.
  2a. similarity >= MIMIC_SIMILARITY_THRESHOLD (default 0.75):
        Strong match ? full RAG context injected into prompt  (source: MIMIC_RAG)
  2b. similarity >= MIMIC_PARTIAL_THRESHOLD (default 0.60):
        Weak match ? partial context injected with low-confidence flag  (source: MIMIC_PARTIAL)
  2c. similarity < MIMIC_PARTIAL_THRESHOLD or no cases:
        ? pure LLM call with no MIMIC context  (source: LLM_FALLBACK)
  3. Parse the LLM JSON response and return DiagnosisResponse.
"""

import json
from agent.cancer_agent import cancer_executor, cancer_rag_executor
from datamodel.models import DiagnosisRequest, DiagnosisResult, DiagnosisResponse
from exception.exceptions import CancerSvcException
from rag.mimic_retriever import search_similar_cases, is_collection_populated
from core.config import MIMIC_SIMILARITY_THRESHOLD, MIMIC_PARTIAL_THRESHOLD, MIMIC_TOP_K
from log.logger import logger


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def _build_mimic_context(cases: list[dict]) -> str:
    """Format retrieved MIMIC-IV cases into a readable context block."""
    lines = []
    for i, c in enumerate(cases, start=1):
        lines.append(
            f"Case {i} (similarity={c['similarity_score']}, source={c['source']}):\n"
            f"  Cancer Type       : {c['cancer_type']}\n"
            f"  ICD-10 Codes      : {c['icd_codes']}\n"
            f"  Chief Complaint   : {c['chief_complaint']}\n"
            f"  Treatment Summary : {c['treatment_summary']}\n"
            f"  Severity          : {c['severity']}"
        )
    return "\n\n".join(lines)


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

    # -- Step 1: MIMIC-IV retrieval --------------------------------------------
    source = "LLM_FALLBACK"
    mimic_cases = []

    if not request.is_followup and is_collection_populated():
        all_cases = search_similar_cases(symptoms, top_k=MIMIC_TOP_K)
        strong  = [c for c in all_cases if c["similarity_score"] >= MIMIC_SIMILARITY_THRESHOLD]
        partial = [c for c in all_cases if MIMIC_PARTIAL_THRESHOLD <= c["similarity_score"] < MIMIC_SIMILARITY_THRESHOLD]
        mimic_cases = strong if strong else partial
        if partial and not strong:
            source = "MIMIC_PARTIAL"

    # -- Step 2a: Strong RAG (similarity >= 0.75) ------------------------------
    # -- Step 2b: Partial RAG (0.60 <= similarity < 0.75) ---------------------
    if mimic_cases:
        if source != "MIMIC_PARTIAL":
            source = "MIMIC_RAG"
        mimic_context = _build_mimic_context(mimic_cases)
        if source == "MIMIC_PARTIAL":
            mimic_context = (
                "[Note: these cases are loosely related (low similarity). "
                "Use as background reference only, do not anchor diagnosis to them.]\n\n"
                + mimic_context
            )
        logger.info(
            "[CANCER_SVC] %s | patient: %s | cases: %d | top_score: %.4f",
            source, patient_id, len(mimic_cases), mimic_cases[0]["similarity_score"],
        )
        try:
            result = cancer_rag_executor.invoke(
                {"input": query, "mimic_context": mimic_context},
                config={"configurable": {"session_id": patient_id}},
            )
        except Exception as e:
            logger.warning(
                "[CANCER_SVC] RAG call failed, falling back to LLM | patient: %s | error: %s",
                patient_id, str(e),
            )
            source = "LLM_FALLBACK"
            mimic_cases = []

    # -- Step 2c: Pure LLM fallback (no MIMIC cases above partial threshold) ---
    if not mimic_cases:
        source = "LLM_FALLBACK"
        logger.info("[CANCER_SVC] LLM fallback | patient: %s", patient_id)
        try:
            result = cancer_executor.invoke(
                {"input": query},
                config={"configurable": {"session_id": patient_id}},
            )
        except Exception as e:
            raise CancerSvcException(
                error_code="LLM_INVOCATION_ERROR",
                message=f"LLM call failed for patient {patient_id}: {e}",
            )

    logger.debug(
        "[CANCER_SVC] LLM response | patient: %s | source: %s | length: %d chars",
        patient_id, source, len(result.content),
    )

    # -- Step 3: Parse response ------------------------------------------------
    try:
        raw = _parse_llm_json(result.content)
        diagnosis = DiagnosisResult(**raw)
        logger.debug(
            "[CANCER_SVC] Parsed | patient: %s | severity: %s | type: %s | source: %s",
            patient_id, diagnosis.severity, diagnosis.suspectedCancerType, source,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise CancerSvcException(
            error_code="LLM_RESPONSE_PARSE_ERROR",
            message=f"Failed to parse LLM response for patient {patient_id}: {e}",
        )

    # agent_id encodes the diagnosis source for traceability
    _agent_id_map = {
        "MIMIC_RAG":     "CANCER-AGENT-RAG-1004",
        "MIMIC_PARTIAL": "CANCER-AGENT-PARTIAL-1004",
        "LLM_FALLBACK":  "CANCER-AGENT-LLM-1004",
    }
    agent_id = _agent_id_map.get(source, "CANCER-AGENT-LLM-1004")

    return DiagnosisResponse(
        agent="Cancer_Oncology_Specialist",
        agent_id=agent_id,
        diagnosis=diagnosis,
    )
