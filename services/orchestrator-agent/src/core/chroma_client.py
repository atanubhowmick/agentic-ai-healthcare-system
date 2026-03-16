"""
ChromaDB client for persisting and retrieving diagnosis and treatment outcomes.

Two collections:
  - diagnosis_outcomes : stores validated diagnoses (indexed by patient symptoms)
  - treatment_outcomes : stores validated treatments (indexed by patient symptoms)

The lookup at orchestration start searches treatment_outcomes for semantically
similar cases. A cache hit short-circuits the full diagnosis/treatment pipeline.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Tuple

from log.logger import logger

# Minimum cosine similarity (0-1) to accept a cached treatment recommendation
SIMILARITY_THRESHOLD = 0.90

_diagnosis_collection = None
_treatment_collection = None


def _get_collections():
    """Lazy-initialise ChromaDB collections (non-blocking on failure)."""
    global _diagnosis_collection, _treatment_collection
    if _diagnosis_collection is not None and _treatment_collection is not None:
        return _diagnosis_collection, _treatment_collection

    try:
        import chromadb
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from core.config import CHROMA_HOST, CHROMA_PORT

        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # When ChromaDB runs as an external HTTP server, langchain-chroma cannot
        # get the distance metric from the collection metadata. So we supplied
        # the relevance score function explicitly.
        # text-embedding-3-small uses cosine distance (range 0–2); convert to
        # similarity in [0, 1] with:  similarity = 1 - distance / 2
        def _cosine_relevance_score_fn(distance: float) -> float:
            return 1.0 - distance / 2.0
        
        _diagnosis_collection = Chroma(
            client=chroma_client,
            collection_name="diagnosis_outcomes",
            embedding_function=embeddings,
            relevance_score_fn=_cosine_relevance_score_fn,
        )
        _treatment_collection = Chroma(
            client=chroma_client,
            collection_name="treatment_outcomes",
            embedding_function=embeddings,
            relevance_score_fn=_cosine_relevance_score_fn,
        )
        logger.info("[CHROMA] Collections initialised | server: %s:%d", CHROMA_HOST, CHROMA_PORT)
    except Exception as e:
        logger.warning("[CHROMA] Init failed (persistence disabled): %s", str(e))
        _diagnosis_collection = None
        _treatment_collection = None

    return _diagnosis_collection, _treatment_collection


async def lookup_treatment_recommendation(symptoms: str) -> Tuple[bool, Optional[dict]]:
    """
    Search treatment_outcomes for a semantically similar prior case.

    Returns:
        (True, cached_result)  — if similarity >= SIMILARITY_THRESHOLD
        (False, None)          — otherwise
    """
    _, treatment_col = _get_collections()
    if treatment_col is None:
        return False, None

    try:
        results = treatment_col.similarity_search_with_relevance_scores(
            query=symptoms,
            k=1,
        )
        if not results:
            logger.info("[CHROMA] treatment_outcomes: no documents yet")
            return False, None

        doc, score = results[0]
        logger.info("[CHROMA] Best treatment match | similarity: %.4f | threshold: %.2f", score, SIMILARITY_THRESHOLD)

        if score >= SIMILARITY_THRESHOLD:
            metadata = doc.metadata
            cached_result = {
                "source": "chroma_cache",
                "similarity_score": round(score, 4),
                "specialist_agent": metadata.get("specialist_agent"),
                "diagnosis_summary": metadata.get("diagnosis_summary"),
                "severity": metadata.get("severity", "N/A"),
                "treatment": json.loads(metadata.get("treatment_json", "{}")),
            }
            logger.info(
                "[CHROMA] Cache hit | similarity: %.4f | specialist: %s",
                score, metadata.get("specialist_agent"),
            )
            return True, cached_result

        logger.info("[CHROMA] No sufficient match | best similarity: %.4f", score)
        return False, None

    except Exception as e:
        logger.warning("[CHROMA] Lookup error (non-blocking): %s", str(e))
        return False, None


async def save_diagnosis_outcome(
    patient_id: str,
    symptoms: str,
    specialist_agent: str,
    diagnosis: dict,
) -> None:
    """Persist a validated diagnosis to the diagnosis_outcomes collection."""
    diagnosis_col, _ = _get_collections()
    if diagnosis_col is None:
        return

    try:
        metadata = {
            "patient_id": patient_id,
            "specialist_agent": specialist_agent,
            "severity": diagnosis.get("severity", "UNKNOWN"),
            "diagnosis_json": json.dumps(diagnosis),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        doc_id = f"diag_{patient_id}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        diagnosis_col.add_texts(
            texts=[symptoms],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.debug("[CHROMA] Diagnosis saved | patient: %s | id: %s", patient_id, doc_id)
    except Exception as e:
        logger.warning("[CHROMA] Diagnosis save failed (non-blocking): %s", str(e))


async def save_treatment_outcome(
    patient_id: str,
    symptoms: str,
    specialist_agent: str,
    diagnosis: dict,
    treatment: dict,
) -> None:
    """Persist a validated treatment to the treatment_outcomes collection."""
    _, treatment_col = _get_collections()
    if treatment_col is None:
        return

    try:
        diagnosis_summary = (
            diagnosis.get("diagnosysDetails") or   # cardiology typo preserved
            diagnosis.get("diagnosisDetails") or
            diagnosis.get("analysisDetails") or
            str(diagnosis)
        )
        metadata = {
            "patient_id": patient_id,
            "specialist_agent": specialist_agent,
            "diagnosis_summary": str(diagnosis_summary)[:500],   # keep metadata compact
            "severity": diagnosis.get("severity", "UNKNOWN"),
            "treatment_json": json.dumps(treatment),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        doc_id = f"treat_{patient_id}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        treatment_col.add_texts(
            texts=[symptoms],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.debug("[CHROMA] Treatment saved | patient: %s | id: %s", patient_id, doc_id)
    except Exception as e:
        logger.warning("[CHROMA] Treatment save failed (non-blocking): %s", str(e))
