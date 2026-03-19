"""
MIMIC-IV Cancer Case Retriever.

Searches a ChromaDB collection populated from MIMIC-IV oncology cases and
returns the top-k most similar historical cases for a given patient query.
These cases are injected into the LLM prompt as RAG context.

Collection schema (set by load_mimic_data.py):
  document  : symptom / presentation text extracted from discharge notes
  metadata  : {
      cancer_type      : str   - ICD-10 long title (e.g. "Malignant neoplasm of upper lobe, bronchus or lung")
      icd_codes        : str   - comma-separated ICD-10 codes (e.g. "C34.12, C78.01")
      chief_complaint  : str   - Chief Complaint section from discharge note
      treatment_summary: str   - Assessment & Plan section from discharge note
      lab_findings     : str   - key lab findings (if available)
      severity         : str   - derived from discharge disposition (CRITICAL / HIGH / LOW)
      source           : str   - always "MIMIC-IV"
  }
"""

from typing import List
from log.logger import logger

_collection = None


def _get_collection():
    """Lazy-initialise the ChromaDB collection (non-blocking on failure)."""
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from core.config import CHROMA_HOST, CHROMA_PORT, MIMIC_COLLECTION_NAME

        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _collection = Chroma(
            client=chroma_client,
            collection_name=MIMIC_COLLECTION_NAME,
            embedding_function=embeddings,
        )
        has_docs = len(_collection.get(limit=1)["ids"]) > 0
        logger.info("[MIMIC_RETRIEVER] Collection initialised | has_docs: %s", has_docs)
    except Exception as e:
        logger.warning("[MIMIC_RETRIEVER] Init failed (MIMIC RAG disabled): %s", str(e))
        _collection = None

    return _collection


def search_similar_cases(symptoms: str, top_k: int = 3) -> List[dict]:
    """
    Search MIMIC-IV oncology cases semantically similar to the given symptoms.

    Returns a list of dicts:
      {
          "cancer_type"      : str,
          "icd_codes"        : str,
          "chief_complaint"  : str,
          "treatment_summary": str,
          "lab_findings"     : str,
          "severity"         : str,
          "similarity_score" : float,
          "source"           : "MIMIC-IV",
      }
    Empty list if the collection is unavailable or no results found.
    """
    col = _get_collection()
    if col is None:
        return []

    try:
        results = col.similarity_search_with_relevance_scores(
            query=symptoms,
            k=top_k,
        )
        cases = []
        for doc, score in results:
            m = doc.metadata
            cases.append({
                "cancer_type":       m.get("cancer_type", "Unknown"),
                "icd_codes":         m.get("icd_codes", ""),
                "chief_complaint":   m.get("chief_complaint", ""),
                "treatment_summary": m.get("treatment_summary", ""),
                "lab_findings":      m.get("lab_findings", ""),
                "severity":          m.get("severity", "UNKNOWN"),
                "similarity_score":  round(score, 4),
                "source":            "MIMIC-IV",
            })
        logger.info(
            "[MIMIC_RETRIEVER] %d case(s) retrieved | top score: %.4f",
            len(cases), cases[0]["similarity_score"] if cases else 0.0,
        )
        return cases
    except Exception as e:
        logger.warning("[MIMIC_RETRIEVER] Search error: %s", str(e))
        return []


def is_collection_populated() -> bool:
    """Return True if the MIMIC ChromaDB collection has at least one document."""
    col = _get_collection()
    if col is None:
        return False
    try:
        return len(col.get(limit=1)["ids"]) > 0
    except Exception:
        return False
