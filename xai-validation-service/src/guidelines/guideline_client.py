"""
ChromaDB client for clinical guideline RAG.

Abstracts are fetched from PubMed at first startup and stored in the
'clinical_guidelines' collection. Subsequent restarts skip the fetch.
Used by the constitutional guard to ground P5 (guideline alignment) checks.
"""

import threading
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from core.config import CHROMA_HOST, CHROMA_PORT
from guidelines.pubmed_fetcher import fetch_guidelines
from log.logger import logger

_guidelines_collection = None
_guidelines_lock = threading.Lock()
_guidelines_seeded = False

_GUIDELINE_SIMILARITY_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Collection initialisation (lazy, thread-safe singleton)
# ---------------------------------------------------------------------------

def _cosine_relevance_score_fn(distance: float) -> float:
    return 1.0 - distance / 2.0


def _get_collection():
    global _guidelines_collection
    if _guidelines_collection is not None:
        return _guidelines_collection
    with _guidelines_lock:
        if _guidelines_collection is not None:
            return _guidelines_collection
        try:
            logger.info("[GUIDELINES] Connecting to ChromaDB at %s:%s...", CHROMA_HOST, CHROMA_PORT)
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            logger.info("[GUIDELINES] ChromaDB HttpClient created.")

            logger.info("[GUIDELINES] Initialising OpenAI embeddings...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("[GUIDELINES] OpenAI embeddings ready.")

            logger.info("[GUIDELINES] Creating Chroma collection handle...")
            _guidelines_collection = Chroma(
                client=chroma_client,
                collection_name="clinical_guidelines",
                embedding_function=embeddings,
                relevance_score_fn=_cosine_relevance_score_fn,
            )
            logger.info("[GUIDELINES] ChromaDB collection initialised: clinical_guidelines")
        except Exception as exc:
            logger.warning("[GUIDELINES] ChromaDB init failed — guideline RAG disabled: %s", exc)
            _guidelines_collection = None
        return _guidelines_collection


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def seed_guidelines() -> None:
    """
    Seed clinical_guidelines collection with PubMed-fetched abstracts.
    Idempotent — skips seeding if the collection already has documents.
    Safe to call at startup (runs in background thread).
    """
    global _guidelines_seeded
    if _guidelines_seeded:
        return
    logger.info("[GUIDELINES] Starting guideline seed...")
    # Call _get_collection() BEFORE acquiring _guidelines_lock to avoid deadlock
    # (_get_collection also acquires _guidelines_lock internally)
    col = _get_collection()
    with _guidelines_lock:
        if _guidelines_seeded:
            return
        if col is None:
            logger.warning("[GUIDELINES] ChromaDB unavailable — skipping guideline seed. Is ChromaDB running on port %d?", CHROMA_PORT)
            _guidelines_seeded = True
            return
        try:
            existing_count = col._collection.count()
            if existing_count > 0:
                logger.info("[GUIDELINES] Collection already has %d guideline(s) — skipping seed.", existing_count)
                _guidelines_seeded = True
                return

            # Fetch from PubMed
            guidelines: list[dict] = []
            try:
                logger.info("[GUIDELINES] Fetching clinical guidelines from PubMed...")
                guidelines = fetch_guidelines()
            except Exception as exc:
                logger.warning("[GUIDELINES] PubMed fetch failed: %s — skipping seed.", exc)

            if not guidelines:
                logger.warning("[GUIDELINES] No guidelines fetched — seed skipped. Will retry on next restart.")
                _guidelines_seeded = True
                return

            texts = [g["text"] for g in guidelines]
            metadatas = [
                {
                    "source": g.get("source", "Unknown"),
                    "specialty": g.get("specialty", ""),
                    "topic": g.get("topic", ""),
                    "pmid": g.get("pmid", ""),
                }
                for g in guidelines
            ]
            ids = [f"guideline_{i}" for i in range(len(guidelines))]
            col.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info("[GUIDELINES] Seeded %d clinical guidelines into ChromaDB.", len(guidelines))
        except Exception as exc:
            logger.warning("[GUIDELINES] Seed failed: %s", exc)
        _guidelines_seeded = True


def search_guidelines(query: str, k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant clinical guidelines for a query.

    Args:
        query: Free-text clinical context (diagnosis + symptoms + treatment).
        k:     Maximum number of results to return.

    Returns:
        List of dicts: [{"text": ..., "source": ..., "specialty": ..., "pmid": ..., "score": ...}]
        Returns empty list if ChromaDB is unavailable or no match above threshold.
    """
    col = _get_collection()
    if col is None:
        return []
    try:
        results = col.similarity_search_with_relevance_scores(query=query, k=k)
        guidelines = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "specialty": doc.metadata.get("specialty", ""),
                "pmid": doc.metadata.get("pmid", ""),
                "score": round(score, 3),
            }
            for doc, score in results
            if score >= _GUIDELINE_SIMILARITY_THRESHOLD
        ]
        logger.debug(
            "[GUIDELINES] Retrieved %d relevant guideline(s) | query: %.60s",
            len(guidelines), query,
        )
        return guidelines
    except Exception as exc:
        logger.warning("[GUIDELINES] Search failed: %s", exc)
        return []
