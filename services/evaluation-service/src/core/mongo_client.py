"""
Synchronous MongoDB client for the Evaluation Service.
Provides helpers to persist and retrieve MIMIC-IV evaluation cases and
TF-IDF evaluation reports.
"""

import datetime
from typing import Optional

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection

from core.config import (
    MONGO_DB, MONGO_EVAL_COLLECTION, MONGO_TFIDF_REPORT_COLLECTION,
    MONGO_XAI_REPORT_COLLECTION, MONGO_URI,
)
from log.logger import logger

# ---------------------------------------------------------------------------
# Module-level client (lazy-initialised, reused across calls)
# ---------------------------------------------------------------------------

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        logger.info("[MONGO] Connected to %s / %s", MONGO_URI, MONGO_DB)
    return _client


def _get_collection(name: str) -> Collection:
    return _get_client()[MONGO_DB][name]


# ---------------------------------------------------------------------------
# Evaluation cases — write
# ---------------------------------------------------------------------------

def save_evaluation_cases(records: list[dict]) -> int:
    """
    Bulk-upsert evaluation records.
    Upserted on (subject_id, hadm_id) so re-running is idempotent.
    Returns total number of upserted documents.
    """
    if not records:
        logger.warning("[MONGO] save_evaluation_cases called with empty list — nothing saved.")
        return 0

    col = _get_collection(MONGO_EVAL_COLLECTION)
    col.create_index(
        [("subject_id", pymongo.ASCENDING), ("hadm_id", pymongo.ASCENDING)],
        unique=True,
        background=True,
    )

    ops = [
        pymongo.UpdateOne(
            {"subject_id": rec.get("subject_id"), "hadm_id": rec.get("hadm_id")},
            {"$set": rec},
            upsert=True,
        )
        for rec in records
    ]

    result = col.bulk_write(ops, ordered=False)
    logger.info(
        "[MONGO] Upserted %d | modified %d | total processed %d",
        result.upserted_count, result.modified_count, len(records),
    )
    return result.upserted_count + result.modified_count


# ---------------------------------------------------------------------------
# Evaluation cases — read
# ---------------------------------------------------------------------------

def load_evaluation_cases(max_cases: int = 0) -> list[dict]:
    """
    Load evaluation records from MongoDB.

    Args:
        max_cases: If > 0, return only the first N documents.

    Returns:
        List of record dicts (MongoDB _id field removed).
    """
    col   = _get_collection(MONGO_EVAL_COLLECTION)
    query = col.find({}, {"_id": 0})
    if max_cases and max_cases > 0:
        query = query.limit(max_cases)

    records = list(query)
    logger.info("[MONGO] Loaded %d evaluation cases", len(records))
    return records


def count_evaluation_cases() -> int:
    """Return the total number of evaluation cases stored in MongoDB."""
    return _get_collection(MONGO_EVAL_COLLECTION).count_documents({})


# ---------------------------------------------------------------------------
# TF-IDF baseline reports — write / read
# ---------------------------------------------------------------------------

def save_tfidf_report(report: dict) -> None:
    """Insert a TF-IDF baseline report with a UTC timestamp."""
    col = _get_collection(MONGO_TFIDF_REPORT_COLLECTION)
    col.create_index([("run_at", pymongo.DESCENDING)], background=True)
    doc = {"run_at": datetime.datetime.utcnow(), **report}
    col.insert_one(doc)
    logger.info("[MONGO] TF-IDF baseline report saved to '%s'", MONGO_TFIDF_REPORT_COLLECTION)


def load_latest_tfidf_report() -> Optional[dict]:
    """Return the most recent TF-IDF baseline report, or None."""
    col = _get_collection(MONGO_TFIDF_REPORT_COLLECTION)
    doc = col.find_one({}, {"_id": 0}, sort=[("run_at", pymongo.DESCENDING)])
    if doc and isinstance(doc.get("run_at"), datetime.datetime):
        doc["run_at"] = doc["run_at"].isoformat() + "Z"
    return doc


def has_tfidf_report() -> bool:
    """Return True if at least one TF-IDF baseline report exists."""
    return _get_collection(MONGO_TFIDF_REPORT_COLLECTION).count_documents({}, limit=1) > 0


# ---------------------------------------------------------------------------
# XAI evaluation reports — write / read
# ---------------------------------------------------------------------------

def save_xai_report(report: dict) -> None:
    """Insert an XAI evaluation report with a UTC timestamp."""
    col = _get_collection(MONGO_XAI_REPORT_COLLECTION)
    col.create_index([("run_at", pymongo.DESCENDING)], background=True)
    doc = {"run_at": datetime.datetime.utcnow(), **report}
    col.insert_one(doc)
    logger.info("[MONGO] XAI evaluation report saved to '%s'", MONGO_XAI_REPORT_COLLECTION)


def load_latest_xai_report() -> Optional[dict]:
    """Return the most recent XAI evaluation report, or None."""
    col = _get_collection(MONGO_XAI_REPORT_COLLECTION)
    doc = col.find_one({}, {"_id": 0}, sort=[("run_at", pymongo.DESCENDING)])
    if doc and isinstance(doc.get("run_at"), datetime.datetime):
        doc["run_at"] = doc["run_at"].isoformat() + "Z"
    return doc


def has_xai_report() -> bool:
    """Return True if at least one XAI evaluation report exists."""
    return _get_collection(MONGO_XAI_REPORT_COLLECTION).count_documents({}, limit=1) > 0
