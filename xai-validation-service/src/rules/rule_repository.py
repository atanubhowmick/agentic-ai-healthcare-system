# Loads clinical safety rules into an in-memory cache at startup.
# Loading order: MongoDB (xai_validation_rules) → JSON file fallback (data/xai_validation_rules.json).
# The cache can be refreshed on demand without restarting the service.

from __future__ import annotations

import json
import os
from pathlib import Path

from pymongo import MongoClient
from pymongo import errors as pymongo_errors

from log.logger import logger

_COLLECTION = "xai_validation_rules"
_cache: list[dict] | None = None

# Path to the bundled JSON seed file (relative to this file's package root)
_JSON_SEED = Path(__file__).parent.parent.parent / "data" / "xai_validation_rules.json"



def _get_collection():
    from core.config import MONGODB_URI, MONGODB_DB_NAME
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    return client[MONGODB_DB_NAME][_COLLECTION]


def _load_from_mongo() -> list[dict] | None:
    """
    Try to load active rules from MongoDB.
    Returns the rule list if the collection has documents, None otherwise.
    """
    try:
        col = _get_collection()
        count = col.count_documents({})
        if count == 0:
            logger.info("[RULE_REPO] MongoDB collection '%s' is empty — will use JSON fallback", _COLLECTION)
            return None
        # Query excludes only documents explicitly set to false.
        # This handles boolean true, string "true", and missing active field.
        rules = list(col.find({"active": {"$ne": False}}, {"_id": 0}))
        logger.info("[RULE_REPO] Loaded %d active rule(s) from MongoDB collection '%s'", len(rules), _COLLECTION)
        return rules
    except pymongo_errors.ServerSelectionTimeoutError:
        logger.warning("[RULE_REPO] MongoDB unreachable — falling back to JSON seed file")
        return None
    except Exception as exc:
        logger.warning("[RULE_REPO] MongoDB load error: %s — falling back to JSON seed file", exc)
        return None


def _load_from_json() -> list[dict]:
    """Load rules from the bundled JSON seed file."""
    if not _JSON_SEED.exists():
        logger.error("[RULE_REPO] JSON seed file not found at %s — no rules loaded", _JSON_SEED)
        return []
    try:
        with open(_JSON_SEED, encoding="utf-8") as f:
            rules = json.load(f)
        active = [r for r in rules if r.get("active", True)]
        logger.info("[RULE_REPO] Loaded %d active rule(s) from JSON seed file", len(active))
        return active
    except Exception as exc:
        logger.error("[RULE_REPO] Failed to load JSON seed file: %s", exc)
        return []



def load_rules(refresh: bool = False) -> list[dict]:
    """
    Return all active clinical safety rules (in-memory cache).

    Called once at startup. Subsequent calls return the cached list.
    Pass refresh=True to re-query MongoDB (e.g. after an admin rule update).

    Loading order: MongoDB → JSON file fallback.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    rules = _load_from_mongo()
    if rules is None:
        rules = _load_from_json()

    _cache = rules
    return _cache


def count_rules_by_source() -> dict[str, int]:
    """Return rule counts grouped by source (for reporting)."""
    counts: dict[str, int] = {}
    for rule in load_rules():
        src = rule.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts
