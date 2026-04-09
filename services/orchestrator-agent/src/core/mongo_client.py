# Async MongoDB persistence. Non-blocking — service continues if MongoDB is unavailable.

import asyncio
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient

from core.config import MONGO_URI, MONGO_DB
from log.logger import logger

_client = None


def _get_db():
    global _client
    if _client is None:
        try:
            _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            return _client[MONGO_DB]
        except Exception as e:
            logger.warning("MongoDB client init failed (persistence disabled): %s", str(e))
            return None
    return _client[MONGO_DB]


async def save_case(case_data: dict) -> None:
    """Persist a completed orchestration case to MongoDB (fire-and-forget)."""
    db = _get_db()
    if db is None:
        return
    try:
        doc = {**case_data, "saved_at": datetime.now(timezone.utc).isoformat()}
        result = await db.cases.insert_one(doc)
        logger.debug("Case saved to MongoDB | id: %s", str(result.inserted_id))
    except Exception as e:
        logger.warning("MongoDB save failed (non-blocking): %s", str(e))
