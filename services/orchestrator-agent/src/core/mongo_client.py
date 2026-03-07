"""
Async MongoDB persistence for orchestrated cases.
Non-blocking: if MongoDB is unavailable the service continues without crashing.
"""

import asyncio
from datetime import datetime, timezone
from log.logger import logger

_client = None


def _get_db():
    global _client
    if _client is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from core.config import MONGO_URI, MONGO_DB
            _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            return _client[MONGO_DB]
        except Exception as e:
            logger.warning("MongoDB client init failed (persistence disabled): %s", str(e))
            return None
    from core.config import MONGO_DB
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
