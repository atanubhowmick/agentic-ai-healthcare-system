import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from rag.tfidf_predictor import warm_up
from log.logger import logger

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tfidf-warmup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Triggering TF-IDF model warm-up in background thread...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, warm_up)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Cancer Oncology Specialist Service", lifespan=lifespan)

register_exception_handlers(app)
app.include_router(router)
