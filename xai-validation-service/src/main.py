import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from explainers.shap_provider import preload_models
from guidelines.guideline_client import seed_guidelines


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_models()
    await asyncio.to_thread(seed_guidelines)
    yield


app = FastAPI(title="XAI Validator Service", lifespan=lifespan)

register_exception_handlers(app)
app.include_router(router)
