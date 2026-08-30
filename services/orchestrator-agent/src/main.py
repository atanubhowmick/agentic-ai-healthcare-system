from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from core.tracing import LangSmithTracingMiddleware
from agents.classifier_router import warm_up_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    await warm_up_models()
    yield


app = FastAPI(title="Healthcare Orchestrator Agent", lifespan=lifespan)

app.add_middleware(LangSmithTracingMiddleware, service_name="orchestrator-agent")
register_exception_handlers(app)
app.include_router(router)
