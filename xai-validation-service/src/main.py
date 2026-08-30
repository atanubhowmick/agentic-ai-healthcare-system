import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from core.tracing import LangSmithTracingMiddleware
from explainers.shap_provider import preload_models
from guidelines.guideline_client import seed_guidelines
from rules.rule_repository import load_rules
from rules.rule_generator import seed_llm_rules


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Preload SHAP models
    preload_models()

    # 2. Seed clinical guidelines into ChromaDB (PubMed abstracts)
    await asyncio.to_thread(seed_guidelines)

    # 3. Load clinical safety rules into memory (MongoDB → JSON fallback)
    await asyncio.to_thread(load_rules)

    # 4. Generate LLM-extracted rules from guideline abstracts (skips if already done)
    await asyncio.to_thread(seed_llm_rules)

    yield


app = FastAPI(title="XAI Validator Service", lifespan=lifespan)

app.add_middleware(LangSmithTracingMiddleware, service_name="xai-validation-service")
register_exception_handlers(app)
app.include_router(router)
