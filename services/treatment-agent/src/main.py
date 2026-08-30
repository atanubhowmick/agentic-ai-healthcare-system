from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from core.tracing import LangSmithTracingMiddleware

app = FastAPI(title="Treatment & Patient Care Agent")

app.add_middleware(LangSmithTracingMiddleware, service_name="treatment-agent")
register_exception_handlers(app)
app.include_router(router)
