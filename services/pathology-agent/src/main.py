from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers
from core.tracing import LangSmithTracingMiddleware

app = FastAPI(title="Pathology Specialist Service")

app.add_middleware(LangSmithTracingMiddleware, service_name="pathology-agent")
register_exception_handlers(app)
app.include_router(router)
