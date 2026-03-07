from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers

app = FastAPI(title="Healthcare Orchestrator Agent")

register_exception_handlers(app)
app.include_router(router)
