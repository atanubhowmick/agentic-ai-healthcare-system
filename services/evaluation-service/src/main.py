from fastapi import FastAPI
from api.server import router
from exception.exception_handler import register_exception_handlers

app = FastAPI(title="Evaluation Service")

register_exception_handlers(app)
app.include_router(router)
