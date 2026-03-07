from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from core.exceptions import (
    OrchestratorBaseException,
    GraphExecutionException,
    SpecialistUnavailableException,
)
from schemas.response import GenericResponse
from log.logger import logger


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(GraphExecutionException)
    async def graph_exec_handler(_request: Request, exc: GraphExecutionException) -> JSONResponse:
        logger.error("GraphExecutionException | %s: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=500, content=response.model_dump())

    @app.exception_handler(SpecialistUnavailableException)
    async def specialist_handler(_request: Request, exc: SpecialistUnavailableException) -> JSONResponse:
        logger.error("SpecialistUnavailableException | %s: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=503, content=response.model_dump())

    @app.exception_handler(OrchestratorBaseException)
    async def orchestrator_base_handler(_request: Request, exc: OrchestratorBaseException) -> JSONResponse:
        logger.error("OrchestratorBaseException | %s: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=500, content=response.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception | %s: %s", type(exc).__name__, str(exc))
        response = GenericResponse.failure(
            error_code="INTERNAL_SERVER_ERROR", error_message=str(exc)
        )
        return JSONResponse(status_code=500, content=response.model_dump())
