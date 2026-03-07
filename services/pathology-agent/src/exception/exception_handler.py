from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exception.exceptions import PathologySvcException
from datamodel.models import GenericResponse
from log.logger import logger


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(PathologySvcException)
    async def pathology_exception_handler(_request: Request, exc: PathologySvcException) -> JSONResponse:
        logger.error("PathologySvcException caught | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=500, content=response.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception caught | type: %s | message: %s", type(exc).__name__, str(exc))
        response = GenericResponse.failure(error_code="INTERNAL_SERVER_ERROR", error_message=str(exc))
        return JSONResponse(status_code=500, content=response.model_dump())
