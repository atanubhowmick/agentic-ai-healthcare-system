from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exception.exceptions import XAIBaseException, ValidationLLMException, ValidationParseException
from datamodel.models import GenericResponse
from log.logger import logger


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(ValidationLLMException)
    async def validation_llm_handler(_request: Request, exc: ValidationLLMException) -> JSONResponse:
        logger.error("ValidationLLMException | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=503, content=response.model_dump())

    @app.exception_handler(ValidationParseException)
    async def validation_parse_handler(_request: Request, exc: ValidationParseException) -> JSONResponse:
        logger.error("ValidationParseException | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=422, content=response.model_dump())

    @app.exception_handler(XAIBaseException)
    async def xai_base_handler(_request: Request, exc: XAIBaseException) -> JSONResponse:
        logger.error("XAIBaseException | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=500, content=response.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception | type: %s | message: %s", type(exc).__name__, str(exc))
        response = GenericResponse.failure(error_code="INTERNAL_SERVER_ERROR", error_message=str(exc))
        return JSONResponse(status_code=500, content=response.model_dump())
