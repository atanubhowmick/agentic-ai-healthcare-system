from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exception.exceptions import NeurologyBaseException, LLMInvocationException, LLMResponseParseException
from datamodel.models import GenericResponse
from log.logger import logger


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(LLMInvocationException)
    async def llm_invocation_exception_handler(_request: Request, exc: LLMInvocationException) -> JSONResponse:
        logger.error("LLMInvocationException caught | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=503, content=response.model_dump())

    @app.exception_handler(LLMResponseParseException)
    async def llm_response_parse_exception_handler(_request: Request, exc: LLMResponseParseException) -> JSONResponse:
        logger.error("LLMResponseParseException caught | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=422, content=response.model_dump())

    @app.exception_handler(NeurologyBaseException)
    async def neurology_exception_handler(_request: Request, exc: NeurologyBaseException) -> JSONResponse:
        logger.error("NeurologyBaseException caught | code: %s | message: %s", exc.error_code, exc.message)
        response = GenericResponse.failure(error_code=exc.error_code, error_message=exc.message)
        return JSONResponse(status_code=500, content=response.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception caught | type: %s | message: %s", type(exc).__name__, str(exc))
        response = GenericResponse.failure(error_code="INTERNAL_SERVER_ERROR", error_message=str(exc))
        return JSONResponse(status_code=500, content=response.model_dump())
