from typing import Generic, TypeVar

from pydantic import BaseModel


class TfidfBaselineRequest(BaseModel):
    max_cases: int = 0                 # 0 = use all available cases for train/test split
    test_size: float = 0.20            # fraction of data reserved for testing (e.g. 0.25 for 75:25)


class EvaluationStatusResponse(BaseModel):
    running: bool
    report_available: bool


T = TypeVar("T")


class ErrorResponse(BaseModel):
    code: str
    message: str


class WarningResponse(BaseModel):
    code: str
    message: str


class GenericResponse(BaseModel, Generic[T]):
    is_success: bool
    payload: T | None = None
    error: ErrorResponse | None = None
    warning: WarningResponse | None = None

    @classmethod
    def success(cls, data: T) -> "GenericResponse[T]":
        return cls(is_success=True, payload=data)

    @classmethod
    def failure(cls, error_code: str, error_message: str) -> "GenericResponse[T]":
        return cls(
            is_success=False,
            error=ErrorResponse(code=error_code, message=error_message),
        )
