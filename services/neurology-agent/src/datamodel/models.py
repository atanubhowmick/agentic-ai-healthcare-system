from pydantic import BaseModel
from typing import List, Generic, TypeVar


class NeurologyRequest(BaseModel):
    patient_id: str
    symptoms: str
    is_followup: bool = False   # True when patient is answering a clarificationQuestion


class NeurologyResult(BaseModel):
    diagnosisDetails: str
    severity: str
    hospitalizationNeeded: str
    emergencyCareNeeded: str
    clarificationQuestion: str
    bloodTestsRequired: List[str]
    imagingRequired: List[str]
    neurologicalTestsRequired: List[str]
    medication: str


class NeurologyResponse(BaseModel):
    agent: str
    agent_id: str
    diagnosis: NeurologyResult


# Use of generics instead of any particular response type
T = TypeVar('T')

class ErrorResponse(BaseModel):
    code: str
    message: str

class WarningResponse(BaseModel):
    code: str
    message: str

class GenericResponse(BaseModel, Generic[T]):

    is_success: bool

    # Python 3.10+ syntax: use '|' instead of Optional
    payload: T | None = None
    error: ErrorResponse | None = None
    warning: WarningResponse | None = None

    @classmethod
    def success(cls, data: T) -> "GenericResponse[T]":
        return cls(is_success=True, payload=data)

    @classmethod
    def success_with_warning(cls, data: T, warn_code: str, warn_msg: str) -> "GenericResponse[T]":
        return cls(
            is_success=True,
            payload=data,
            warning=WarningResponse(code=warn_code, message=warn_msg)
        )

    @classmethod
    def failure(cls, error_code: str, error_message: str) -> "GenericResponse[T]":
        return cls(
            is_success=False,
            error=ErrorResponse(code=error_code, message=error_message)
        )
