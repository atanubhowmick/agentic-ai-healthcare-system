from pydantic import BaseModel
from typing import List, Generic, TypeVar


class DiagnosisValidationRequest(BaseModel):
    patient_id: str
    symptoms: str
    specialist_agent: str           # e.g. "Cardiology_Specialist"
    diagnosis: dict                 # Full specialist diagnosis dict (varies by agent)


class TreatmentValidationRequest(BaseModel):
    patient_id: str
    specialist_agent: str
    diagnosis_summary: str
    severity: str
    treatment_recommendation: str


class ValidationResult(BaseModel):
    is_validated: bool
    confidence_score: float         # 0.0 – 1.0
    validation_summary: str
    key_concerns: List[str]
    recommendation: str             # "APPROVE" | "REJECT" | "REVIEW"


class ValidationResponse(BaseModel):
    agent: str
    agent_id: str
    patient_id: str
    validation_type: str            # "DIAGNOSIS" | "TREATMENT"
    result: ValidationResult


T = TypeVar("T")


class ErrorResponse(BaseModel):
    code: str
    message: str


class GenericResponse(BaseModel, Generic[T]):
    is_success: bool
    payload: T | None = None
    error: ErrorResponse | None = None

    @classmethod
    def success(cls, data: T) -> "GenericResponse[T]":
        return cls(is_success=True, payload=data)

    @classmethod
    def failure(cls, error_code: str, error_message: str) -> "GenericResponse[T]":
        return cls(
            is_success=False,
            error=ErrorResponse(code=error_code, message=error_message)
        )
