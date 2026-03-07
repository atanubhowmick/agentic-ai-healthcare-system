from pydantic import BaseModel
from typing import List, Generic, TypeVar


class DiagnosisDetail(BaseModel):
    summary: str
    severity: str
    emergency_care_needed: str
    hospitalization_needed: str
    full_details: dict


class OrchestratorResponse(BaseModel):
    patient_id: str
    agent_id: str
    status: str                         # "COMPLETED" | "HUMAN_REVIEW_REQUIRED"
    specialist_agent: str | None
    diagnosis: DiagnosisDetail | None
    xai_diagnosis_validation: dict | None
    treatment: dict | None
    xai_treatment_validation: dict | None
    conflict_detected: bool
    conflict_reason: str
    human_review_reason: str | None
    audit_trail: List[str]


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
            error=ErrorResponse(code=error_code, message=error_message),
        )
