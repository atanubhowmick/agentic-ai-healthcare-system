from pydantic import BaseModel
from typing import List, Generic, TypeVar


class TreatmentRequest(BaseModel):
    patient_id: str
    diagnosis: str              # Diagnosis summary forwarded by the orchestrator
    specialist_notes: str       # Additional specialist context (agent name, hospitalization, etc.)
    is_followup: bool = False   # True when continuing a conversation about the care plan


class TreatmentResult(BaseModel):
    treatmentPlan: str                      # Detailed care plan within 200 words
    medications: List[str]                  # Each item: "drug name – dosage – frequency"
    followUpRequired: str                   # YES / NO
    followUpTimeframe: str                  # e.g. "1 week", "1 month", "3 months", "NONE"
    lifestyleRecommendations: List[str]     # Diet, exercise, stress management, etc.
    monitoringRequired: List[str]           # Parameters to track (BP, glucose, troponin, etc.)
    referralRequired: str                   # Specialist referral or "NONE"
    urgency: str                            # IMMEDIATE / SOON / ROUTINE


class TreatmentResponse(BaseModel):
    agent: str
    agent_id: str
    treatment: TreatmentResult


# -- Generic response wrapper (same pattern as all other agents) --------------

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
    def success_with_warning(cls, data: T, warn_code: str, warn_msg: str) -> "GenericResponse[T]":
        return cls(
            is_success=True,
            payload=data,
            warning=WarningResponse(code=warn_code, message=warn_msg),
        )

    @classmethod
    def failure(cls, error_code: str, error_message: str) -> "GenericResponse[T]":
        return cls(
            is_success=False,
            error=ErrorResponse(code=error_code, message=error_message),
        )
