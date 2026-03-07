from pydantic import BaseModel
from typing import List, Generic, TypeVar


class DiagnosisRequest(BaseModel):
    patient_id: str
    symptoms: str
    is_followup: bool = False   # True when patient is answering a clarificationQuestion


class DiagnosisResult(BaseModel):
    diagnosisDetails: str           # Detailed oncology assessment within 200 words
    suspectedCancerType: str        # e.g. "Lung Adenocarcinoma", "Breast Cancer (HER2+)", "Unknown"
    stage: str                      # TNM staging if determinable, otherwise "Undetermined"
    severity: str                   # LOW / HIGH / CRITICAL
    hospitalizationNeeded: str      # YES / NO
    emergencyCareNeeded: str        # YES / NO
    clarificationQuestion: str      # Clarification needed from patient, within 100 words
    biomarkersRequired: List[str]   # Tumor markers / genetic tests (PSA, CA-125, CEA, BRCA1/2, etc.)
    imagingRequired: List[str]      # CT, PET-CT, MRI, bone scan, etc.
    biopsyRequired: str             # Type of biopsy if needed, otherwise "NOT REQUIRED"
    oncologyReferralNeeded: str     # Referral type (Medical Oncology, Radiation Oncology, etc.) or "NONE"
    medication: str                 # Initial symptom management or targeted therapy recommendations


class DiagnosisResponse(BaseModel):
    agent: str
    agent_id: str
    diagnosis: DiagnosisResult


T = TypeVar('T')


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
