from pydantic import BaseModel
from typing import List


class DiagnosisResult(BaseModel):
    diagnosysDetails: str
    severity: str
    hospitalizationNeeded: str
    emergencyCareNeeded: str
    clarificationQuestion: str
    bloodTestsRequired: List[str]
    labTestsRequired: List[str]
    medication: str


class DiagnosisResponse(BaseModel):
    agent: str
    diagnosis: DiagnosisResult
    status: str
