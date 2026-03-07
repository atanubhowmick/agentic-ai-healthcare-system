from pydantic import BaseModel


class OrchestratorRequest(BaseModel):
    patient_id: str
    symptoms: str
