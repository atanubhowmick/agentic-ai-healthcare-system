from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import ClinicalRAG

app = FastAPI(title="Treatment & Patient Care Service")

# Initialize RAG with the clinical guidelines vector database
rag_system = ClinicalRAG("data/vector_store/clinical_guidelines")

class TreatmentRequest(BaseModel):
    patient_id: str
    diagnosis: str
    specialist_notes: str

@app.post("/recommend")
async def recommend_treatment(request: TreatmentRequest):
    """
    Synthesizes specialist findings into a patient care plan.
    """
    # 1. Retrieve evidence-based protocol from Vector DB
    guidelines = rag_system.get_treatment_guidelines(request.diagnosis)
    
    # 2. Formulate personalized care plan
    # This aligns with Objective 2: medication or patient care decision making
    care_plan = (
        f"Based on the diagnosis '{request.diagnosis}' and specialist notes: "
        f"'{request.specialist_notes}', we recommend the following protocol: {guidelines}"
    )
    
    return {
        "agent": "Treatment_Care_Agent",
        "patient_id": request.patient_id,
        "recommendation": care_plan,
        "status": "Success"
    }
    