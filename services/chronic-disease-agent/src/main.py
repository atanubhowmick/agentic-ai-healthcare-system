from fastapi import FastAPI
from pydantic import BaseModel
from src.chronic_agent import chronic_executor

app = FastAPI(title="Chronic Disease Specialist Service")

class ChronicRequest(BaseModel):
    patient_id: str
    symptoms: str

@app.post("/diagnose")
async def diagnose_chronic_condition(request: ChronicRequest):
    """
    Endpoint for the Master Router to initiate chronic analysis.
    """
    query = (
        f"Analyze patient {request.patient_id} for chronic risks. "
        f"Symptoms: {request.symptoms}. Check historical trends."
    )
    
    # Execute autonomous diagnosis [cite: 188]
    result = chronic_executor.invoke({"input": query, "chat_history": []})
    
    return {
        "agent": "Chronic_Disease_Specialist",
        "diagnosis": result["output"],
        "status": "Success"
    }
    