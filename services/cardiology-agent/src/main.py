from fastapi import FastAPI
from pydantic import BaseModel
from cardiology_agent import cardiology_executor

app = FastAPI(title="Cardiology Specialist Service")

class DiagnosisRequest(BaseModel):
    patient_id: str
    symptoms: str

@app.post("/diagnose")
async def diagnose_heart_condition(request: DiagnosisRequest):
    # Constructing the prompt for the agent
    query = f"Analyze patient {request.patient_id} with symptoms: {request.symptoms}. Check heart records for anomalies."
    
    result = cardiology_executor.invoke({"input": query, "chat_history": []})
    
    return {
        "agent": "Cardiology_Specialist",
        "diagnosis": result["output"],
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
    