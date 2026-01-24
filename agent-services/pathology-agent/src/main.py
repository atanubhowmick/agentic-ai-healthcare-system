from fastapi import FastAPI
from pydantic import BaseModel
from src.agent_logic import PathologyAnalyst

app = FastAPI(title="Pathology Specialist Service")

# Initialize the analyst with the path to lab data
analyst = PathologyAnalyst("data/lab_results.csv")

class PathologyRequest(BaseModel):
    patient_id: str
    query: str

@app.post("/analyze")
async def analyze_pathology(request: PathologyRequest):
    """
    Endpoint for the Master Router to request lab report insights.
    """
    analysis_output = await analyst.analyze_lab_reports(
        request.patient_id, 
        request.query
    )
    
    return {
        "agent": "Pathology_Specialist",
        "patient_id": request.patient_id,
        "analysis": analysis_output,
        "status": "Success"
    }
    