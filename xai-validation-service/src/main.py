from fastapi import FastAPI
from src.explainers.shap_provider import HeartSHAPExplainer
from src.validators.ethical_guard import validate_ethics
from pydantic import BaseModel

app = FastAPI(title="XAI Validator Service")
explainer = HeartSHAPExplainer("models/cardiology_v1.pkl")

class XAIRequest(BaseModel):
    patient_id: str
    diagnosis: str
    features: dict # Features used for SHAP
    symptoms: str

@app.post("/validate")
async def validate_decision(request: XAIRequest):
    # 1. Fidelity Check: Why was this diagnosis made? [cite: 446]
    top_features = explainer.explain_diagnosis(request.features)
    
    # 2. Ethical Check: Is it safe? [cite: 368]
    is_safe, message = validate_ethics(request.diagnosis, request.symptoms)
    
    # 3. Consistency Check [cite: 388]
    # If safety fails, trigger 'Revised Decision-Making' in Orchestrator [cite: 297]
    return {
        "is_validated": is_safe,
        "explanation": {
            "top_contributing_factors": top_features,
            "safety_message": message
        },
        "status": "Validated" if is_safe else "Rejected"
    }
    