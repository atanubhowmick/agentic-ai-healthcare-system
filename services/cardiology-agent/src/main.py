import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cardiology_agent import cardiology_executor
from models import DiagnosisResult, DiagnosisResponse

app = FastAPI(title="Cardiology Specialist Service")


class DiagnosisRequest(BaseModel):
    patient_id: str
    symptoms: str
    is_followup: bool = False   # True when patient is answering a clarificationQuestion


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences if present and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_heart_condition(request: DiagnosisRequest):
    # Initial request: wrap with clinical context
    # Follow-up: send the patient's answer as-is so the LLM reads it
    # against the prior conversation history that is automatically injected
    if request.is_followup:
        query = request.symptoms   # patient's answer to the clarificationQuestion
    else:
        query = (
            f"Analyze patient {request.patient_id} with symptoms: {request.symptoms}. "
            "Check for cardiac anomalies and respond strictly in the requested JSON format."
        )

    result = cardiology_executor.invoke(
        {"input": query},
        config={"configurable": {"session_id": request.patient_id}}
    )

    try:
        raw = _parse_llm_json(result.content)
        diagnosis = DiagnosisResult(**raw)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {e}")

    return DiagnosisResponse(
        agent="Cardiology_Specialist",
        diagnosis=diagnosis,
        status="Success"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
