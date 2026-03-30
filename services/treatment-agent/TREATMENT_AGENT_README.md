# Treatment Agent

Treatment and patient care agent — port **8012**. Generates comprehensive treatment plans based on specialist diagnoses, including medications, lifestyle recommendations, monitoring requirements, and follow-up schedules.

---

## How It Works

Uses `deepagents.create_deep_agent` (LangGraph-based) with a treatment planning system prompt. Takes a specialist diagnosis as input and produces a structured treatment plan with urgency, medications, referrals, and monitoring requirements.

---

## API Endpoint

### `POST /treatment-agent/recommend`

**Request**
```json
{
  "patient_id": "P001",
  "diagnosis": "Acute STEMI with cardiogenic shock",
  "specialist_notes": "Cardiology_Specialist - hospitalization required, emergency care needed",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Treatment_Care_Agent",
    "agent_id": "TREAT-AGENT-1004",
    "treatment": {
      "treatmentPlan": "...",
      "medications": [
        "Aspirin - 325mg - once daily",
        "Heparin - weight-based - IV infusion"
      ],
      "followUpRequired": "YES",
      "followUpTimeframe": "1 week",
      "lifestyleRecommendations": ["Low-sodium diet", "Cardiac rehabilitation"],
      "monitoringRequired": ["BP", "Troponin", "ECG"],
      "referralRequired": "Cardiac surgeon",
      "urgency": "IMMEDIATE"
    }
  }
}
```

---

## Running Locally

```bash
cd services/treatment-agent

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8012 --reload
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model |

---

## Source Structure

```
src/
├── agent/      # DeepAgent executor + system prompt
├── api/        # FastAPI router
├── core/       # config.py
├── datamodel/  # Pydantic models
├── exception/  # TreatmentSvcException + handler
├── service/    # treatment_service.py
├── log/
└── main.py
```
