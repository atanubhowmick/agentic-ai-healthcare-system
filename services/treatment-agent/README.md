# Treatment Agent

A FastAPI microservice that generates comprehensive treatment and patient care plans. It receives a specialist diagnosis forwarded by the orchestrator and uses a DeepAgent (LangGraph-based) backed by an OpenAI LLM to produce a structured plan — covering medications, monitoring, lifestyle advice, referrals, and urgency.

Runs on **port 8012**. Always called after specialist diagnosis and XAI validation in the orchestrator pipeline.

---

## How It Works

1. The `/treatment-agent/recommend` endpoint receives a patient ID, diagnosis summary, and specialist notes.
2. `treatment_service` builds a query (or uses the `diagnosis` field as free-text for follow-ups) and maintains per-patient chat history in memory.
3. The DeepAgent invokes the LLM with the system prompt and session history, then returns a JSON treatment plan.
4. The response is validated against the `TreatmentResult` Pydantic model and returned wrapped in a `GenericResponse`.

**Urgency levels:** `IMMEDIATE` = care required within hours, `SOON` = within days, `ROUTINE` = weeks/scheduled.

Follow-up calls (where `is_followup: true`) pass the `diagnosis` field directly as free-text, continuing the session for care plan clarification.

---

## API

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

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Treatment_Care_Agent",
    "agent_id": "TREAT-AGENT-1004",
    "treatment": {
      "treatmentPlan": "Immediate revascularisation via primary PCI is indicated...",
      "medications": [
        "Aspirin - 325mg - once daily",
        "Clopidogrel - 600mg loading, then 75mg daily",
        "Heparin - weight-based IV infusion",
        "IV metoprolol - 5mg over 2 min, repeat x3 if haemodynamically stable"
      ],
      "followUpRequired": "YES",
      "followUpTimeframe": "1 week",
      "lifestyleRecommendations": ["Low-sodium, low-fat diet", "Cardiac rehabilitation programme", "Smoking cessation"],
      "monitoringRequired": ["BP every 2 hours", "Troponin at 6 and 12 hours", "Continuous ECG monitoring"],
      "referralRequired": "Cardiac surgery for CABG if PCI not feasible",
      "urgency": "IMMEDIATE"
    }
  }
}
```

**Error response**
```json
{
  "is_success": false,
  "error": {
    "code": "LLM_INVOCATION_ERROR",
    "message": "Agent call failed for patient P001: ..."
  }
}
```

---

## Running Locally

```bash
cd services/treatment-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8012 --reload
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model used by the agent |
| `OPENAI_API_KEY` | — | Required. Set in your environment or a `.env` file |
| `LANGSMITH_TRACING` | `false` | Set `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | Required when `LANGSMITH_TRACING=true` |
| `LANGSMITH_PROJECT` | `agentic-ai-healthcare-system` | Shared project name — see root [README](../../README.md#observability--langsmith-tracing) for distributed tracing details |

---

## Project Structure

```
services/treatment-agent/
├── Dockerfile
├── requirements.txt
├── run.sh
└── src/
    ├── main.py                        # FastAPI app entry point
    ├── agent/
    │   └── treatment_agent.py         # DeepAgent setup, system prompt, JSON schema
    ├── api/
    │   └── server.py                  # POST /treatment-agent/recommend
    ├── core/
    │   └── config.py                  # Env config + agent identity constants
    ├── datamodel/
    │   └── models.py                  # Request / response Pydantic models
    ├── exception/
    │   ├── exceptions.py              # TreatmentSvcException
    │   └── exception_handler.py       # FastAPI exception handlers
    ├── log/
    │   └── logger.py                  # Stdout logger
    └── service/
        └── treatment_service.py       # Plan logic — query builder, agent invocation, response parser
```
