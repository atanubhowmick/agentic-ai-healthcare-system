# Cardiology Agent

A FastAPI microservice that acts as a cardiology specialist. It takes a patient's reported symptoms and metrics, runs them through a DeepAgent (LangGraph-based) backed by an OpenAI LLM, and returns a structured cardiac diagnosis — including severity, required blood tests, lab investigations, and medication guidance.

Runs on **port 8001**. Called by the orchestrator agent as part of the multi-specialist diagnostic pipeline.

---

## How It Works

1. The `/cardiology-agent/diagnose` endpoint receives a patient ID and symptom description.
2. `cardiology_service` builds a query (or uses the raw symptom text for follow-ups) and maintains per-patient chat history in memory.
3. The DeepAgent invokes the LLM with the system prompt and session history, then returns a JSON diagnosis.
4. The response is validated against the `DiagnosisResult` Pydantic model and returned wrapped in a `GenericResponse`.

Follow-up calls (where `is_followup: true`) pass the symptom text directly into the session, so the LLM can answer clarification questions in context.

---

## API

### `POST /cardiology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain radiating to left arm, diaphoresis, BP 90/60, HR 110 bpm",
  "is_followup": false
}
```

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Cardiology_Specialist",
    "agent_id": "CARDIOLOGY-AGENT-1001",
    "diagnosis": {
      "diagnosisDetails": "Presentation is consistent with STEMI. Troponin elevation expected...",
      "severity": "CRITICAL",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "Has the patient had any prior cardiac events or stent placements?",
      "bloodTestsRequired": ["Troponin I/T", "BNP", "CBC", "Lipid panel", "D-dimer"],
      "labTestsRequired": ["12-lead ECG", "Chest X-ray", "Echocardiogram"],
      "medication": "Aspirin 300mg stat, Clopidogrel 600mg loading dose, IV heparin"
    }
  }
}
```

**Error response** (LLM failure or parse error)
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
cd services/cardiology-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8001 --reload
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model used by the agent |
| `OPENAI_API_KEY` | — | Required. Set in your environment or a `.env` file |

---

## Project Structure

```
services/cardiology-agent/
├── Dockerfile
├── requirements.txt
├── run.sh
└── src/
    ├── main.py                        # FastAPI app entry point
    ├── agent/
    │   └── cardiology_agent.py        # DeepAgent setup, system prompt, JSON schema
    ├── api/
    │   └── server.py                  # POST /cardiology-agent/diagnose
    ├── core/
    │   └── config.py                  # Env config + agent identity constants
    ├── datamodel/
    │   └── models.py                  # Request / response Pydantic models
    ├── exception/
    │   ├── exceptions.py              # CardiologySvcException
    │   └── exception_handler.py       # FastAPI exception handlers
    ├── log/
    │   └── logger.py                  # Stdout logger
    └── service/
        └── cardiology_service.py      # Diagnosis logic — query builder, agent invocation, response parser
```
