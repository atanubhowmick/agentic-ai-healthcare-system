# Neurology Agent

A FastAPI microservice that acts as a neurology specialist. It takes a patient's reported symptoms, runs them through a DeepAgent (LangGraph-based) backed by an OpenAI LLM, and returns a structured diagnosis — including severity, required tests, imaging, and medication guidance.

Runs on **port 8002**. Called by the orchestrator agent as part of the multi-specialist diagnostic pipeline.

---

## How It Works

1. The `/neurology-agent/diagnose` endpoint receives a patient ID and symptom description.
2. `neurology_service` builds a query (or uses the raw symptom text for follow-ups) and maintains per-patient chat history in memory.
3. The DeepAgent invokes the LLM with the system prompt and session history, then returns a JSON diagnosis.
4. The response is validated against the `NeurologyResult` Pydantic model and returned wrapped in a `GenericResponse`.

Follow-up calls (where `is_followup: true`) pass the symptom text directly into the session, so the LLM can answer clarification questions in context.

---

## API

### `POST /neurology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Sudden onset severe headache, neck stiffness, photophobia, nausea",
  "is_followup": false
}
```

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Neurology_Specialist",
    "agent_id": "NEURO-AGENT-1002",
    "diagnosis": {
      "diagnosisDetails": "Presentation is consistent with bacterial meningitis...",
      "severity": "CRITICAL",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "Has the patient had any recent infections or vaccinations?",
      "bloodTestsRequired": ["CBC", "CRP", "Blood culture", "Procalcitonin"],
      "imagingRequired": ["CT head without contrast", "MRI brain with contrast"],
      "neurologicalTestsRequired": ["Lumbar puncture"],
      "medication": "Empirical IV ceftriaxone 2g q12h + dexamethasone 0.15mg/kg q6h"
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
cd services/neurology-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8002 --reload
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
services/neurology-agent/
├── Dockerfile
├── requirements.txt
├── run.sh
└── src/
    ├── main.py                        # FastAPI app entry point
    ├── agent/
    │   └── neurology_agent.py         # DeepAgent setup, system prompt, JSON schema
    ├── api/
    │   └── server.py                  # POST /neurology-agent/diagnose
    ├── core/
    │   └── config.py                  # Env config + agent identity constants
    ├── datamodel/
    │   └── models.py                  # Request / response Pydantic models
    ├── exception/
    │   ├── exceptions.py              # NeurologySvcException
    │   └── exception_handler.py       # FastAPI exception handlers
    ├── log/
    │   └── logger.py                  # Stdout logger
    └── service/
        └── neurology_service.py       # Diagnosis logic — query builder, agent invocation, response parser
```
