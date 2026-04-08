# Pathology Agent

A FastAPI microservice that acts as a pathology specialist. It receives lab results and biomarker data for a patient, runs them through a DeepAgent (LangGraph-based) backed by an OpenAI LLM, and returns a structured analysis — including severity, additional tests, imaging, and specialist referral recommendations.

Runs on **port 8011**. Called by the orchestrator agent as part of the multi-specialist diagnostic pipeline.

---

## How It Works

1. The `/pathology-agent/diagnose` endpoint receives a patient ID and lab result details.
2. `pathology_service` builds a query (or uses the raw text for follow-ups) and maintains per-patient chat history in memory.
3. The DeepAgent invokes the LLM with the system prompt and session history, then returns a JSON analysis.
4. The response is validated against the `DiagnosisResult` Pydantic model and returned wrapped in a `GenericResponse`.

Follow-up calls (where `is_followup: true`) pass the text directly into the session so the LLM can answer clarification questions in context.

---

## API

### `POST /pathology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "HbA1c: 9.2%, Fasting glucose: 280 mg/dL, Creatinine: 1.8 mg/dL",
  "is_followup": false
}
```

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Pathology_Specialist",
    "agent_id": "PATHO-AGENT-1002",
    "diagnosis": {
      "analysisDetails": "HbA1c of 9.2% indicates poorly controlled type 2 diabetes...",
      "severity": "HIGH",
      "hospitalizationNeeded": "NO",
      "emergencyCareNeeded": "NO",
      "clarificationQuestion": "Has the patient had any recent changes to diabetes medication?",
      "additionalTestsRequired": ["Urine microalbumin", "Fasting lipid panel", "eGFR"],
      "imagingRequired": ["Renal ultrasound"],
      "referralNeeded": "Endocrinology and Nephrology referral recommended"
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
cd services/pathology-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8011 --reload
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model used by the agent |
| `OPENAI_API_KEY` | — | Required. Set in your environment or a `.env` file |

---

## Project Structure

```
services/pathology-agent/
├── Dockerfile
├── requirements.txt
├── run.sh
└── src/
    ├── main.py                        # FastAPI app entry point
    ├── agent/
    │   └── pathology_agent.py         # DeepAgent setup, system prompt, JSON schema
    ├── api/
    │   └── server.py                  # POST /pathology-agent/diagnose
    ├── core/
    │   └── config.py                  # Env config + agent identity constants
    ├── datamodel/
    │   └── models.py                  # Request / response Pydantic models
    ├── exception/
    │   ├── exceptions.py              # PathologySvcException
    │   └── exception_handler.py       # FastAPI exception handlers
    ├── log/
    │   └── logger.py                  # Stdout logger
    └── service/
        └── pathology_service.py       # Diagnosis logic — query builder, agent invocation, response parser
```
