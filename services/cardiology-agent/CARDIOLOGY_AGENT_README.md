# Cardiology Agent

Cardiac specialist agent — port **8001**. Analyses cardiac symptoms, flags anomalies, and recommends appropriate cardiac workup including blood tests, ECG, and imaging.

---

## How It Works

Uses `deepagents.create_deep_agent` (LangGraph-based) with a cardiac specialist system prompt. The agent evaluates symptoms for severity, emergency status, hospitalisation need, required investigations, and medication guidance, returning a structured diagnosis.

---

## API Endpoint

### `POST /cardiology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain radiating to left arm, shortness of breath, diaphoresis",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Cardiology_Specialist",
    "agent_id": "CARDIOLOGY-AGENT-1001",
    "diagnosis": {
      "diagnosysDetails": "...",
      "severity": "HIGH",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "...",
      "bloodTestsRequired": ["Troponin I", "BNP"],
      "labTestsRequired": ["ECG", "Echocardiogram"],
      "medication": "..."
    }
  }
}
```

> **Note:** `diagnosysDetails` is a known typo in the original model — preserved for backward compatibility.

---

## Running Locally

```bash
cd services/cardiology-agent

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8001 --reload
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
├── exception/  # CardiologySvcException + handler
├── service/    # cardiology_service.py
├── log/
└── main.py
```
