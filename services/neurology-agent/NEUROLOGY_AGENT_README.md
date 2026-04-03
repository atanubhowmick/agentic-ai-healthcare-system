# Neurology Agent

Neurology specialist agent — port **8002**. Analyses neurological symptoms and recommends appropriate imaging, neurological tests, and treatment.

---

## How It Works

Uses `deepagents.create_deep_agent` (LangGraph-based) with a neurology specialist system prompt. Evaluates symptoms for severity, emergency status, required imaging (CT, MRI), neurological tests (lumbar puncture, EEG), and medication guidance.

---

## API Endpoint

### `POST /neurology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Sudden onset severe headache, neck stiffness, photophobia, nausea",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Neurology_Specialist",
    "agent_id": "NEURO-AGENT-1002",
    "diagnosis": {
      "diagnosisDetails": "...",
      "severity": "CRITICAL",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "...",
      "bloodTestsRequired": ["CBC", "CRP", "Blood culture"],
      "imagingRequired": ["CT head without contrast", "MRI brain"],
      "neurologicalTestsRequired": ["Lumbar puncture"],
      "medication": "..."
    }
  }
}
```

---

## Running Locally

```bash
cd services/neurology-agent

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8002 --reload
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
├── exception/  # NeurologySvcException + handler
├── service/    # neurology_service.py
├── log/
└── main.py
```
