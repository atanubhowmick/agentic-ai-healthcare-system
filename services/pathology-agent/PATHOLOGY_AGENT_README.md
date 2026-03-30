# Pathology Agent

Pathology specialist agent — port **8011**. Analyses laboratory results and biomarker abnormalities to identify pathological conditions.

---

## How It Works

Uses `deepagents.create_deep_agent` (LangGraph-based) with a pathology specialist system prompt. Interprets lab values (HbA1c, creatinine, lipid panels, tumour markers, etc.) and returns structured findings with severity, required follow-up tests, and recommendations.

---

## API Endpoint

### `POST /pathology-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "HbA1c: 9.2%, Fasting glucose: 280 mg/dL, Creatinine: 1.8 mg/dL",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Pathology_Specialist",
    "agent_id": "PATHOLOGY-AGENT-1003",
    "diagnosis": {
      "analysisDetails": "...",
      "severity": "HIGH",
      "hospitalizationNeeded": "NO",
      "emergencyCareNeeded": "NO",
      "clarificationQuestion": "...",
      "bloodTestsRequired": ["Lipid panel", "Urine microalbumin"],
      "labTestsRequired": ["Renal function panel"],
      "medication": "..."
    }
  }
}
```

---

## Running Locally

```bash
cd services/pathology-agent

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8011 --reload
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
├── constant/   # constants.py (PATHOLOGY_AGENT_ID)
├── core/       # config.py
├── datamodel/  # Pydantic models
├── exception/  # PathologySvcException + handler
├── service/    # pathology_service.py
├── log/
└── main.py
```
