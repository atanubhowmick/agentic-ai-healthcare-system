# Orchestrator Agent

LangGraph master agent — port **8015**. Coordinates the full diagnosis and treatment pipeline: semantic cache lookup, triage classification, specialist routing, XAI validation, conflict detection, and ChromaDB persistence.

---

## How It Works

### LangGraph Flow

```
chroma_lookup
  ├── cache hit  → finish (return cached result)
  └── cache miss → triage_classifier
                      │
                      ▼
               specialist_node (cardiology / neurology / cancer / pathology)
                      │
                      ├── secondary_check (if triggered)
                      │      └── conflict_check
                      ▼
               xai_diagnosis_validator ──► retry (max 3×) ──► specialist_node
                      │ validated
                      ▼
               treatment_node
                      │
                      ▼
               xai_treatment_validator ──► retry (max 3×) ──► treatment_node
                      │ validated
                      ▼
                    finish
```

### Key Components

**Semantic Cache** (`core/chroma_client.py`): ChromaDB `treatment_outcomes` is searched first. A similarity score ≥ 0.85 returns the cached result immediately, bypassing the full pipeline.

**Triage Classifier** (`agents/classifier_router.py`): LLM-based classifier routes the patient to the appropriate specialist. Also determines if a secondary check (e.g. pathology alongside cardiology) is needed.

**Retry Loops**: Both XAI validation nodes retry up to 3 times. If validation fails after 3 attempts, the case is flagged as `HUMAN_REVIEW_REQUIRED`.

**ChromaDB Persistence**: Validated diagnoses are saved to `diagnosis_outcomes` and validated treatments to `treatment_outcomes` (fire-and-forget, non-blocking).

---

## API Endpoint

### `POST /orchestrator/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain, shortness of breath, SpO2 86%",
  "patient_name": "John Doe"
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "patient_id": "P001",
    "status": "COMPLETED",
    "specialist_agent": "Cardiology_Specialist",
    "diagnosis": { "..." },
    "treatment": { "..." },
    "xai_diagnosis_validation": { "..." },
    "xai_treatment_validation": { "..." },
    "conflict_detected": false,
    "conflict_reason": "",
    "human_review_reason": "",
    "audit_trail": [
      "[CHROMA_LOOKUP] No cache hit - proceeding with full diagnosis flow",
      "[CLASSIFIER] Routing to 'cardiology'...",
      "[CARDIOLOGY_SPECIALIST] Severity: CRITICAL | Emergency: YES",
      "[XAI_DIAGNOSIS] Validated on attempt 1 - diagnosis saved to ChromaDB",
      "[TREATMENT] Treatment plan generated",
      "[XAI_TREATMENT] Validated on attempt 1 - treatment saved to ChromaDB"
    ]
  }
}
```

---

## Running Locally

```bash
cd services/orchestrator-agent

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8015 --reload
```

### Prerequisites

All downstream services must be running: specialist agents (8001-8003, 8011), treatment agent (8012), XAI validation service (8016), ChromaDB (8020).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM for classifier and agent nodes |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB host |
| `CHROMA_PORT` | `8020` | ChromaDB port |
| `CHROMA_PERSIST_DIR` | `./chroma_store` | Local persist dir (if not using HTTP server) |
| `XAI_SERVICE_URL` | `http://localhost:8016` | XAI validation service base URL |

---

## Source Structure

```
src/
├── agents/
│   ├── graph.py             # LangGraph state machine definition
│   ├── nodes.py             # All graph node implementations
│   ├── state.py             # AgentState TypedDict
│   └── classifier_router.py # Triage LLM classifier
├── api/                     # FastAPI router
├── core/
│   ├── config.py
│   └── chroma_client.py     # ChromaDB semantic cache client
├── datamodel/               # Pydantic models
├── exception/
├── log/
└── main.py
```
