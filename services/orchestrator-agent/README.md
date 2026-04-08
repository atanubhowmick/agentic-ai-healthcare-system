# Orchestrator Agent

LangGraph master agent — port **8015**. Accepts a patient case and runs the full diagnosis and treatment pipeline: semantic cache check, 4-tier triage classification, specialist routing, optional secondary pathology cross-check, XAI validation with retry loops, and ChromaDB/MongoDB persistence.

---

## Pipeline

```
POST /orchestrator/diagnose
        │
        ▼
  chroma_lookup ──── cache hit (similarity ≥ 0.85) ────────────────────────┐
        │ miss                                                               │
        ▼                                                                    │
  classifier (4-tier hybrid)                                                 │
        │                                                                    │
        ▼                                                                    │
  specialist_node (cardiology / neurology / cancer / pathology)             │
        │                                                                    │
        ├── secondary_check_needed → pathology cross-check → conflict_check │
        │                                                                    │
        ▼                                                                    │
  xai_diagnosis_validator ◄──── retry loop (max 3×) ────────────────────┐  │
        │ validated                                                       │  │
        ▼                                                                 │  │
  treatment_node                                                          │  │
        │                                                                 │  │
        ▼                                                                 │  │
  xai_treatment_validator ◄──── retry loop (max 3×) ─────────────────┐  │  │
        │ validated                                                    │  │  │
        ▼                                                              │  │  │
      finish ◄──────────────────────────────────────────────────────────┘  │
        │                                                                   │
        └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  MongoDB save + ChromaDB cache update (non-blocking)
```

### 4-Tier Classifier

Each tier escalates only when confidence is below its threshold:

| Tier | Model | Trigger |
|------|-------|---------|
| 1 | Keyword rule router | dominance ratio ≥ 0.80, min 4 keyword hits |
| 2 | BioBERT zero-shot NLI | entailment score ≥ 0.80 |
| 3 | Fine-tuned ClinicalBERT | softmax probability ≥ 0.80 (skipped if model absent) |
| 4 | GPT-5.2 LLM fallback | always returns |

### Retry and Human Review

Both XAI validation nodes retry up to 3 times. After 3 failures the case is returned with `status: HUMAN_REVIEW_REQUIRED`.

### Conflict Detection

When a secondary pathology cross-check is triggered, an LLM compares primary and secondary severity assessments. A conflict routes directly to finish with `HUMAN_REVIEW_REQUIRED`.

---

## API

### `POST /orchestrator/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain radiating to left arm, diaphoresis, SpO2 86%"
}
```

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "patient_id": "P001",
    "agent_id": "ORCH-AGENT-1000",
    "status": "COMPLETED",
    "specialist_agent": "Cardiology_Specialist",
    "diagnosis": {
      "summary": "Presentation consistent with STEMI...",
      "severity": "CRITICAL",
      "emergency_care_needed": "YES",
      "hospitalization_needed": "YES",
      "full_details": {}
    },
    "xai_diagnosis_validation": { "..." },
    "treatment": { "..." },
    "xai_treatment_validation": { "..." },
    "conflict_detected": false,
    "conflict_reason": "",
    "human_review_reason": null,
    "audit_trail": [
      "[CHROMA_LOOKUP] No cache hit - proceeding with full diagnosis flow",
      "[CLASSIFIER] Routing to 'cardiology'. Secondary check: false. Reason: [Rule] conf=0.90",
      "[CARDIOLOGY_SPECIALIST] Severity: CRITICAL | Emergency: YES",
      "[XAI_DIAGNOSIS] Validated on attempt 1 - diagnosis saved to ChromaDB",
      "[TREATMENT] Care plan generated for patient P001 | urgency: IMMEDIATE",
      "[XAI_TREATMENT] Validated on attempt 1 - treatment saved to ChromaDB"
    ]
  }
}
```

Cache-hit response uses `status: COMPLETED_FROM_CACHE` and omits XAI validation fields.

---

## Running Locally

```bash
cd services/orchestrator-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8015 --reload
```

### Prerequisites

All downstream services must be running before the orchestrator starts:
- Cardiology (8001), Neurology (8002), Cancer (8003), Pathology (8011)
- Treatment agent (8012)
- XAI validation service (8016)
- ChromaDB (8020)
- MongoDB (optional — persistence is non-blocking)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM for the classifier and conflict check |
| `CARDIOLOGY_SERVICE_URL` | `http://127.0.0.1:8001/cardiology-agent` | |
| `NEUROLOGY_SERVICE_URL` | `http://127.0.0.1:8002/neurology-agent` | |
| `CANCER_SERVICE_URL` | `http://127.0.0.1:8003/cancer-agent` | |
| `PATHOLOGY_SERVICE_URL` | `http://127.0.0.1:8011/pathology-agent` | |
| `TREATMENT_SERVICE_URL` | `http://127.0.0.1:8012/treatment-agent` | |
| `XAI_SERVICE_URL` | `http://127.0.0.1:8016/xai-validator` | |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB host |
| `CHROMA_PORT` | `8020` | ChromaDB port |
| `CHROMA_SIMILARITY_THRESHOLD` | `0.85` | Minimum score for cache hit |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | |
| `MONGO_DB` | `agentic_ai_healthcare_db` | |
| `MAX_RETRY_COUNT` | `3` | Max XAI validation retries per loop |
| `HTTP_TIMEOUT` | `60.0` | Per-request timeout for downstream calls (seconds) |
| `CLINICALBERT_MODEL_DIR` | `./clinicalbert_router` | Fine-tuned ClinicalBERT model path (Tier 3) |
| `RULE_DOMINANCE_RATIO` | `0.80` | Tier 1: min keyword share to route |
| `RULE_MIN_KEYWORD_HITS` | `4` | Tier 1: min total keyword hits |
| `BIOBERT_CONFIDENCE_THRESHOLD` | `0.80` | Tier 2: min NLI entailment score |
| `CLINICAL_CONFIDENCE_THRESHOLD` | `0.80` | Tier 3: min softmax probability |

---

## Source Structure

```
src/
├── agents/
│   ├── graph.py              # LangGraph state machine — nodes + edges
│   ├── nodes.py              # All node implementations
│   ├── state.py              # AgentState TypedDict
│   └── classifier_router.py  # 4-tier hybrid classifier
├── api/
│   └── server.py             # POST /orchestrator/diagnose
├── core/
│   ├── config.py             # All env vars + classifier label map
│   ├── chroma_client.py      # ChromaDB semantic cache (diagnosis + treatment collections)
│   ├── mongo_client.py       # Motor async MongoDB persistence
│   └── exceptions.py         # Orchestrator exception hierarchy
├── exception/
│   └── exception_handler.py  # FastAPI exception handlers
├── schemas/
│   ├── request.py            # OrchestratorRequest
│   └── response.py           # OrchestratorResponse + GenericResponse
├── tools/                    # @tool HTTP clients for each downstream service
│   ├── cardiology_client.py
│   ├── neurology_client.py
│   ├── cancer_client.py
│   ├── pathology_client.py
│   ├── treatment_client.py
│   └── xai_client.py
├── log/
│   └── logger.py
└── main.py                   # FastAPI app + warm_up_models lifespan
```
