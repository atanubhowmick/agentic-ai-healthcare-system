# XAI Validation Service

Clinical safety validator — port **8016**. Validates specialist diagnoses and treatment recommendations before they are returned to the patient. Combines deterministic rule checks, LLM-based reasoning, SHAP explainability, and constitutional safety principles grounded in real clinical guidelines from PubMed.

---

## How It Works

```
Diagnosis/Treatment request
        │
        ▼
Rule pre-filter (medical_rules.py)      ← deterministic, no LLM
        │ fail → REJECT immediately
        │ pass ↓
        ▼
SHAP explainability (shap_provider.py)  ← pre-computed before LLM
        │
        ▼
DeepAgent reasoning (xai_agent.py)      ← single LLM call with factors injected
        │
        ▼
Constitutional guard (constitutional_guard.py)
  ├── Pre-filter: skip if no red flags
  ├── Guideline RAG: retrieve PubMed abstracts from ChromaDB
  ├── Critique: LLM checks P1–P5 principles
  └── Revision: LLM rewrites response if violations found
        │
        ▼
ValidationResult (with metadata + explanation_factors)
```

### Rule Pre-filter

`medical_rules.py` — fast deterministic checks before any LLM call:
- `check_emergency_consistency`: CRITICAL symptoms must have emergency=YES
- `check_severity_validity`: severity must be LOW / HIGH / CRITICAL

If a rule fails, the request is rejected immediately with zero LLM cost.

### SHAP Explainability

`shap_provider.py` loads trained cancer agent models from `trained_model/cancer_agent_models.pkl` at startup. Computes feature attribution for three classifiers:

| Task | Algorithm | Attribution method |
|------|-----------|-------------------|
| Severity | LogisticRegression | Linear: `feature_value × coef` |
| Emergency | HistGBM + TruncatedSVD | TreeSHAP → SVD back-projection |
| Cancer type | CalibratedClassifierCV(LinearSVC) | Linear coef from first fold |

Results are stored in a `ContextVar` (request-scoped, thread-safe) and attached to the final `ValidationResult` as `explanation_factors`. Falls back to an LLM-based explanation if the pkl is not found.

### Constitutional Guard

Custom critique→revision loop (not LangChain's deprecated `ConstitutionalChain`):

| Principle | Description |
|-----------|-------------|
| P1 — Emergency conservatism | CRITICAL severity must have emergency=YES to APPROVE |
| P2 — Epistemic humility | confidence ≥ 0.9 only when evidence is clear and unambiguous |
| P3 — Safety-first triage | emergency symptoms + LOW severity requires explicit justification |
| P4 — Completeness | APPROVE needs at least one specific clinical observation |
| P5 — Guideline alignment | response must not contradict retrieved PubMed clinical guidelines |

A pre-filter skips the LLM critique call for routine LOW-severity cases with no red flags, reducing cost and improving stability.

### Clinical Guidelines RAG

`guidelines/guideline_client.py` — ChromaDB `clinical_guidelines` collection seeded from PubMed at first startup:
- **24 API calls** to NCBI E-utilities (esearch + efetch) at 3 req/s
- 12 query topics: oncology, cardiology, neurology, sepsis, respiratory, endocrinology
- Sources: NCCN, AHA, WHO, SSC, GOLD, ADA
- Subsequent restarts skip the seed (already in ChromaDB)

---

## API Endpoints

### `POST /xai-validator/validate-diagnosis`

Validate a specialist diagnosis.

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain, shortness of breath",
  "specialist_agent": "Cardiology_Specialist",
  "diagnosis": {
    "severity": "HIGH",
    "emergencyCareNeeded": "YES",
    "diagnosysDetails": "Suspected STEMI"
  }
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "XAI_Validator",
    "agent_id": "XAI-VALIDATOR-1003",
    "patient_id": "P001",
    "validation_type": "DIAGNOSIS",
    "result": {
      "is_validated": true,
      "confidence_score": 0.88,
      "validation_summary": "Diagnosis is consistent with presented symptoms.",
      "key_concerns": [],
      "recommendation": "APPROVE",
      "explanation_factors": [
        {"factor": "chest pain", "importance": 0.87, "direction": "increases_risk"},
        {"factor": "persistent", "importance": 0.16, "direction": "increases_risk"}
      ],
      "validator_latency_ms": 3420.5,
      "model_used": "gpt-5.2",
      "explainability_method": "SHAP",
      "rules_triggered": [],
      "constitutional_revised": false
    }
  }
}
```

### `POST /xai-validator/validate-treatment`

Validate a treatment recommendation.

**Request**
```json
{
  "patient_id": "P001",
  "specialist_agent": "Cardiology_Specialist",
  "diagnosis_summary": "Suspected STEMI — emergency care required",
  "severity": "HIGH",
  "treatment_recommendation": "PCI within 90 minutes, dual antiplatelet therapy"
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "XAI_Validator",
    "agent_id": "XAI-VALIDATOR-1003",
    "patient_id": "P001",
    "validation_type": "TREATMENT",
    "result": {
      "is_validated": true,
      "confidence_score": 0.91,
      "validation_summary": "Treatment plan aligns with AHA STEMI guidelines.",
      "key_concerns": [],
      "recommendation": "APPROVE",
      "explanation_factors": [],
      "validator_latency_ms": 2180.0,
      "model_used": "gpt-5.2",
      "explainability_method": "",
      "rules_triggered": [],
      "constitutional_revised": false
    }
  }
}
```

**recommendation values:** `APPROVE` | `REJECT` | `REVIEW`

---

## Running Locally

```bash
cd xai-validation-service

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Start the service (port 8016)
bash run.sh

# Or with Uvicorn directly
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8016 --reload
```

### Prerequisites

- `OPENAI_API_KEY` set in environment or `.env`
- ChromaDB running on port 8020 (for clinical guidelines RAG)
- `trained_model/cancer_agent_models.pkl` at project root (for SHAP — generate via `services/cancer-agent/scripts/export_models.py`)

On first startup, if ChromaDB is reachable, the service will fetch ~22 PubMed abstracts and seed the `clinical_guidelines` collection (~10 seconds). Subsequent starts skip this.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model for validation and constitutional guard |
| `XAI_AGENT_ID` | `XAI-VALIDATOR-1003` | Agent identity in responses |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB host |
| `CHROMA_PORT` | `8020` | ChromaDB port |

---

## Source Structure

```
src/
├── agent/          # DeepAgent executor + @tool definitions (xai_agent.py)
├── api/            # FastAPI router
├── core/           # config.py
├── datamodel/      # Pydantic models (ValidationResult, ValidationResponse)
├── exception/      # ValidationSvcException + handler
├── explainers/
│   ├── shap_provider.py    # SHAP attribution (LR, HistGBM, LinearSVC)
│   └── context.py          # ContextVar for request-scoped SHAP results
├── guardrails/
│   └── constitutional_guard.py  # P1–P5 critique → revision loop
├── guidelines/
│   ├── guideline_client.py      # ChromaDB client for clinical guidelines
│   └── pubmed_retriever.py      # NCBI E-utilities fetcher (3 req/s)
├── service/        # Validation orchestration (validator_service.py)
├── validators/     # Deterministic rule checks (medical_rules.py)
├── log/
└── main.py
```
