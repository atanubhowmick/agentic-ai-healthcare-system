# Cancer Agent

A FastAPI microservice that acts as an oncology specialist. Unlike the other specialist agents, it combines a TF-IDF ML pipeline trained on MIMIC-IV clinical data with a DeepAgent (LangGraph-based) LLM to produce structured oncology diagnoses — including TNM staging, tumour markers, biopsy guidance, and referral recommendations.

Runs on **port 8003**. Called by the orchestrator agent as part of the multi-specialist diagnostic pipeline.

---

## How It Works

```
Patient symptoms
      │
      ├─▶ TF-IDF Predictor (trained on MIMIC-IV at startup)
      │       Predicts: severity, emergencyCareNeeded,
      │                 hospitalizationNeeded, suspectedCancerType
      │
      └─▶ DeepAgent (LLM + MIMIC RAG tool)
              Calls: search_mimic_cases → retrieves similar historical cases from ChromaDB
              Produces: diagnosisDetails, stage, biomarkersRequired,
                        imagingRequired, biopsyRequired, oncologyReferralNeeded, medication
              │
              ▼
        Structured response (LLM fields + TF-IDF fields merged)
```

### TF-IDF Pipeline

Four independent classifiers are trained from MongoDB MIMIC-IV records at service startup:

| Classifier | Algorithm | Input features |
|---|---|---|
| Severity | LogisticRegression | TF-IDF on discharge notes + ICD codes |
| Emergency | HistGradientBoosting + TruncatedSVD | TF-IDF on notes + ICD codes + chief complaint |
| Hospitalisation | HistGradientBoosting + TruncatedSVD | Same as emergency |
| Cancer type | CalibratedClassifierCV(LinearSVC) | TF-IDF on notes + ICD codes |

Training is skipped if no records are found in MongoDB; safe defaults are used instead.

### MIMIC-IV RAG

`mimic_retriever.py` retrieves the top-k most semantically similar historical cancer cases from a ChromaDB collection (populated from MIMIC-IV discharge notes). These cases are injected into the LLM prompt as clinical reference material. Three similarity tiers are applied:

| Similarity score | Behaviour |
|---|---|
| `>= MIMIC_SIMILARITY_THRESHOLD` (0.75) | High-confidence context injected |
| `>= MIMIC_PARTIAL_THRESHOLD` (0.60) | Low-confidence context, flagged in prompt |
| Below both thresholds | LLM-only, no MIMIC context |

---

## API

### `POST /cancer-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Unexplained weight loss 12kg over 3 months, hard neck lump, night sweats, blood in stool",
  "is_followup": false
}
```

**Success response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Cancer_Oncology_Specialist",
    "agent_id": "CANCER-AGENT-1004",
    "diagnosis": {
      "diagnosisDetails": "Presentation raises strong suspicion of lymphoma with possible colorectal involvement...",
      "suspectedCancerType": "Lymphoma",
      "stage": "Undetermined — staging requires imaging and biopsy",
      "severity": "HIGH",
      "severityConfidence": 82,
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "NO",
      "emergencyCareConfidence": 30,
      "clarificationQuestion": "Any family history of haematological malignancies?",
      "biomarkersRequired": ["LDH", "Beta-2 microglobulin", "CEA", "CA 19-9"],
      "imagingRequired": ["PET-CT", "CT chest/abdomen/pelvis", "MRI abdomen"],
      "biopsyRequired": "Excisional biopsy of cervical lymph node",
      "oncologyReferralNeeded": "Haematology-Oncology + Gastroenterology",
      "medication": "Symptom management: antiemetics, nutritional support"
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
cd services/cancer-agent

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8003 --reload
```

### Prerequisites

- MongoDB running with MIMIC-IV records loaded (run `scripts/load_mimic_mongo.py`)
- ChromaDB running on port 8020 with the `mimic_cancer_cases` collection populated (run `scripts/load_mimic_data.py`)
- `OPENAI_API_KEY` set in environment or `.env`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model for the DeepAgent |
| `OPENAI_API_KEY` | — | Required |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection string |
| `MONGO_DB` | `agentic_ai_healthcare_db` | Database name |
| `MONGO_MIMIC_COLLECTION` | `mimic_iv_records` | Collection holding MIMIC-IV records |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB host |
| `CHROMA_PORT` | `8020` | ChromaDB port |
| `MIMIC_SIMILARITY_THRESHOLD` | `0.75` | Min cosine similarity for high-confidence RAG context |
| `MIMIC_PARTIAL_THRESHOLD` | `0.60` | Min cosine similarity for low-confidence RAG context |
| `MIMIC_TOP_K` | `3` | Number of MIMIC cases to retrieve per query |

---

## Project Structure

```
services/cancer-agent/
├── Dockerfile
├── requirements.txt
├── run.sh
├── scripts/
│   ├── load_mimic_data.py             # Load MIMIC-IV oncology cases into ChromaDB
│   ├── load_mimic_mongo.py            # Load MIMIC-IV records into MongoDB
│   └── train_models.py                # Train TF-IDF models and save to pickle for XAI SHAP
└── src/
    ├── main.py                        # FastAPI app entry point
    ├── agent/
    │   └── cancer_agent.py            # DeepAgent setup, system prompt, MIMIC search tool
    ├── api/
    │   └── server.py                  # POST /cancer-agent/diagnose
    ├── core/
    │   └── config.py                  # Env config + agent identity + MIMIC thresholds
    ├── datamodel/
    │   └── models.py                  # Request / response Pydantic models
    ├── exception/
    │   ├── exceptions.py              # CancerSvcException
    │   └── exception_handler.py       # FastAPI exception handlers
    ├── log/
    │   └── logger.py                  # Stdout logger
    ├── rag/
    │   ├── mimic_retriever.py         # ChromaDB similarity search for MIMIC-IV cases
    │   └── tfidf_predictor.py         # TF-IDF + ML classifiers trained from MongoDB
    └── service/
        └── cancer_service.py          # Diagnosis logic — query builder, agent invocation, response parser
```
