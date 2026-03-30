# Cancer Agent

Oncology specialist agent — port **8003**. Performs TNM staging, tumour marker assessment, and biopsy/imaging guidance for cancer presentations. Unique among specialist agents in using a TF-IDF ML pipeline trained on MIMIC-IV clinical data alongside the LLM.

---

## How It Works

```
Patient symptoms
      │
      ▼
TF-IDF Predictor (ML pipeline)
  ├── Severity classifier  (LogisticRegression on discharge notes)
  ├── Emergency classifier (HistGBM + TruncatedSVD)
  ├── Hospitalisation classifier (HistGBM + TruncatedSVD)
  └── Cancer type classifier (CalibratedClassifierCV → LinearSVC)
      │
      ▼
MIMIC-IV RAG (MongoDB → structured case retrieval)
      │
      ▼
DeepAgent (LLM) — @tool: search_mimic_cases
      │
      ▼
Structured oncology diagnosis response
```

### TF-IDF Pipeline

The `TfidfPredictor` (`src/rag/tfidf_predictor.py`) trains four independent classifiers from MongoDB MIMIC-IV records at startup:

| Classifier | Algorithm | Features |
|------------|-----------|----------|
| Severity | LogisticRegression | TF-IDF on discharge notes + ICD codes |
| Emergency | HistGradientBoosting + TruncatedSVD | TF-IDF on notes + ICD codes + chief complaint |
| Hospitalisation | HistGradientBoosting + TruncatedSVD | Same as emergency |
| Cancer type | CalibratedClassifierCV(LinearSVC) | TF-IDF on notes + ICD codes |

Training is skipped if fewer than 50 records exist in MongoDB. The models live in memory and are re-trained on each cold start.

### MIMIC-IV RAG

`MimicRetriever` (`src/rag/mimic_retriever.py`) retrieves the top-k most similar historical cancer cases from MongoDB using cosine similarity on TF-IDF vectors. These cases are injected into the LLM prompt as clinical context.

### DeepAgent

Uses `deepagents.create_deep_agent` (LangGraph-based) with one tool:
- `search_mimic_cases(symptoms)` — retrieves similar MIMIC-IV cases and their outcomes

---

## API Endpoint

### `POST /cancer-agent/diagnose`

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Unexplained weight loss 12kg over 3 months, hard neck lump, night sweats, blood in stool",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Cancer_Oncology_Specialist",
    "agent_id": "CANCER-AGENT-1003",
    "diagnosis": {
      "diagnosisDetails": "...",
      "severity": "LOW",
      "hospitalizationNeeded": "NO",
      "emergencyCareNeeded": "NO",
      "cancerType": "Lymphoma (suspected)",
      "tnmStaging": "...",
      "tumourMarkers": ["LDH", "Beta-2 microglobulin"],
      "biopsyRequired": "YES",
      "imagingRequired": ["PET-CT", "CT chest/abdomen/pelvis"],
      "clarificationQuestion": "..."
    }
  }
}
```

**Error envelope**
```json
{
  "is_success": false,
  "error": {
    "code": "LLM_INVOCATION_ERROR",
    "message": "..."
  }
}
```

---

## Model Export (for XAI SHAP)

The trained TF-IDF models can be exported for SHAP explainability in the XAI validation service:

```bash
cd services/cancer-agent
python scripts/export_models.py
```

Output: `trained_model/cancer_agent_models.pkl` at the project root. The XAI service loads this file for SHAP attribution.

---

## Running Locally

```bash
cd services/cancer-agent

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Start the service (port 8003)
bash run.sh

# Or with Uvicorn directly
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8003 --reload
```

### Prerequisites

- MongoDB running with MIMIC-IV records loaded (run `scripts/extract_evaluation_dataset.py` first)
- `OPENAI_API_KEY` set in environment or `.env`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_DEFAULT_MODEL` | `gpt-5.2` | LLM model for the DeepAgent |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `healthcare_db` | Database name |

---

## Source Structure

```
src/
├── agent/          # DeepAgent executor + @tool definitions
├── api/            # FastAPI router
├── core/           # config.py
├── datamodel/      # Pydantic request/response models
├── exception/      # CancerSvcException + handler
├── rag/            # TfidfPredictor + MimicRetriever
├── service/        # Business logic (cancer_service.py)
├── log/
└── main.py
scripts/
├── export_models.py           # Export trained models for SHAP
└── extract_evaluation_dataset.py  # Load MIMIC-IV into MongoDB
```
