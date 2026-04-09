# Evaluation Service

Metrics and evaluation service — port **8017**. Loads MIMIC-IV cases from MongoDB and runs them through the XAI validation service and a TF-IDF baseline to measure clinical safety, explainability quality, and classifier performance.

---

## Evaluations

### TF-IDF Baseline (`POST /evaluate/tfidf-baseline`)

Trains TF-IDF + classifier models on MIMIC-IV cases and reports metrics on a held-out test split. Provides a classical ML reference for comparison against the LLM-based cancer agent. Note that TF-IDF is supervised (trained on MIMIC data) while the LLM agent is zero-shot — that asymmetry is intentional.

| Task | Classifier | Features |
|------|-----------|----------|
| Emergency care needed (binary) | HistGradientBoosting | TF-IDF(text) + TF-IDF(ICD) + TF-IDF(chief complaint) + has_icu_stay → SVD(300) |
| Hospitalization needed (binary) | HistGradientBoosting | Same as above |
| Severity (LOW / HIGH / CRITICAL) | Logistic Regression | TF-IDF(text) + TF-IDF(ICD) + has_icu_stay |
| Cancer type (multi-class) | LinearSVC | TF-IDF(text) + TF-IDF(ICD) |

### XAI Evaluation (`POST /evaluate/xai`)

Sends MIMIC-IV cases to the XAI validation service and measures:

| Metric | What it measures |
|--------|-----------------|
| Decision accuracy | Correctly diagnosed cases approved without false rejection |
| Safety net effectiveness | Under-triage cases (HIGH/CRITICAL labelled LOW) correctly caught |
| Rule engine coverage | Fraction handled by deterministic rules vs LLM path |
| Over-rejection rate | False positive rejection rate for correct diagnoses |
| Stability | Same payload sent 3×: fraction with consistent recommendation |
| Fidelity | Severity perturbed — does decision and explanation change accordingly? |
| Consistency | Paraphrased symptoms — fraction with same recommendation |
| Sparsity | Avg key concerns per response + avg summary word count |
| Interpretability | Flesch Reading Ease of validation summaries |

Both evaluations run in a background thread. Poll the `/status` endpoint for progress and use `/report` to fetch results.

---

## API

### TF-IDF endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/evaluate/tfidf-baseline` | Start a TF-IDF evaluation |
| `GET` | `/evaluate/tfidf-baseline/status` | Check if running / report available |
| `GET` | `/evaluate/tfidf-baseline/report` | Fetch the latest report |

**Request body**
```json
{ "max_cases": 0, "test_size": 0.2 }
```

### XAI endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/evaluate/xai` | Start an XAI evaluation |
| `GET` | `/evaluate/xai/status` | Check if running / report available |
| `GET` | `/evaluate/xai/report` | Fetch the latest report |

**Request body**
```json
{
  "max_cases": 0,
  "max_correct_cases": 150,
  "max_undertriage_cases": 50,
  "max_stability_cases": 30,
  "max_fidelity_cases": 30,
  "max_consistency_cases": 30
}
```

`max_cases=0` means no cap. Each stability/fidelity/consistency case costs 3×, 2×, 2× LLM calls respectively.

---

## Running Locally

```bash
cd services/evaluation-service

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8017 --reload
```

### Prerequisites

- MongoDB running with MIMIC-IV data loaded into `mimic_iv_records`
- XAI validation service running on port 8016 (for XAI evaluation only)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection string |
| `MONGO_DB` | `agentic_ai_healthcare_db` | Database name |
| `MONGO_MIMIC_COLLECTION` | `mimic_iv_records` | MIMIC-IV evaluation records |
| `MONGO_CANCER_AGENT_REPORT_COLLECTION` | `cancer_agent_report` | TF-IDF report storage |
| `MONGO_XAI_REPORT_COLLECTION` | `xai_evaluation_reports` | XAI report storage |
| `XAI_SERVICE_URL` | `http://localhost:8016` | XAI validation service base URL |
| `EVALUATION_SVC_REPORTS` | `evaluation-svc-reports` | Base directory for plot outputs |

---

## Source Structure

```
src/
├── api/
│   └── server.py                  # All REST endpoints
├── core/
│   ├── config.py                  # Env vars + report output paths
│   └── mongo_client.py            # MongoDB read/write helpers
├── datamodel/
│   └── models.py                  # Request/response Pydantic models
├── evaluators/
│   ├── metrics_calculator.py      # ROC-AUC, PR-AUC, F1, accuracy
│   ├── system_monitor.py          # Latency and throughput tracking
│   ├── label_mapper.py            # MIMIC-IV → ground truth label conversion
│   ├── tfidf_baseline_evaluator.py # TF-IDF + classifier pipeline
│   └── xai_evaluator.py           # All 9 XAI metrics
├── graph/                         # Matplotlib/Seaborn report generators
│   ├── cancer_agent_aggregate_report_graph_generator.py
│   ├── cancer_agent_statistical_report_graph_generator.py
│   ├── xai_aggregate_report_graph_gnerator.py
│   └── xai_statistical_report_graph_gnerator.py
├── service/
│   └── evaluation_service.py      # Background thread orchestration
├── exception/
├── log/
└── main.py
```
