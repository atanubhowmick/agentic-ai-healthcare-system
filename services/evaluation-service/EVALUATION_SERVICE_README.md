# Evaluation Service

System monitoring and metrics service — port **8017**. Evaluates the XAI validation service quality using MIMIC-IV test cases drawn from MongoDB.

---

## How It Works

Loads MIMIC-IV cases from MongoDB and runs them through the XAI validation service to measure clinical safety, explainability quality, and consistency.

### Evaluation Options

| Option | Metric | What it measures |
|--------|--------|------------------|
| 1 | Decision accuracy | Correctly diagnosed cases that pass XAI without false rejection |
| 2 | Safety net effectiveness | Undertriage cases (HIGH/CRITICAL labelled LOW) that are correctly caught |
| 4 | Rule engine coverage | Fraction of cases handled by deterministic rules vs LLM path |
| 6 | Over-rejection rate | Inverse of Option 1 — false positive rejection rate |
| XAI Sparsity | Focus | Avg key concerns per response + avg summary word count |
| XAI Interpretability | Readability | Flesch Reading Ease of validation summaries |
| XAI Stability | Determinism | Same payload sent 3× — fraction with consistent recommendation |
| XAI Fidelity | Faithfulness | Severity perturbed — does explanation reference the changed factor? |
| XAI Consistency | Robustness | Paraphrased symptoms — fraction with same recommendation |

---

## API Endpoints

### `POST /evaluate/xai`

Run XAI evaluation batch.

**Request**
```json
{
  "max_cases": 0,
  "max_correct_cases": 50,
  "max_undertriage_cases": 20,
  "max_stability_cases": 10,
  "max_fidelity_cases": 10,
  "max_consistency_cases": 10
}
```

- `max_cases`: total case cap (0 = no cap)
- `max_correct_cases`: cap for Option 1/6 (LLM calls = this number)
- `max_undertriage_cases`: cap for Option 2
- `max_stability_cases`: cases × 3 LLM calls each
- `max_fidelity_cases`: cases × 2 LLM calls each
- `max_consistency_cases`: cases × 2 LLM calls each

**Response**
```json
{
  "is_success": true,
  "payload": {
    "run_at": "2026-03-30T...",
    "total_cases_loaded": 28457,
    "elapsed_seconds": 1080.9,
    "decision_accuracy": { "approval_accuracy": 0.27, "..." },
    "safety_net_effectiveness": { "sensitivity": 1.0, "..." },
    "rule_engine_coverage": { "rule_coverage_rate": 0.20, "..." },
    "over_rejection_rate": { "over_rejection_rate": 0.73, "..." },
    "xai_sparsity": { "avg_key_concerns_per_response": 1.9, "..." },
    "xai_interpretability": { "avg_flesch_reading_ease": 9.5, "..." },
    "xai_stability": { "stability_rate": 0.80, "..." },
    "xai_fidelity": { "explanation_fidelity_rate": 1.0, "..." },
    "xai_consistency": { "consistency_rate": 0.50, "..." }
  }
}
```

---

## Running Locally

```bash
cd services/evaluation-service

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt

bash run.sh
# or
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8017 --reload
```

### Prerequisites

- MongoDB running with MIMIC-IV data loaded
- XAI validation service running on port 8016

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `MONGODB_DB_NAME` | `healthcare_db` | Database name |
| `XAI_SERVICE_URL` | `http://localhost:8016` | XAI service base URL |

---

## Source Structure

```
src/
├── api/            # FastAPI router
├── core/
│   ├── config.py
│   └── mongo_client.py     # MongoDB client + report persistence
├── datamodel/      # Pydantic models (XaiEvaluationRequest etc.)
├── evaluators/
│   ├── tfidf_baseline_evaluator.py
│   └── xai_evaluator.py    # All 9 metrics implemented here
├── service/        # evaluation_service.py (orchestrates evaluators)
├── exception/
├── log/
└── main.py
```
