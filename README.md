# Agentic AI Healthcare System

A multi-agent AI system for clinical decision support, built with FastAPI and LangChain. Specialist agents independently analyse patient data and produce structured diagnostic and treatment outputs, with an XAI validation layer that enforces clinical safety rules and ethical consistency before results are returned.

---

## Architecture

![Architecture Diagram](Architecture-Diagram.jpeg)


### Services

| # | Service | Description | Port |
|---|---------|-------------|------|
| 1 | **Cardiology Agent** | Analyses cardiac symptoms, flags anomalies, recommends cardiac workup | 8001 |
| 2 | **Neurology Agent** | Analyses neurological symptoms, recommends imaging and neurological tests | 8002 |
| 3 | **Cancer Agent** | Oncology assessment - TNM staging, tumour markers, biopsy and imaging guidance | 8003 |
| 4 | **Pathology Agent** | Analyses lab results and biomarker abnormalities | 8011 |
| 5 | **Treatment Agent** | Generates comprehensive treatment and patient care plans | 8012 |
| 6 | **Orchestrator Agent** | LangGraph master agent - classifier, ChromaDB cache, retry loops, XAI gating | 8015 |
| 7 | **XAI Validation Service** | LLM-based clinical safety validation with rule-based checks and SHAP explainability | 8016 |
| 8 | **Evaluation Service** | System monitoring and metrics calculation | 8017 |
| 9 | **ChromaDB** | Externalized vector store - shared by all agents for RAG and semantic caching | 8020 |
| 10 | **Patient UI** | Streamlit patient-facing web app - check-in, symptom input, and diagnosis report | 8021 |

---

## Swagger / OpenAPI / UI URLs

Each service exposes interactive API documentation via FastAPI's built-in Swagger UI and ReDoc.

| Service | Swagger UI | ReDoc |
|---------|-----------|-------|
| Cardiology Agent | http://localhost:8001/docs | http://localhost:8001/redoc |
| Neurology Agent | http://localhost:8002/docs | http://localhost:8002/redoc |
| Cancer Agent | http://localhost:8003/docs | http://localhost:8002/redoc |
| Pathology Agent | http://localhost:8011/docs | http://localhost:8011/redoc |
| Treatment Agent | http://localhost:8012/docs | http://localhost:8012/redoc |
| Orchestrator Svc | http://localhost:8015/docs | http://localhost:8015/redoc |
| XAI Validation Service | http://localhost:8016/docs | http://localhost:8016/redoc |
| Evaluation Service | http://localhost:8017/docs | http://localhost:8017/redoc |
| ChromaDB | http://localhost:8020/docs | http://localhost:8020/redoc |
| Patient UI | http://localhost:8021 | - |

---

## Service Documentation

Each service has its own detailed README covering API endpoints, implementation details, and local setup:

| Service | README |
|---------|--------|
| Cardiology Agent | [CARDIOLOGY_AGENT_README.md](services/cardiology-agent/CARDIOLOGY_AGENT_README.md) |
| Neurology Agent | [NEUROLOGY_AGENT_README.md](services/neurology-agent/NEUROLOGY_AGENT_README.md) |
| Cancer Agent | [CANCER_AGENT_README.md](services/cancer-agent/CANCER_AGENT_README.md) |
| Pathology Agent | [PATHOLOGY_AGENT_README.md](services/pathology-agent/PATHOLOGY_AGENT_README.md) |
| Treatment Agent | [TREATMENT_AGENT_README.md](services/treatment-agent/TREATMENT_AGENT_README.md) |
| Orchestrator Agent | [ORCHESTRATOR_AGENT_README.md](services/orchestrator-agent/ORCHESTRATOR_AGENT_README.md) |
| XAI Validation Service | [XAI_VALIDATION_SERVICE_README.md](xai-validation-service/XAI_VALIDATION_SERVICE_README.md) |
| Evaluation Service | [EVALUATION_SERVICE_README.md](services/evaluation-service/EVALUATION_SERVICE_README.md) |

---

## Error Response

All services return a consistent error envelope on failure.

## Error Response

All services return a consistent error envelope on failure.

```json
{
  "is_success": false,
  "error": {
    "code": "LLM_INVOCATION_ERROR",
    "message": "LLM call failed for patient P001: <detail>"
  }
}
```

| Error Code | Description |
|------------|-------------|
| `LLM_INVOCATION_ERROR` | The LLM call to the specialist agent failed |
| `LLM_RESPONSE_PARSE_ERROR` | The LLM response could not be parsed into the expected structure |
| `VALIDATION_LLM_ERROR` | The XAI validator LLM call failed |
| `VALIDATION_PARSE_ERROR` | The XAI validator response could not be parsed |
| `INTERNAL_SERVER_ERROR` | Unhandled internal error |

---

## Patient UI

A Streamlit web application that provides a patient-facing interface to the Agentic AI Healthcare System.

### Pages

| Page | File | Description |
|------|------|-------------|
| **Patient Check-in** | `pages/1_patient_login.py` | Entry point - captures Patient ID and Full Name before proceeding |
| **Diagnosis** | `pages/2_diagnosis.py` | Symptom input form; calls the Orchestrator and renders the full structured report |

### UI Flow

```
Patient Check-in (Patient ID + Name)
        │
        ▼
Diagnosis Page - symptom text area (max 2 000 chars)
        │  POST /orchestrator/diagnose
        ▼
Diagnosis Report card
  ├── Status badge (COMPLETED / HUMAN_REVIEW_REQUIRED)
  ├── Severity + Emergency / Hospitalisation flags
  ├── Diagnosis Summary + Full Details (expandable)
  ├── Treatment Recommendations (expandable)
  ├── XAI Diagnosis Validation (expandable)
  ├── XAI Treatment Validation (expandable)
  └── Audit Trail (expandable)
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| `render_banner()` | `components/banner.py` | Blue top-bar with app title; optionally displays patient name and ID |
| `render_footer()` | `components/banner.py` | Clinical disclaimer footer |

### Running the UI locally

```bash
cd patient-ui

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Start the app (port 8021)
bash run.sh

# Or with Streamlit directly
streamlit run app.py --server.port 8021 --server.address 127.0.0.1
```

Open `http://localhost:8021` in your browser.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_URL` | `http://127.0.0.1:8015` | Base URL of the Orchestrator Agent |

---

## Running Locally

Each service is independent. See the service-specific README linked in the [Service Documentation](#service-documentation) table above for setup steps, environment variables, and port numbers.

**Common requirement for all services:** set `OPENAI_API_KEY` in your environment or a `.env` file before starting.

---

## ChromaDB - Externalized Vector Store

ChromaDB runs as a standalone HTTP server (not embedded) so that:
- All agents share a single vector store instance
- Data persists across restarts via a bind-mounted host directory
- The `chroma_data/` folder is excluded from git (too large; ~300 MB after loading MIMIC-IV)

### Running ChromaDB locally (without Docker)

Set the `CHROMA_DATA_PATH` environment variable to your chosen storage directory, then start the server:

```bash
# Git Bash
export CHROMA_DATA_PATH="Path_to_Chroma_DB/chroma_data"
chroma run --host 127.0.0.1 --port 8020 --path "$CHROMA_DATA_PATH"
```

Verify it is running:
```bash
curl http://localhost:8020/api/v1/heartbeat
```

### ChromaDB collections

| Collection | Used By | Purpose |
|------------|---------|---------|
| `mimic_cancer_cases` | Cancer Agent | MIMIC-IV RAG - historical oncology cases for context retrieval |
| `diagnosis_outcomes` | Orchestrator | Semantic cache of validated diagnoses |
| `treatment_outcomes` | Orchestrator | Semantic cache of validated treatment plans |

### Loading MIMIC-IV data into ChromaDB

After ChromaDB is running, load oncology cases from BigQuery (requires GCP credentials):

```bash
cd services/cancer-agent
python scripts/load_mimic_data.py --project YOUR-GCP-PROJECT-ID --limit 50000
```

### Reusing locally loaded data with Docker

Set `CHROMA_DATA_PATH` before running Docker Compose so that the container bind-mounts your existing data directory:

```bash
# Git Bash
export CHROMA_DATA_PATH="E:/Atanu/Python/LJMU_MS/Database/chroma_data"
docker-compose up --build
```

The `docker-compose.yml` maps this path into the ChromaDB container at `/chroma/chroma`, so all previously loaded vectors are immediately available - no reload required.

---

## Running with Docker Compose

```bash
# Git Bash - with a custom ChromaDB data path (recommended)
export CHROMA_DATA_PATH="Path_to_Chroma_DB/chroma_data"
docker-compose up --build

# Default (stores chroma_data in the project directory)
docker-compose up --build
```

---

## Project Structure

```
agentic-ai-healthcare-system/
├── services/
│   │
│   ├── cardiology-agent/           # Cardiology Specialist - port 8001
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agent/              # DeepAgent executor + @tool (cardiology_agent.py)
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── core/               # config.py - OPENAI_DEFAULT_MODEL env var
│   │       ├── datamodel/          # Pydantic request/response models
│   │       ├── exception/          # CardiologySvcException + handler
│   │       ├── service/            # Business logic (cardiology_service.py)
│   │       ├── log/
│   │       └── main.py
│   │
│   ├── neurology-agent/            # Neurology Specialist - port 8002
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agent/              # DeepAgent executor + @tool (neurology_agent.py)
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── core/               # config.py - OPENAI_DEFAULT_MODEL env var
│   │       ├── datamodel/          # Pydantic request/response models
│   │       ├── exception/          # NeurologySvcException + handler
│   │       ├── service/            # Business logic (neurology_service.py)
│   │       ├── log/
│   │       └── main.py
│   │
│   ├── cancer-agent/               # Cancer / Oncology Specialist - port 8003
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agent/              # DeepAgent + @tool: search_mimic_cases (cancer_agent.py)
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── core/               # config.py - OPENAI_DEFAULT_MODEL, CHROMA_* env vars
│   │       ├── datamodel/          # Pydantic request/response models
│   │       ├── exception/          # CancerSvcException + handler
│   │       ├── rag/                # MIMIC-IV ChromaDB retriever (mimic_retriever.py)
│   │       ├── service/            # Business logic (cancer_service.py)
│   │       ├── log/
│   │       └── main.py
│   │
│   ├── pathology-agent/            # Pathology Specialist - port 8011
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agent/              # DeepAgent executor + @tool (pathology_agent.py)
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── constant/           # constants.py - PATHOLOGY_AGENT_ID
│   │       ├── core/               # config.py - OPENAI_DEFAULT_MODEL env var
│   │       ├── datamodel/          # Pydantic request/response models
│   │       ├── exception/          # PathologySvcException + handler
│   │       ├── service/            # Business logic (pathology_service.py)
│   │       ├── log/
│   │       └── main.py
│   │
│   ├── treatment-agent/            # Treatment & Patient Care Agent - port 8012
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agent/              # LangChain LLM executor (treatment_agent.py)
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── datamodel/          # Pydantic request/response models
│   │       ├── exception/          # TreatmentSvcException + handler
│   │       ├── service/            # Business logic (treatment_service.py)
│   │       ├── log/
│   │       └── main.py
│   │
│   ├── orchestrator-agent/         # Router / Master Agent - port 8015
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── agents/             # classifier_router.py - ClinicalBERT/BioBERT classifier
│   │       ├── api/                # FastAPI router (server.py)
│   │       ├── core/               # config.py - service URLs, Chroma, Mongo env vars
│   │       ├── exception/          # OrchestratorSvcException + handler
│   │       ├── schemas/            # Shared request/response schemas
│   │       ├── tools/              # HTTP client wrappers for specialist agents
│   │       ├── training/           # BERT model training scripts
│   │       ├── log/
│   │       ├── constants.py
│   │       └── main.py
│   │
│   └── evaluation-service/         # Metrics & Evaluation Service - port 8017
│       ├── Dockerfile
│       ├── requirements.txt
│       └── src/
│           ├── metrics_calculator.py
│           ├── system_monitor.py
│           └── main.py
│
├── xai-validation-service/         # XAI & Ethical Validator - port 8016
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── api/                    # FastAPI router (server.py)
│       ├── datamodel/              # Validation request/response models
│       ├── exception/              # ValidationSvcException + handler
│       ├── explainers/             # SHAP-based explainability (shap_provider.py)
│       ├── service/                # Business logic (validator_service.py)
│       ├── validators/             # Rule-based checks + ethical_guard LLM validator
│       ├── log/
│       ├── constants.py
│       └── main.py
│
├── patient-ui/                     # Patient-facing Streamlit Web App - port 8021
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                      # Entry point - redirects to patient check-in
│   ├── run.sh                      # Starts Streamlit on port 8021
│   ├── components/
│   │   └── banner.py               # Shared top-bar and footer components
│   ├── constant/
│   │   └── constants.py            # ORCHESTRATOR_URL_DEFAULT, MAX_SYMPTOMS_CHARS
│   └── pages/
│       ├── 1_patient_login.py      # Patient check-in page
│       └── 2_diagnosis.py          # Symptom input + diagnosis report page
│
├── docker-compose.yml              # Full stack orchestration (all services + infra)
└── README.md
```