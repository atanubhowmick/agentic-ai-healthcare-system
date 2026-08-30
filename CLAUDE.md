# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent AI system for clinical decision support, built with FastAPI, LangChain/LangGraph, and DeepAgents. Independent specialist microservices analyse patient symptoms and produce structured diagnostic/treatment output; an orchestrator routes cases through a LangGraph pipeline with an XAI (explainable AI) validation layer that enforces clinical safety rules before results are returned to the patient.

Each service is a fully independent FastAPI app with its own `venv`, `requirements.txt`, `Dockerfile`, and `run.sh` — there is no shared Python package or monorepo tooling. Cross-service communication is plain HTTP (see `tools/`/`*_client.py` wrappers in the orchestrator).

## Services and Ports

| Service | Path | Port | Role |
|---|---|---|---|
| Cardiology Agent | `services/cardiology-agent` | 8001 | Cardiac symptoms → diagnosis |
| Neurology Agent | `services/neurology-agent` | 8002 | Neuro symptoms → diagnosis |
| Cancer Agent | `services/cancer-agent` | 8003 | Oncology: TNM staging, TF-IDF + DeepAgent + MIMIC-IV RAG |
| Pathology Agent | `services/pathology-agent` | 8011 | Lab/biomarker analysis, secondary cross-check |
| Treatment Agent | `services/treatment-agent` | 8012 | Care plan generation |
| Orchestrator Agent | `services/orchestrator-agent` | 8015 | LangGraph master: classify → route → validate → treat |
| XAI Validation Service | `xai-validation-service` | 8016 | Rule engine + SHAP + constitutional guard, gates diagnosis & treatment |
| Evaluation Service | `services/evaluation-service` | 8017 | Offline metrics: TF-IDF baseline, XAI quality (9 metrics) |
| ChromaDB | (external container) | 8020 | Shared vector store: MIMIC RAG + semantic diagnosis/treatment cache |
| MongoDB | (external container) | 27017 | Persisted patient cases, MIMIC-IV records, XAI rules |
| Patient UI | `patient-ui` | 8021 | Streamlit patient-facing app (check-in → symptoms → report) |

Each service exposes Swagger at `http://localhost:<port>/docs`. Every service has its own detailed `README.md` — read the relevant one before working in that service; the root `README.md` also documents ChromaDB/MongoDB setup end to end.

## Common Commands

There is no root-level build/test/lint tooling — everything is per-service. There are currently no automated tests or linters configured in this repo.

Run a single service locally (pattern is identical across all services/`patient-ui`):
```bash
cd services/<service-name>       # or xai-validation-service, or patient-ui
python -m venv venv
source venv/Scripts/activate     # Windows Git Bash
pip install -r requirements.txt
bash run.sh
# equivalent direct command (FastAPI services):
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port <port> --reload
# patient-ui is Streamlit, not FastAPI:
streamlit run app.py --server.port 8021 --server.address 127.0.0.1
```

Run the full stack with Docker Compose (requires `OPENAI_API_KEY`, `CHROMA_DATA_PATH`, `MONGO_DB_PATH` set first):
```bash
export OPENAI_API_KEY=...
export CHROMA_DATA_PATH="Path_to_Chroma_DB/chroma_data"
export MONGO_DB_PATH="Path_to_Mongo_DB/mongo_data"
docker-compose up --build
```

Infra-only, for local (non-Docker) service development:
```bash
chroma run --host 127.0.0.1 --port 8020 --path "$CHROMA_DATA_PATH"
mongod --dbpath "$MONGO_DB_PATH" --port 27017 --quiet
```

Loading MIMIC-IV data (cancer-agent, requires GCP BigQuery credentials + running Mongo/Chroma):
```bash
cd services/cancer-agent
python scripts/load_mimic_mongo.py                                  # → MongoDB mimic_iv_records
python scripts/load_mimic_data.py --project YOUR-GCP-PROJECT-ID     # → ChromaDB mimic_cancer_cases
python scripts/train_models.py                                      # → trained_model/cancer_agent_models.pkl (used by XAI SHAP)
```

Service startup order matters: the orchestrator calls all specialist agents + XAI service directly over HTTP, so bring up Mongo/Chroma, then the specialists (8001/8002/8003/8011), treatment (8012), and XAI (8016) before starting the orchestrator (8015). `OPENAI_API_KEY` is required by every service that calls an LLM (all except ChromaDB/MongoDB).

## Architecture

### Orchestrator pipeline (`services/orchestrator-agent`)

This is the entry point for all patient cases (`POST /orchestrator/diagnose`), implemented as a LangGraph state machine (`src/agents/graph.py` builds nodes+edges, `src/agents/nodes.py` implements them, `src/agents/state.py` defines `AgentState`):

```
chroma_lookup ──── cache hit (similarity ≥ threshold) ─────────────────────┐
     │ miss                                                                │
     ▼                                                                     │
classifier (4-tier hybrid, src/agents/classifier_router.py)                │
     ▼                                                                     │
specialist_node (cardiology / neurology / cancer / pathology) ─────────┐   │
     ├── secondary_check_needed → pathology cross-check → conflict_check│  │
     ▼                                                                  │  │
xai_diagnosis_validator ◄──── fail: retry (max MAX_RETRY_COUNT) ────────┘  │
     │ pass                                                                │
     ▼                                                                     │
treatment_node                                                             │
     ▼                                                                     │
xai_treatment_validator ◄──── fail: retry (max MAX_RETRY_COUNT) ───────────┤
     │ pass                                                                │
     ▼                                                                     │
finish ◄─────────────────────────────────────────────────────────────────┘
     │
     ▼
MongoDB save + ChromaDB cache update (non-blocking)
```

- **4-tier classifier** (escalates only when confidence is below threshold): (1) keyword rule router, (2) BioBERT zero-shot NLI, (3) fine-tuned ClinicalBERT (skipped if model absent at `CLINICALBERT_MODEL_DIR`), (4) LLM fallback (always returns).
- **Conflict detection**: when pathology cross-check triggers, an LLM compares primary vs secondary severity; a conflict routes straight to `finish` with `status: HUMAN_REVIEW_REQUIRED`.
- **Retry exhaustion**: after `MAX_RETRY_COUNT` (default 3) failed XAI validations, the case finishes with `status: HUMAN_REVIEW_REQUIRED` instead of looping forever.
- **Semantic cache**: `chroma_lookup` short-circuits the whole pipeline on a cache hit (`CHROMA_SIMILARITY_THRESHOLD`), returning `status: COMPLETED_FROM_CACHE` without re-running XAI validation.

### XAI Validation Service (`xai-validation-service`)

Gates every diagnosis and every treatment plan before it reaches the patient. Called twice per case by the orchestrator. Pipeline: (1) fast keyword/SpO2/severity pre-filter → immediate reject, (2) rule engine (46 hardcoded + LLM-generated rules loaded from MongoDB, JSON-file fallback) → REJECT hard-stops, REVIEW injects a concern into the LLM query, (3) SHAP explainability precompute (diagnosis only, from `trained_model/cancer_agent_models.pkl`, LLM fallback if absent), (4) single DeepAgent LLM call with rules+explainability injected, (5) constitutional critique→revision loop enforcing 5 safety principles P1–P5 (diagnosis only) — falls back to the pre-revision response if the revision LLM call produces invalid JSON.

### Cancer Agent (`services/cancer-agent`)

Differs from the other specialists by combining two independent prediction paths merged into one response: a **TF-IDF ML pipeline** (4 classifiers — severity, emergency, hospitalisation, cancer type — trained from MongoDB MIMIC-IV records at service startup; skipped with safe defaults if no records are found) and a **DeepAgent LLM** with a `search_mimic_cases` tool that does ChromaDB similarity RAG over historical MIMIC-IV cases (three similarity tiers: high-confidence context ≥0.75, low-confidence flagged context ≥0.60, LLM-only below both).

### Shared conventions across services

- Every service follows the same internal layout: `agent/` (DeepAgent/LangChain executor + `@tool`s), `api/` (FastAPI router, `server.py`), `core/config.py` (env vars), `datamodel/` (Pydantic request/response models), `exception/` (a `<Service>SvcException` hierarchy + FastAPI exception handler), `service/` (business logic), `log/logger.py`, `main.py`.
- All responses use a `GenericResponse[T]` envelope: `{is_success, payload, error}`. Diagnosis payloads nest under `diagnosis`, treatment under `treatment`.
- Error envelope on failure: `{is_success: false, error: {code, message}}`, with codes `LLM_INVOCATION_ERROR`, `LLM_RESPONSE_PARSE_ERROR`, `VALIDATION_LLM_ERROR`, `VALIDATION_PARSE_ERROR`, `INTERNAL_SERVER_ERROR`.
- All LLM calls use `gpt-5.2` (`OPENAI_DEFAULT_MODEL`, `ChatOpenAI`, temperature=0) unless a service overrides it via env var.
- ChromaDB is always an external HTTP server (never embedded) so all agents share one vector store; collections are `mimic_cancer_cases` (cancer agent RAG), `diagnosis_outcomes` / `treatment_outcomes` (orchestrator semantic cache).
- MongoDB collections: `mimic_iv_records` (training/eval data), `cancer_agent_report` / `xai_evaluation_reports` (evaluation service outputs), `xai_validation_rules` (safety rules, JSON-seeded then MongoDB-overridden), `patient_diagnosis_treatment_records` (orchestrator's completed cases).
- `trained_model/` and `**/clinicalbert_router/` are gitignored (large generated artifacts) — regenerate via `cancer-agent/scripts/train_models.py` and the orchestrator's `training/` ClinicalBERT script rather than expecting them to be present in a fresh checkout.

### Observability — LangSmith distributed tracing

The 7 services in the live diagnosis pipeline (all specialists + treatment + orchestrator + XAI validation — not `evaluation-service` or `patient-ui`) each have a `core/tracing.py` with `LangSmithTracingMiddleware` (added to the FastAPI app in `main.py`) and, in the orchestrator only, a `trace_headers()` helper used by every `tools/*_client.py` HTTP call. The middleware reads an incoming `langsmith-trace` header and continues that trace via `tracing_context(parent=...)`; `trace_headers()` serializes the orchestrator's currently active run (`get_current_run_tree()`) onto outgoing requests. Together this makes one patient case appear as a single connected trace tree across all 7 processes rather than disconnected per-service traces — see the root README's "Observability — LangSmith Tracing" section for the full mechanism and env vars (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`). Tracing is opt-in (`LANGSMITH_TRACING=false` by default) and requires no other code changes — LangChain/LangGraph/DeepAgents auto-trace LLM calls, tool invocations, and graph nodes once enabled.
