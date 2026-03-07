# Agentic AI Healthcare System

A multi-agent AI system for clinical decision support, built with FastAPI and LangChain. Specialist agents independently analyse patient data and produce structured diagnostic and treatment outputs, with an XAI validation layer that enforces clinical safety rules and ethical consistency before results are returned.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Agentic AI Healthcare System                │
├─────────────────────────────────┬────────────────────────────┤
│       Specialist Agents         │      Support Services      │
│                                 │                            │
│  ┌───────────────────────────┐  │  ┌──────────────────────┐  │
│  │  Cardiology Agent (:8001) │  │  │  XAI Validation      │  │
│  │  Neurology Agent  (:8005) │──┼─▶│  Service     (:8004)│  │
│  │  Pathology Agent  (:8002) │  │  └──────────────────────┘  │
│  └───────────────────────────┘  │                            │
│                                 │  ┌──────────────────────┐  │
│  ┌───────────────────────────┐  │  │  Treatment Agent     │  │
│  │  LangChain ReAct Executor │  │  │             (:8003)  │  │
│  │  + Conversation Memory    │  │  └──────────────────────┘  │
│  └───────────────────────────┘  │                            │
└─────────────────────────────────┴────────────────────────────┘
```

### Services

| # | Service | Description | Port |
|---|---------|-------------|------|
| 1 | **Cardiology Agent** | Analyses cardiac symptoms, flags anomalies, recommends cardiac workup | 8001 |
| 2 | **Pathology Agent** | Analyses lab results and biomarker abnormalities | 8002 |
| 3 | **Treatment Agent** | Generates comprehensive treatment and patient care plans | 8003 |
| 4 | **XAI Validation Service** | LLM-based clinical safety validation with rule-based checks and SHAP explainability | 8004 |
| 5 | **Neurology Agent** | Analyses neurological symptoms, recommends imaging and neurological tests | 8005 |

---

## Swagger / OpenAPI URLs

Each service exposes interactive API documentation via FastAPI's built-in Swagger UI and ReDoc.

| Service | Swagger UI | ReDoc |
|---------|-----------|-------|
| Cardiology Agent | http://localhost:8001/docs | http://localhost:8001/redoc |
| Neurology Agent | http://localhost:8002/docs | http://localhost:8002/redoc |
| Pathology Agent | http://localhost:8011/docs | http://localhost:8011/redoc |
| Treatment Agent | http://localhost:8012/docs | http://localhost:8012/redoc |
| Orchestrator Svc | http://localhost:8015/docs | http://localhost:8015/redoc |
| XAI Validation Service | http://localhost:8016/docs | http://localhost:8016/redoc |
| Evaluation Service | http://localhost:8017/docs | http://localhost:8017/redoc |

---

## API Endpoints

### Cardiology Agent — `http://localhost:8001`

#### `POST /cardiology-agent/diagnose`
Analyse cardiac symptoms for a patient.

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Chest pain radiating to left arm, shortness of breath, diaphoresis",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Cardiology_Specialist",
    "agent_id": "CARDIOLOGY-AGENT-1001",
    "diagnosis": {
      "diagnosysDetails": "...",
      "severity": "HIGH",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "...",
      "bloodTestsRequired": ["Troponin I", "BNP"],
      "labTestsRequired": ["ECG", "Echocardiogram"],
      "medication": "..."
    }
  }
}
```

---

### Pathology Agent — `http://localhost:8011`

#### `POST /pathology-agent/diagnose`
Analyse laboratory results and biomarker data.

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
      "diagnosysDetails": "...",
      "severity": "MODERATE",
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

### Treatment Agent — `http://localhost:8012`

#### `POST /treatment-agent/recommend`
Generate a treatment and patient care plan based on a specialist diagnosis.

**Request**
```json
{
  "patient_id": "P001",
  "diagnosis": "Acute STEMI with cardiogenic shock",
  "specialist_notes": "Cardiology_Specialist — hospitalization required, emergency care needed",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Treatment_Care_Agent",
    "agent_id": "TREAT-AGENT-1004",
    "treatment": {
      "treatmentPlan": "...",
      "medications": ["Aspirin – 325mg – once daily", "Heparin – weight-based – IV infusion"],
      "followUpRequired": "YES",
      "followUpTimeframe": "1 week",
      "lifestyleRecommendations": ["Low-sodium diet", "Cardiac rehabilitation"],
      "monitoringRequired": ["BP", "Troponin", "ECG"],
      "referralRequired": "Cardiac surgeon",
      "urgency": "IMMEDIATE"
    }
  }
}
```

---

### XAI Validation Service — `http://localhost:8016`

#### `POST /xai-validator/validate-diagnosis`
Validate a specialist diagnosis against clinical safety rules and LLM-based ethical checks.

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
    "agent_id": "XAI-AGENT-2001",
    "patient_id": "P001",
    "validation_type": "DIAGNOSIS",
    "result": {
      "is_validated": true,
      "confidence_score": 0.92,
      "validation_summary": "Diagnosis is clinically consistent with presented symptoms.",
      "key_concerns": [],
      "recommendation": "APPROVE"
    }
  }
}
```

#### `POST /xai-validator/validate-treatment`
Validate a treatment plan for clinical appropriateness and safety.

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
    "agent_id": "XAI-AGENT-2001",
    "patient_id": "P001",
    "validation_type": "TREATMENT",
    "result": {
      "is_validated": true,
      "confidence_score": 0.95,
      "validation_summary": "Treatment plan is appropriate for the diagnosis severity.",
      "key_concerns": [],
      "recommendation": "APPROVE"
    }
  }
}
```

---

### Neurology Agent — `http://localhost:8002`

#### `POST /neurology-agent/diagnose`
Analyse neurological symptoms and recommend imaging and neurological tests.

**Request**
```json
{
  "patient_id": "P001",
  "symptoms": "Sudden onset severe headache, neck stiffness, photophobia, nausea",
  "is_followup": false
}
```

**Response**
```json
{
  "is_success": true,
  "payload": {
    "agent": "Neurology_Specialist",
    "agent_id": "NEURO-AGENT-1002",
    "diagnosis": {
      "diagnosisDetails": "...",
      "severity": "CRITICAL",
      "hospitalizationNeeded": "YES",
      "emergencyCareNeeded": "YES",
      "clarificationQuestion": "...",
      "bloodTestsRequired": ["CBC", "CRP", "Blood culture"],
      "imagingRequired": ["CT head without contrast", "MRI brain"],
      "neurologicalTestsRequired": ["Lumbar puncture"],
      "medication": "..."
    }
  }
}
```

---

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

## Running Locally

Each service is a standalone FastAPI application. Run from the service's root directory:

```bash


# Step-1: # Go to respective service's directory. Example for Cardiology agent go to 'services/cardiology-agent'
cd services/cardiology-agent

# Step-2: Create the virtual environment
python -m venv venv

# Step-3: Activate the virtual environment
source venv/Scripts/activate

# Step-4: Install the required libraries
pip install -r requirements.txt

# Step-5: Run using the SH file (Need GitBash)
bash run.sh

# Step-5 (Alternate) : Using Uvicorn
uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8001 --reload

```

### Environment Variables

| Variable | Description | Required By |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | All services |

---

## Running with Docker Compose

```bash
docker-compose up --build
```

---

## Project Structure

```
agentic-ai-healthcare-system/
├── services/
│   ├── cardiology-agent/
│   │   └── src/
│   │       ├── agent/          # LangChain ReAct executor
│   │       ├── api/            # FastAPI router (server.py)
│   │       ├── datamodel/      # Pydantic request/response models
│   │       ├── exception/      # CardiologySvcException + handler
│   │       ├── service/        # Business logic (cardiology_service.py)
│   │       ├── log/
│   │       └── main.py
│   ├── neurology-agent/        # Same structure; NeurologySvcException
│   ├── pathology-agent/        # Same structure; PathologySvcException
│   └── treatment-agent/        # Same structure; TreatmentSvcException
├── xai-validation-service/
│   └── src/
│       ├── api/                # FastAPI router (server.py)
│       ├── datamodel/          # Validation request/response models
│       ├── exception/          # ValidationSvcException + handler
│       ├── explainers/         # SHAP-based explainability (shap_provider.py)
│       ├── service/            # Business logic (validator_service.py)
│       ├── validators/         # Rule-based checks + ethical_guard LLM validator
│       ├── log/
│       └── main.py
├── docker-compose.yml
└── README.md
```