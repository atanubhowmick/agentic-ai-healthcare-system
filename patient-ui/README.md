# Patient UI

Patient-facing Streamlit web application — port **8021**. Provides a two-page flow for patient check-in and symptom submission, then renders the full structured diagnosis and treatment report returned by the Orchestrator.

---

## How It Works

```
Patient Check-in (Patient ID, Name, Gender, Age Group)
        │
        ▼
Diagnosis Page — symptom text area (max 1 000 characters)
        │  POST /orchestrator/diagnose
        ▼
Diagnosis Report
  ├── Status badge (COMPLETED / HUMAN_REVIEW_REQUIRED)
  ├── Severity + Emergency / Hospitalisation flags
  ├── Diagnosis Summary + Full Details (expandable)
  ├── Treatment Recommendations (expandable)
  ├── XAI Diagnosis Validation (expandable)
  ├── XAI Treatment Validation (expandable)
  └── Audit Trail (expandable)
```

Gender and age group collected at check-in are prepended to the symptom text before sending to the Orchestrator, giving the specialist agents richer context.

---

## Pages

| Page | File | Description |
|------|------|-------------|
| **Patient Check-in** | `pages/1_patient_login.py` | Captures Patient ID, Full Name, Gender, and Age Group before proceeding |
| **Diagnosis** | `pages/2_diagnosis.py` | Symptom input form; calls the Orchestrator and renders the full structured report |

---

## Components

| Component | File | Description |
|-----------|------|-------------|
| `render_banner()` | `components/banner.py` | Blue top-bar with app title; optionally displays patient name and ID |
| `render_footer()` | `components/banner.py` | Clinical disclaimer footer |

---

## Running Locally

```bash
cd patient-ui

python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

bash run.sh
# or directly:
streamlit run app.py --server.port 8021 --server.address 127.0.0.1
```

Open `http://localhost:8021` in your browser.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_URL` | `http://127.0.0.1:8015` | Base URL of the Orchestrator Agent |

---

## Source Structure

```
patient-ui/
├── app.py                      # Entry point — redirects to patient check-in
├── run.sh                      # Starts Streamlit on port 8021
├── components/
│   └── banner.py               # Shared top-bar and footer components
├── constant/
│   └── constants.py            # ORCHESTRATOR_URL_DEFAULT, MAX_SYMPTOMS_CHARS
└── pages/
    ├── 1_patient_login.py      # Patient check-in page
    └── 2_diagnosis.py          # Symptom input + diagnosis report page
```
