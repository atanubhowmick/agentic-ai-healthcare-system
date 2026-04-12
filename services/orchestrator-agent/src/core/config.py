import os
from dotenv import load_dotenv

load_dotenv()

CARDIOLOGY_SERVICE_URL = os.getenv(
    "CARDIOLOGY_SERVICE_URL", "http://127.0.0.1:8001/cardiology-agent"
)
NEUROLOGY_SERVICE_URL = os.getenv(
    "NEUROLOGY_SERVICE_URL", "http://127.0.0.1:8002/neurology-agent"
)
CANCER_SERVICE_URL = os.getenv(
    "CANCER_SERVICE_URL", "http://127.0.0.1:8003/cancer-agent"
)
PATHOLOGY_SERVICE_URL = os.getenv(
    "PATHOLOGY_SERVICE_URL", "http://127.0.0.1:8011/pathology-agent"
)
TREATMENT_SERVICE_URL = os.getenv(
    "TREATMENT_SERVICE_URL", "http://127.0.0.1:8012/treatment-agent"
)
XAI_SERVICE_URL = os.getenv(
    "XAI_SERVICE_URL", "http://127.0.0.1:8016/xai-validator"
)

MONGO_URI                  = os.getenv("MONGO_URI",                  "mongodb://127.0.0.1:27017")
MONGO_DB                   = os.getenv("MONGO_DB",                   "agentic_ai_healthcare_db")
MONGO_PATIENT_RECORDS_COLLECTION = os.getenv("MONGO_PATIENT_RECORDS_COLLECTION", "patient_diagnosis_treatment_records")

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))
CHROMA_SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.85"))

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")

ORCHESTRATOR_AGENT_ID = "ORCH-AGENT-1000"

MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))

# Zero-shot NLI hypothesis labels and valid routing targets.
CLASSIFIER_SPECIALISTS: list[str] = ["cardiology", "neurology", "cancer", "pathology"]

RULE_DOMINANCE_RATIO          = float(os.getenv("RULE_DOMINANCE_RATIO",          "0.80"))
RULE_MIN_KEYWORD_HITS         = int(os.getenv("RULE_MIN_KEYWORD_HITS",           "4"))
BIOBERT_CONFIDENCE_THRESHOLD  = float(os.getenv("BIOBERT_CONFIDENCE_THRESHOLD",  "0.80"))
CLINICAL_CONFIDENCE_THRESHOLD = float(os.getenv("CLINICAL_CONFIDENCE_THRESHOLD", "0.80"))

# Path to the fine-tuned ClinicalBERT model dir (produced by train_clinicalbert.py).
# Tier 3 is skipped if the path does not exist.
CLINICALBERT_MODEL_DIR = os.getenv("CLINICALBERT_MODEL_DIR", "./clinicalbert_router")

# Label map shared by training scripts and the runtime classifier.
# Extended labels (gastroenterology, etc.) are declared for future training data.
LABEL2ID: dict[str, int] = {
    "cardiology":         0,
    "neurology":          1,
    "cancer":             2,
    "pathology":          3,
    "gastroenterology":   4,
    "dermatology":        5,
    "orthopedics":        6,
    "pulmonology":        7,
    "urology":            8,
    "endocrinology":      9,
    "psychiatry":         10,
    "ophthalmology":      11,
    "rheumatology":       12,
    "nephrology":         13,
    "gynecology":         14,
    "hematology":         15,
    "infectious_disease": 16,
    "allergy":            17,
    "otolaryngology":     18,
    "unknown":            19,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
