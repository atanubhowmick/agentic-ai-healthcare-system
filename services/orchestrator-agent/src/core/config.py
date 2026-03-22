import os
from dotenv import load_dotenv

load_dotenv()

# -- Specialist agent URLs ---------------------------------------------------
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

# -- MongoDB -----------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "agentic_ai_healthcare_db")

# -- ChromaDB (external HTTP server) -----------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))
CHROMA_SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.85"))

# -- OpenAI model --------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")

# -- Orchestrator identity ----------------------------------------------------
ORCHESTRATOR_AGENT_ID = "ORCH-AGENT-1000"

# -- Orchestrator settings ----------------------------------------------------
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))

# -- Classifier router: BioBERT zero-shot candidate labels -------------------
# These are the specialists the orchestrator has agents for.
# Used as zero-shot NLI hypothesis labels and as the valid routing targets.
CLASSIFIER_SPECIALISTS: list[str] = ["cardiology", "neurology", "cancer", "pathology"]

# -- Classifier router thresholds ---------------------------------------------
RULE_DOMINANCE_RATIO          = float(os.getenv("RULE_DOMINANCE_RATIO",          "0.80"))
RULE_MIN_KEYWORD_HITS         = int(os.getenv("RULE_MIN_KEYWORD_HITS",           "4"))
BIOBERT_CONFIDENCE_THRESHOLD  = float(os.getenv("BIOBERT_CONFIDENCE_THRESHOLD",  "0.80"))
CLINICAL_CONFIDENCE_THRESHOLD = float(os.getenv("CLINICAL_CONFIDENCE_THRESHOLD", "0.80"))

# -- Fine-tuned ClinicalBERT classifier (Tier 3) ------------------------------
# Path to the saved fine-tuned model directory (produced by train_clinicalbert.py).
# If the path does not exist, Tier 3 is skipped and the cascade falls through to LLM.
CLINICALBERT_MODEL_DIR = os.getenv("CLINICALBERT_MODEL_DIR", "./clinicalbert_router")

# -- ClinicalBERT label map ---------------------------------------------------
# Single source of truth for the specialist/domain classification labels.
# Shared by the training scripts and the runtime classifier in classifier_router.py.
# New domains without keyword rules will be excluded from the active training set
# automatically (see train_clinicalbert_classifier.py) but are declared here so
# they are ready once training data becomes available.
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
