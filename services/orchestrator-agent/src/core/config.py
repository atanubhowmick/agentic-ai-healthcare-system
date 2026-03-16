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
CHROMA_SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.90"))

# -- Orchestrator settings ----------------------------------------------------
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))

# -- Triage router thresholds -------------------------------------------------
RULE_DOMINANCE_RATIO          = float(os.getenv("RULE_DOMINANCE_RATIO",          "0.80"))
RULE_MIN_KEYWORD_HITS         = int(os.getenv("RULE_MIN_KEYWORD_HITS",           "4"))
BIOBERT_CONFIDENCE_THRESHOLD  = float(os.getenv("BIOBERT_CONFIDENCE_THRESHOLD",  "0.80"))
CLINICAL_CONFIDENCE_THRESHOLD = float(os.getenv("CLINICAL_CONFIDENCE_THRESHOLD", "0.80"))

# -- Fine-tuned ClinicalBERT classifier (Tier 3) ------------------------------
# Path to the saved fine-tuned model directory (produced by train_clinicalbert.py).
# If the path does not exist, Tier 3 is skipped and the cascade falls through to LLM.
CLINICALBERT_MODEL_DIR = os.getenv("CLINICALBERT_MODEL_DIR", "./clinicalbert_router")
