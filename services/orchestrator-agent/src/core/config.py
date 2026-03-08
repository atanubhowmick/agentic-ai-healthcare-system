import os
from dotenv import load_dotenv

load_dotenv()

# ── Specialist agent URLs ───────────────────────────────────────────────────
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

# ── MongoDB ─────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "healthcare_ai")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.90"))

# ── Orchestrator settings ────────────────────────────────────────────────────
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))
