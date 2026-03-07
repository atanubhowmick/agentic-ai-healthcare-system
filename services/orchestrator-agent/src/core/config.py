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
PATHOLOGY_SERVICE_URL = os.getenv(
    "PATHOLOGY_SERVICE_URL", "http://127.0.0.1:8003/pathology-agent"
)
TREATMENT_SERVICE_URL = os.getenv(
    "TREATMENT_SERVICE_URL", "http://127.0.0.1:8004"
)
XAI_SERVICE_URL = os.getenv(
    "XAI_SERVICE_URL", "http://127.0.0.1:8006/xai-validator"
)

# ── MongoDB ─────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "healthcare_ai")

# ── Orchestrator settings ────────────────────────────────────────────────────
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))
