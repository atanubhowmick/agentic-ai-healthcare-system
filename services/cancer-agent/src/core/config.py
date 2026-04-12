import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
AGENT_NAME = "Cancer_Oncology_Specialist"
AGENT_ID = "CANCER-AGENT-1004"

# MongoDB — MIMIC evaluation cases (shared with evaluation-service)
MONGO_URI             = os.getenv("MONGO_URI",             "mongodb://127.0.0.1:27017")
MONGO_DB              = os.getenv("MONGO_DB",              "agentic_ai_healthcare_db")
MONGO_MIMIC_COLLECTION = os.getenv("MONGO_MIMIC_COLLECTION", "mimic_iv_records")

# ChromaDB — external HTTP server shared with orchestrator
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))
MIMIC_COLLECTION_NAME = "mimic_cancer_cases"

# Cosine similarity thresholds for MIMIC case retrieval:
#   >= MIMIC_SIMILARITY_THRESHOLD → high-confidence RAG context
#   >= MIMIC_PARTIAL_THRESHOLD    → low-confidence RAG context (flagged in prompt)
#   below both                    → LLM-only, no MIMIC context
MIMIC_SIMILARITY_THRESHOLD = float(os.getenv("MIMIC_SIMILARITY_THRESHOLD", "0.75"))
MIMIC_PARTIAL_THRESHOLD    = float(os.getenv("MIMIC_PARTIAL_THRESHOLD",    "0.60"))
MIMIC_TOP_K                = int(os.getenv("MIMIC_TOP_K", "3"))
