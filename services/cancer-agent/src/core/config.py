import os
from dotenv import load_dotenv

load_dotenv()

# ── ChromaDB (MIMIC-IV vector store) ─────────────────────────────────────────
# Separate store from the orchestrator's treatment cache
CHROMA_PERSIST_DIR = os.getenv("CANCER_CHROMA_PERSIST_DIR", "./cancer_chroma_store")
MIMIC_COLLECTION_NAME = "mimic_cancer_cases"

# Minimum cosine similarity to accept a MIMIC case as a strong match (full RAG context).
MIMIC_SIMILARITY_THRESHOLD = float(os.getenv("MIMIC_SIMILARITY_THRESHOLD", "0.85"))

# Below MIMIC_SIMILARITY_THRESHOLD but above this value → partial context (weaker match).
# The case is still injected into the prompt but flagged as low-confidence reference.
# Below this value → pure LLM call with no MIMIC context.
MIMIC_PARTIAL_THRESHOLD = float(os.getenv("MIMIC_PARTIAL_THRESHOLD", "0.60"))

# Number of MIMIC cases to retrieve and inject as RAG context
MIMIC_TOP_K = int(os.getenv("MIMIC_TOP_K", "3"))
