import os
from dotenv import load_dotenv

load_dotenv()

# -- ChromaDB (external HTTP server, shared with orchestrator) ----------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))
MIMIC_COLLECTION_NAME = "mimic_cancer_cases"

# Minimum cosine similarity to accept a MIMIC case as a strong match (full RAG context).
MIMIC_SIMILARITY_THRESHOLD = float(os.getenv("MIMIC_SIMILARITY_THRESHOLD", "0.75"))

# Below MIMIC_SIMILARITY_THRESHOLD but above this value → partial context (weaker match).
# The case is still injected into the prompt but flagged as low-confidence reference.
# Below this value → pure LLM call with no MIMIC context.
MIMIC_PARTIAL_THRESHOLD = float(os.getenv("MIMIC_PARTIAL_THRESHOLD", "0.60"))

# Number of MIMIC cases to retrieve and inject as RAG context
MIMIC_TOP_K = int(os.getenv("MIMIC_TOP_K", "3"))
