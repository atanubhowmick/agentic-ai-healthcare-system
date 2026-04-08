import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
XAI_AGENT_ID = os.getenv("XAI_AGENT_ID", "XAI-VALIDATOR-1003")

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))

MONGODB_URI     = os.getenv("MONGODB_URI",      "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME",  "agentic_ai_healthcare_db")
