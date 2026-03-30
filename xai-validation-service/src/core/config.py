import os
from dotenv import load_dotenv

load_dotenv()

# -- OpenAI model --------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")

# -- XAI Validator identity ----------------------------------------------------
XAI_AGENT_ID = os.getenv("XAI_AGENT_ID", "XAI-VALIDATOR-1003")

# -- ChromaDB (clinical guidelines + shared collections) -----------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8020"))
