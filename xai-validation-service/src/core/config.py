import os
from dotenv import load_dotenv

load_dotenv()

# -- OpenAI model --------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")

# -- XAI Validator identity ----------------------------------------------------
XAI_AGENT_ID = os.getenv("XAI_AGENT_ID", "XAI-VALIDATOR-1003")
