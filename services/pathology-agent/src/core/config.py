import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
AGENT_NAME = "Pathology_Specialist"
AGENT_ID = "PATHO-AGENT-1002"
