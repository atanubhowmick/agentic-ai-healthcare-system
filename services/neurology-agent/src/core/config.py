import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
AGENT_NAME = "Neurology_Specialist"
AGENT_ID = "NEURO-AGENT-1002"
