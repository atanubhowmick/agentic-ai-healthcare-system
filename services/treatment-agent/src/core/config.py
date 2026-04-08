import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
AGENT_NAME = "Treatment_Care_Agent"
AGENT_ID = "TREAT-AGENT-1004"
