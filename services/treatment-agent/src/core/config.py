import os
from dotenv import load_dotenv

load_dotenv()

# -- OpenAI model --------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
