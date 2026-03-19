"""
Cardiology Agent - DeepAgent-based implementation.

Architecture:
  - Uses deepagents.create_deep_agent (built on LangGraph) as the executor.
  - @tool decorator exposes the response schema to the agent.
  - SystemMessage / HumanMessage used for explicit message construction.

Public interface (used by cardiology_service.py):
  cardiology_executor  - the raw DeepAgent instance
  BASE_SYSTEM          - system prompt (used by service to build messages)
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


# -- JSON response schema -------------------------------------------------------

_JSON_SCHEMA = """
{
    "diagnosysDetails": "Detailed cardiac assessment within 200 words",
    "severity": "LOW/HIGH/CRITICAL",
    "hospitalizationNeeded": "YES/NO",
    "emergencyCareNeeded": "YES/NO",
    "clarificationQuestion": "Any clarification question within 100 words",
    "bloodTestsRequired": ["All blood tests needed e.g. Troponin, BNP, lipid panel, CBC"],
    "labTestsRequired": ["Lab tests other than blood e.g. Chest X-ray, ECG, Echocardiogram, USG"],
    "medication": "Medication name and dosages if applicable, otherwise NONE"
}"""

BASE_SYSTEM = (
    "You are a specialized Cardiology AI Agent. Your goal is to provide diagnostic insights "
    "based on heart-related symptoms and metrics. Always be very precise and cite specific "
    "cardiac indicators such as Troponin levels, BNP/NT-proBNP, blood pressure readings, "
    "ECG patterns (ST elevation, QRS widening, arrhythmias), ejection fraction, and "
    "echocardiographic findings. "
    "Provide the response strictly in the following JSON format:" + _JSON_SCHEMA
)


# -- Tools ----------------------------------------------------------------------

@tool
def get_cardiology_response_schema() -> str:
    """Return the required JSON response schema for cardiology diagnosis output.
    Call this tool whenever you need a reminder of the exact JSON format expected."""
    return _JSON_SCHEMA


# -- LLM ------------------------------------------------------------------------

logger.debug("Initializing Cardiology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# -- DeepAgent ------------------------------------------------------------------

logger.debug("Building Cardiology DeepAgent")
cardiology_executor = create_deep_agent(
    model=_llm,
    tools=[get_cardiology_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Cardiology DeepAgent ready")
