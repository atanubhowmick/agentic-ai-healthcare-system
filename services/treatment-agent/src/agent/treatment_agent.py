"""
Treatment & Patient Care Agent - DeepAgent-based implementation.

Architecture:
  - Uses deepagents.create_deep_agent (built on LangGraph) as the executor.
  - @tool decorator exposes the response schema to the agent.
  - Session history and message construction are handled by treatment_service.py.

Public interface (used by treatment_service.py):
  treatment_executor  - the raw DeepAgent instance
  BASE_SYSTEM         - system prompt (used by service to build messages)
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


# -- JSON response schema -------------------------------------------------------

_JSON_SCHEMA = """
{
    "treatmentPlan": "Detailed treatment approach within 200 words",
    "medications": ["drug name - dosage - frequency"],
    "followUpRequired": "YES/NO",
    "followUpTimeframe": "e.g. 3 days / 1 week / 2 weeks / 1 month / 3 months / NONE",
    "lifestyleRecommendations": ["Diet change or restriction", "Exercise guidance", "Stress management"],
    "monitoringRequired": ["Parameter to track e.g. Blood pressure daily", "Troponin levels at 6 hours"],
    "referralRequired": "Specialist referral recommendation if needed, otherwise NONE",
    "urgency": "IMMEDIATE/SOON/ROUTINE"
}"""

BASE_SYSTEM = (
    "You are a specialized Treatment & Patient Care AI Agent. Your goal is to synthesize "
    "specialist diagnostic findings into a comprehensive, evidence-based treatment and care plan. "
    "Always cite standard clinical protocols, specify medications with exact dosages and frequencies, "
    "and clearly define urgency and follow-up requirements. "
    "Urgency guide: IMMEDIATE = requires care within hours, SOON = within days, ROUTINE = weeks/scheduled. "
    "Provide the response strictly in the following JSON format:" + _JSON_SCHEMA
)


# -- Tools ----------------------------------------------------------------------

@tool
def get_treatment_response_schema() -> str:
    """Return the required JSON response schema for treatment plan output.
    Call this tool whenever you need a reminder of the exact JSON format expected."""
    return _JSON_SCHEMA


# -- LLM ------------------------------------------------------------------------

logger.debug("Initializing Treatment LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# -- DeepAgent ------------------------------------------------------------------

logger.debug("Building Treatment DeepAgent")
treatment_executor = create_deep_agent(
    model=_llm,
    tools=[get_treatment_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Treatment DeepAgent ready")
