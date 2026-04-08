from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


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


@tool
def get_treatment_response_schema() -> str:
    """Returns the expected JSON response schema for a treatment plan."""
    return _JSON_SCHEMA


logger.debug("Initializing Treatment LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

logger.debug("Building Treatment DeepAgent")
treatment_executor = create_deep_agent(
    model=_llm,
    tools=[get_treatment_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Treatment DeepAgent ready")
