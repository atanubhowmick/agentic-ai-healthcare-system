from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


_JSON_SCHEMA = """
{
    "diagnosisDetails": "Detailed cardiac assessment within 200 words",
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


@tool
def get_cardiology_response_schema() -> str:
    """Returns the expected JSON response schema for a cardiology diagnosis."""
    return _JSON_SCHEMA


logger.debug("Initializing Cardiology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

logger.debug("Building Cardiology DeepAgent")
cardiology_executor = create_deep_agent(
    model=_llm,
    tools=[get_cardiology_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Cardiology DeepAgent ready")
