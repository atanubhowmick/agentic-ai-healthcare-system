from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


_JSON_SCHEMA = """
{
    "diagnosisDetails": "Detailed neurological assessment within 200 words",
    "severity": "LOW/HIGH/CRITICAL",
    "hospitalizationNeeded": "YES/NO",
    "emergencyCareNeeded": "YES/NO",
    "clarificationQuestion": "Any clarification question within 100 words",
    "bloodTestsRequired": ["All blood tests needed e.g. CBC, metabolic panel, thyroid function"],
    "imagingRequired": ["MRI brain/spine, CT scan, PET scan or other imaging required"],
    "neurologicalTestsRequired": ["EEG, nerve conduction study, lumbar puncture, EMG etc"],
    "medication": "Medication name and dosages if applicable, otherwise NONE"
}"""

BASE_SYSTEM = (
    "You are a specialized Neurology AI Agent. Your goal is to provide diagnostic insights "
    "based on neurological symptoms and clinical indicators. Always be very precise and cite "
    "specific neurological markers such as reflexes, cognitive assessments, CSF analysis, "
    "EEG patterns and MRI/CT findings. "
    "Provide the response strictly in the following JSON format:" + _JSON_SCHEMA
)


@tool
def get_neurology_response_schema() -> str:
    """Returns the expected JSON response schema for a neurology diagnosis."""
    return _JSON_SCHEMA


logger.debug("Initializing Neurology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

logger.debug("Building Neurology DeepAgent")
neurology_executor = create_deep_agent(
    model=_llm,
    tools=[get_neurology_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Neurology DeepAgent ready")
