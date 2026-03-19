"""
Neurology Agent - DeepAgent-based implementation.

Architecture:
  - Uses deepagents.create_deep_agent (built on LangGraph) as the executor.
  - @tool decorator exposes the response schema to the agent.
  - SystemMessage / HumanMessage used for explicit message construction.

Public interface (used by neurology_service.py):
  neurology_executor  - the raw DeepAgent instance
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


# -- Tools ----------------------------------------------------------------------

@tool
def get_neurology_response_schema() -> str:
    """Return the required JSON response schema for neurology diagnosis output.
    Call this tool whenever you need a reminder of the exact JSON format expected."""
    return _JSON_SCHEMA


# -- LLM ------------------------------------------------------------------------

logger.debug("Initializing Neurology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# -- DeepAgent ------------------------------------------------------------------

logger.debug("Building Neurology DeepAgent")
neurology_executor = create_deep_agent(
    model=_llm,
    tools=[get_neurology_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Neurology DeepAgent ready")
