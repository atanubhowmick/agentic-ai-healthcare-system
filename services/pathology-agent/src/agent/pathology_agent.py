from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL
from log.logger import logger


_JSON_SCHEMA = """
{
    "analysisDetails": "Detailed pathology analysis within 200 words",
    "severity": "LOW/HIGH/CRITICAL",
    "hospitalizationNeeded": "YES/NO",
    "emergencyCareNeeded": "YES/NO",
    "clarificationQuestion": "Any clarification question within 100 words",
    "additionalTestsRequired": ["All additional lab tests needed e.g. biopsy, culture, PCR"],
    "imagingRequired": ["Imaging studies needed e.g. CT scan, MRI, Ultrasound, X-ray"],
    "referralNeeded": "Specialist referral recommendation if any, otherwise NONE"
}"""

BASE_SYSTEM = (
    "You are a specialized Pathology AI Agent. Your goal is to provide diagnostic insights "
    "by interpreting laboratory test results and identifying abnormalities in biomarkers. "
    "Always be precise and cite specific lab indicators such as CBC (Complete Blood Count), "
    "metabolic panels (glucose, creatinine, BUN), liver function tests (ALT, AST, bilirubin), "
    "tumour markers (PSA, CA-125, CEA), urinalysis findings, and culture results. "
    "Provide the response strictly in the following JSON format:" + _JSON_SCHEMA
)


@tool
def get_pathology_response_schema() -> str:
    """Returns the expected JSON response schema for a pathology analysis."""
    return _JSON_SCHEMA


logger.debug("Initializing Pathology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

logger.debug("Building Pathology DeepAgent")
pathology_executor = create_deep_agent(
    model=_llm,
    tools=[get_pathology_response_schema],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Pathology DeepAgent ready")
