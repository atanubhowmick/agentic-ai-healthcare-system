from langchain.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from core.config import OPENAI_MODEL, MIMIC_SIMILARITY_THRESHOLD, MIMIC_PARTIAL_THRESHOLD
from rag.mimic_retriever import search_similar_cases, is_collection_populated
from log.logger import logger


_JSON_SCHEMA = """
{
    "diagnosisDetails": "Detailed oncology assessment within 200 words",
    "stage": "TNM staging if determinable e.g. Stage II (T2N1M0) or Undetermined",
    "clarificationQuestion": "Any clarification question within 100 words",
    "biomarkersRequired": ["Tumor markers and genetic tests e.g. CEA, CA-125, BRCA1/2, EGFR mutation"],
    "imagingRequired": ["Imaging studies e.g. CT chest/abdomen/pelvis, PET-CT, MRI brain, Bone scan"],
    "biopsyRequired": "Type of biopsy if needed e.g. Core needle biopsy of lung mass, otherwise NOT REQUIRED",
    "oncologyReferralNeeded": "Referral recommendation e.g. Medical Oncology + Radiation Oncology, otherwise NONE",
    "medication": "Initial medication or supportive care e.g. pain management, anti-emetics, or targeted therapy if diagnosis confirmed"
}"""

BASE_SYSTEM = (
    "You are a specialized Oncology AI Agent. Your goal is to provide diagnostic insights "
    "based on cancer-related symptoms, lab findings, and clinical indicators. Always be precise "
    "and cite specific oncological markers such as tumor biomarkers (PSA, CA-125, CEA, AFP, "
    "CA 19-9, BRCA1/2), TNM staging criteria, imaging characteristics, biopsy findings, and "
    "relevant genetic mutations (EGFR, ALK, KRAS, HER2). "
    "When assessing a new patient query, use the search_mimic_cases tool to retrieve "
    "semantically similar historical oncology cases from the MIMIC-IV database and use them "
    "as evidence-based reference to improve diagnostic accuracy. "
    "NOTE: severity, suspectedCancerType, hospitalizationNeeded and emergencyCareNeeded "
    "are determined separately — do NOT include them in your response. "
    "Provide the final response strictly in the following JSON format:" + _JSON_SCHEMA
)


@tool
def get_oncology_response_schema() -> str:
    """Returns the expected JSON response schema for an oncology diagnosis."""
    return _JSON_SCHEMA


@tool
def search_mimic_cases(symptoms: str, top_k: int = 3) -> str:
    """Search the MIMIC-IV clinical database for historical oncology cases semantically
    similar to the given symptoms. Returns a formatted summary of the most relevant cases
    to use as evidence-based diagnostic context. Call this for all new (non-follow-up) queries.

    Args:
        symptoms: Patient symptoms or clinical presentation text to search against.
        top_k: Number of similar cases to retrieve (default 3).
    """
    if not is_collection_populated():
        logger.debug("MIMIC collection unavailable - skipping RAG search")
        return "MIMIC-IV database is not available. Proceed with LLM-only diagnosis."

    cases = search_similar_cases(symptoms, top_k=top_k)
    if not cases:
        return "No similar cases found in MIMIC-IV. Proceed with LLM-only diagnosis."

    lines = []
    for i, c in enumerate(cases, start=1):
        score = c["similarity_score"]
        confidence = (
            "HIGH confidence match" if score >= MIMIC_SIMILARITY_THRESHOLD else
            "LOW confidence match"  if score >= MIMIC_PARTIAL_THRESHOLD    else
            "weak match"
        )
        lines.append(
            f"Case {i} ({confidence}, similarity={score}):\n"
            f"  Cancer Type       : {c['cancer_type']}\n"
            f"  ICD-10 Codes      : {c['icd_codes']}\n"
            f"  Chief Complaint   : {c['chief_complaint']}\n"
            f"  Treatment Summary : {c['treatment_summary']}\n"
            f"  Severity          : {c['severity']}"
        )

    logger.info("[MIMIC_TOOL] %d case(s) retrieved | top score: %.4f", len(cases), cases[0]["similarity_score"])
    return "\n\n".join(lines)


logger.debug("Initializing Cancer/Oncology LLM | model: %s", OPENAI_MODEL)
_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

logger.debug("Building Cancer DeepAgent")
cancer_executor = create_deep_agent(
    model=_llm,
    tools=[get_oncology_response_schema, search_mimic_cases],
    system_prompt=BASE_SYSTEM,
)
logger.debug("Cancer DeepAgent ready")
