"""
Cancer Oncology Agent - LangChain chains.

Two chains are exposed:
  cancer_executor      - standard LLM-only chain (fallback)
  cancer_rag_executor  - RAG-augmented chain (MIMIC-IV context injected at call time)

The service layer (cancer_service.py) decides which to use based on MIMIC retrieval results.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from log.logger import logger


# -- JSON response schema (shared by both chains) ------------------------------

_JSON_SCHEMA = """
{{
    "diagnosisDetails": "Detailed oncology assessment within 200 words",
    "suspectedCancerType": "Suspected cancer type e.g. Lung Adenocarcinoma / Breast Cancer HER2+ / Colorectal Cancer / Unknown",
    "stage": "TNM staging if determinable e.g. Stage II (T2N1M0) or Undetermined",
    "severity": "LOW/HIGH/CRITICAL",
    "hospitalizationNeeded": "YES/NO",
    "emergencyCareNeeded": "YES/NO",
    "clarificationQuestion": "Any clarification question within 100 words",
    "biomarkersRequired": ["Tumor markers and genetic tests e.g. CEA, CA-125, BRCA1/2, EGFR mutation"],
    "imagingRequired": ["Imaging studies e.g. CT chest/abdomen/pelvis, PET-CT, MRI brain, Bone scan"],
    "biopsyRequired": "Type of biopsy if needed e.g. Core needle biopsy of lung mass, otherwise NOT REQUIRED",
    "oncologyReferralNeeded": "Referral recommendation e.g. Medical Oncology + Radiation Oncology, otherwise NONE",
    "medication": "Initial medication or supportive care e.g. pain management, anti-emetics, or targeted therapy if diagnosis confirmed"
}}"""

_BASE_SYSTEM = (
    "You are a specialized Oncology AI Agent. Your goal is to provide diagnostic insights "
    "based on cancer-related symptoms, lab findings, and clinical indicators. Always be precise "
    "and cite specific oncological markers such as tumor biomarkers (PSA, CA-125, CEA, AFP, "
    "CA 19-9, BRCA1/2), TNM staging criteria, imaging characteristics, biopsy findings, and "
    "relevant genetic mutations (EGFR, ALK, KRAS, HER2). Provide the response strictly in the "
    "following JSON format:" + _JSON_SCHEMA
)

# -- Chain 1: Standard LLM-only (fallback when MIMIC has no relevant cases) ---

_standard_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# -- Chain 2: RAG-augmented (MIMIC-IV context injected via {mimic_context}) ---
# The {mimic_context} placeholder is filled by cancer_service.py at call time.

_rag_system = (
    _BASE_SYSTEM
    + "\n\n"
    "The following are real oncology cases retrieved from the MIMIC-IV clinical database "
    "that are semantically similar to this patient's presentation. Use them as evidence-based "
    "reference to improve diagnostic accuracy, but always reason from the specific symptoms "
    "provided:\n\n{mimic_context}"
)

_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", _rag_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# -- LLM -----------------------------------------------------------------------

logger.debug("Initializing Cancer/Oncology LLM")
llm = ChatOpenAI(model="gpt-5.2", temperature=0)

_standard_chain = _standard_prompt | llm
_rag_chain      = _rag_prompt      | llm
logger.debug("Cancer chains initialised")

# -- Session store -------------------------------------------------------------

_session_store: dict = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        logger.debug("Creating new chat history for session: %s", session_id)
        _session_store[session_id] = ChatMessageHistory()
    else:
        logger.debug(
            "Reusing chat history for session: %s | messages: %d",
            session_id, len(_session_store[session_id].messages),
        )
    return _session_store[session_id]


# -- Public executors ----------------------------------------------------------

# Standard fallback - invoked when MIMIC returns no relevant cases
cancer_executor = RunnableWithMessageHistory(
    _standard_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# RAG-augmented - invoked when MIMIC returns relevant cases
# Caller must pass mimic_context in the input dict alongside "input"
cancer_rag_executor = RunnableWithMessageHistory(
    _rag_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

logger.debug("Cancer executors ready (standard + RAG)")
