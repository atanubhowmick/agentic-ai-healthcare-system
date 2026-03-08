from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from log.logger import logger

# 1. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Oncology AI Agent. Your goal is to provide diagnostic insights
               based on cancer-related symptoms, lab findings, and clinical indicators. Always be precise
               and cite specific oncological markers such as tumor biomarkers (PSA, CA-125, CEA, AFP,
               CA 19-9, BRCA1/2), TNM staging criteria, imaging characteristics, biopsy findings, and
               relevant genetic mutations (EGFR, ALK, KRAS, HER2). Provide the response strictly in the
               following JSON format:
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
               }}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. Initialize LLM and chain
logger.debug("Initializing Cancer/Oncology LLM and chain")
llm = ChatOpenAI(model="gpt-5.2", temperature=0)
cancer_chain = prompt | llm
logger.debug("Cancer chain initialized successfully")

# 3. Session store for chat history
_session_store: dict = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        logger.debug("Creating new chat history for session: %s", session_id)
        _session_store[session_id] = ChatMessageHistory()
    else:
        logger.debug(
            "Reusing existing chat history for session: %s | messages in history: %d",
            session_id, len(_session_store[session_id].messages),
        )
    return _session_store[session_id]


# 4. Chain with message history
logger.debug("Wrapping chain with RunnableWithMessageHistory")
cancer_executor = RunnableWithMessageHistory(
    cancer_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
logger.debug("Cancer executor ready")
