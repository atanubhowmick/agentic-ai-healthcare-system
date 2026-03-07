from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from log.logger import logger

# 1. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Pathology AI Agent. Your goal is to provide diagnostic insights
               by interpreting laboratory test results and identifying abnormalities in biomarkers.
               Always be precise and cite specific lab indicators such as CBC (Complete Blood Count),
               metabolic panels (glucose, creatinine, BUN), liver function tests (ALT, AST, bilirubin),
               tumour markers (PSA, CA-125, CEA), urinalysis findings, and culture results.
               Provide the response strictly in the following JSON format:
               {{
                   "analysisDetails": "Within 200 words",
                   "severity": "LOW/HIGH/CRITICAL",
                   "hospitalizationNeeded": "YES/NO",
                   "clarificationQuestion": "any clarification question you have within 100 words",
                   "emergencyCareNeeded": "YES/NO",
                   "additionalTestsRequired": ["All additional lab tests needed in list"],
                   "imagingRequired": ["Imaging studies needed e.g. CT scan, MRI, Ultrasound, X-ray"],
                   "referralNeeded": "Specialist referral recommendation if any, otherwise NONE"
                }}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. Initialize LLM and chain
logger.debug("Initializing Pathology LLM and chain")
llm = ChatOpenAI(model="gpt-5.2", temperature=0)
pathology_chain = prompt | llm
logger.debug("Pathology chain initialized successfully")

# 3. Session store for chat history
_session_store: dict = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        logger.debug("Creating new chat history for session: %s", session_id)
        _session_store[session_id] = ChatMessageHistory()
    else:
        logger.debug("Reusing existing chat history for session: %s | messages in history: %d",
                     session_id, len(_session_store[session_id].messages))
    return _session_store[session_id]

# 4. Chain with message history
logger.debug("Wrapping chain with RunnableWithMessageHistory")
pathology_executor = RunnableWithMessageHistory(
    pathology_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
logger.debug("Pathology executor ready")
