from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from log.logger import logger

# 1. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Cardiology AI Agent. Your goal is to provide diagnostic insights
               based on heart-related symptoms and metrics. Always be very precise and cite specific cardiac
               indicators such as Troponin levels, blood pressure, and ECG patterns. Provide the response in
               the JSON format:
               {{
                   "diagnosysDetails":"Within 200 words",
                   "severity": "LOW/HIGH/CRITICAL",
                   "hospitalizationNeeded": "YES/NO",
                   "clarificationQuestion": "any clarification question you have within 100 words",
                   "emergencyCareNeeded": "YES/NO",
                   "bloodTestsRequired": ["All the blood tests needed in list"],
                   "labTestsRequired": ["Lab test other than blood test like Chest X-ray, ECG, USG etc"],
                   "medication": "Any medication to take? if yes please provide medicine name and dosages"
                }}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. Initialize LLM and chain
logger.debug("Initializing Cardiology LLM and chain")
llm = ChatOpenAI(model="gpt-5.2", temperature=0)
cardiology_chain = prompt | llm
logger.debug("Cardiology chain initialized successfully")

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

# 4. Chain with message history (drop-in replacement for AgentExecutor interface)
logger.debug("Wrapping chain with RunnableWithMessageHistory")
cardiology_executor = RunnableWithMessageHistory(
    cardiology_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
logger.debug("Cardiology executor ready")
