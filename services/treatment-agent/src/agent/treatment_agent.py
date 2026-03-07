from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from log.logger import logger

# 1. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Treatment & Patient Care AI Agent. Your goal is to synthesize
               specialist diagnostic findings into a comprehensive, evidence-based treatment and care plan.
               Always cite standard clinical protocols, specify medications with exact dosages and frequencies,
               and clearly define urgency and follow-up requirements.
               Provide the response strictly in the following JSON format:
               {{
                   "treatmentPlan": "Detailed treatment approach within 200 words",
                   "medications": ["drug name - dosage - frequency"],
                   "followUpRequired": "YES/NO",
                   "followUpTimeframe": "e.g. 3 days / 1 week / 2 weeks / 1 month / 3 months / NONE",
                   "lifestyleRecommendations": ["Diet change or restriction", "Exercise guidance", "Stress management"],
                   "monitoringRequired": ["Parameter to track e.g. Blood pressure daily", "Troponin levels at 6 hours"],
                   "referralRequired": "Specialist referral recommendation if needed, otherwise NONE",
                   "urgency": "IMMEDIATE/SOON/ROUTINE"
               }}
               Urgency guide: IMMEDIATE = requires care within hours, SOON = within days, ROUTINE = weeks/scheduled."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. Initialize LLM and chain
logger.debug("Initializing Treatment LLM and chain")
llm = ChatOpenAI(model="gpt-5.2", temperature=0)
treatment_chain = prompt | llm
logger.debug("Treatment chain initialized successfully")

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
treatment_executor = RunnableWithMessageHistory(
    treatment_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
logger.debug("Treatment executor ready")
