from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Cardiology AI Agent. Your goal is to provide diagnostic insights
               based on heart-related symptoms and metrics. Always be precise and cite specific cardiac
               indicators such as Troponin levels, blood pressure, and ECG patterns."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. Initialize LLM and chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
cardiology_chain = prompt | llm

# 3. Session store for chat history
_session_store: dict = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

# 4. Chain with message history (drop-in replacement for AgentExecutor interface)
cardiology_executor = RunnableWithMessageHistory(
    cardiology_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
