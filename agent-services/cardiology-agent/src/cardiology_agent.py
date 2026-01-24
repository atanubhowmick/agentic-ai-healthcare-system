from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from heart_model_tool import get_heart_data_analysis

# 1. Define the Tools for the Cardiologist
tools = [
    Tool(
        name="Heart_Data_Analyzer",
        func=get_heart_data_analysis,
        description="Useful for querying patient heart metrics like Troponin, BP, and ECG trends from records."
    )
]

# 2. Define the Specialist Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized Cardiology AI Agent. Your goal is to provide diagnostic insights "
               "based on heart-related data. Always be precise and cite the metrics used."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. Initialize Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
cardiology_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
