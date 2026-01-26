from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from src.tools.trend_analyzer import analyze_chronic_trends

# Define tools specialized for chronic condition monitoring
tools = [
    Tool(
        name="Chronic_Trend_Analyzer",
        func=analyze_chronic_trends,
        description="Analyzes long-term health trends, glucose levels, and hypertension history."
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized Chronic Disease AI Agent. "
               "Focus on identifying long-term health risks and disease progression."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
chronic_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
