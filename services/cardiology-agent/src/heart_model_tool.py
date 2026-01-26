import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

def get_heart_data_analysis(query: str):
    """Analyzes heart-related EHR data using a Pandas Agent."""
    # Load heart-specific dataset
    df = pd.read_csv("data/heart_disease_data.csv") 
    
    llm = ChatOpenAI(model = "gpt-4o", temperature = 0)
    
    # Create the specialized pandas agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    
    return agent.run(query)
