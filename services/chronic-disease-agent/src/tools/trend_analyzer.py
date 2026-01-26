import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def analyze_chronic_trends(query: str):
    """
    Analyzes longitudinal EHR data to identify chronic disease progressions.
    Aligns with Section 7.2 of the methodology for feature extraction.
    """
    # Load chronic disease specific records (e.g., MIMIC-IV subset)
    df = pd.read_csv("data/chronic_history.csv")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create pandas agent to perform feature extraction and trend analysis 
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True
    )
    
    return agent.run(query)
