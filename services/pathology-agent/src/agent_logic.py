import pandas as pd
import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

logger = logging.getLogger(__name__)

class PathologyAnalyst:
    def __init__(self, csv_path: str):
        # Loading the laboratory results dataset as per Section 7.2
        self.df = pd.read_csv(csv_path)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Initializing the Pandas Dataframe Agent
        # This allows the agent to read and query CSV data autonomously
        self.agent = create_pandas_dataframe_agent(
            self.llm, 
            self.df, 
            verbose=True, 
            allow_dangerous_code=True # Required for pandas operations
        )

    async def analyze_lab_reports(self, patient_id: str, query: str):
        """
        Scans through medical records to provide insight into risks.
        """
        full_query = (
            f"Focus on Patient ID {patient_id}. "
            f"Analyze their lab results based on this request: {query}. "
            "Identify any abnormalities in biomarkers (e.g., glucose, creatinine, troponin)."
        )
        
        try:
            response = self.agent.run(full_query)
            return response
        except Exception as e:
            logger.error(f"Pathology Analysis Error: {str(e)}")
            return "Unable to complete lab analysis at this time."
        