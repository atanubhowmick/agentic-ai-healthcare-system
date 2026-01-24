import httpx
import logging
from src.core.config import CARDIOLOGY_SERVICE_URL

logger = logging.getLogger(__name__)

async def call_cardiology_api(patient_id: str, symptoms: str):
    """
    Communicates with the Cardiology Microservice to get a diagnostic prediction.
    Implements the handshake between the Orchestration Layer and Worker Layer.
    """
    payload = {
        "patient_id": patient_id,
        "symptoms": symptoms
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Objective 2: Call the autonomous decision-making framework endpoint
            response = await client.post(
                f"{CARDIOLOGY_SERVICE_URL}/diagnose", 
                json=payload
            )
            response.raise_for_status()
            
            # Returns the diagnosis for the LangGraph state
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Cardiology Service Error: {e.response.text}")
            return {"diagnosis": "Error: Specialist unavailable", "status": "Failed"}
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return {"diagnosis": "Error: Connection Refused", "status": "Failed"}
        