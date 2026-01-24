import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState
from src.tools.cardiology_client import call_cardiology_api

# Initialize the LLM for the Master Router
# This supports autonomous reasoning and independent decision-making
llm = ChatOpenAI(model="gpt-4o", temperature=0)
logger = logging.getLogger(__name__)

async def master_router_node(state: AgentState):
    """
    The Supervisor/Router Agent.
    Analyzes patient data/symptoms and directs flow to the correct medical specialist.
    """
    symptoms = state.get('symptoms', "")
    
    # Prompting the LLM to act as a triage supervisor
    prompt = (
        f"You are the Master Orchestrator for an Agentic Healthcare Framework. "
        f"Analyze these patient symptoms: '{symptoms}'. "
        "Determine the most relevant specialist for diagnosis. "
        "Options: 'cardiology', 'chronic_disease', 'pathology', or 'finish' if data is insufficient. "
        "Respond with only the lower-case specialist name."
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        # Log the decision for transparency
        logger.info(f"Master Router Decision: {decision}")
        
        return {"next_node": decision}
    except Exception as e:
        logger.error(f"Router Decision Failed: {str(e)}")
        return {"next_node": "finish"}

async def cardiology_node(state: AgentState):
    """
    Worker Agent specialized in Cardiology.
    Invokes the remote Cardiology Microservice via the handshake client.
    """
    patient_id = state.get('patient_id')
    symptoms = state.get('symptoms')

    logger.info(f"Invoking Cardiology Microservice for Patient: {patient_id}")

    # Objective 2: Trigger autonomous decision-making in heart diagnosis
    diagnosis_data = await call_cardiology_api(patient_id, symptoms)
    
    diagnosis_text = diagnosis_data.get("diagnosis", "Diagnosis unavailable.")
    
    # Update the global state with the specialist's findings
    return {
        "current_diagnosis": diagnosis_text,
        "messages": state.get('messages', []) + [f"Cardiology Specialist Output: {diagnosis_text}"]
    }

async def xai_validator_node(state: AgentState):
    """
    XAI Revalidation Layer.
    Placeholder for the revalidation logic before final patient care.
    """
    diagnosis = state.get('current_diagnosis', "")
    
    # Placeholder Logic: In next step, this will call the XAI Microservice 
    # It will verify findings against clinical guidelines (RAG)
    is_valid = len(diagnosis) > 10  # Mock validation logic
    
    logger.info(f"XAI Validation Status: {'Passed' if is_valid else 'Failed'}")
    
    return {"is_validated": is_valid}

async def finish_node(state: AgentState):
    """
    Final node to terminate the workflow once revalidation is successful.
    """
    return {"next_node": "END"}