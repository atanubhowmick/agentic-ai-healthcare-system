from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import (
    master_router_node, 
    cardiology_node, 
    xai_validator_node, 
    finish_node
)

def create_orchestrator_graph():
    """
    Constructs the Agentic AI Framework architecture.
    Implements Objective 1: Multi-Agent Architecture with LangGraph.
    """
    # Initialize the graph with the shared AgentState
    workflow = StateGraph(AgentState)

    # 1. Add Nodes to the framework 
    workflow.add_node("router", master_router_node)
    workflow.add_node("cardiology", cardiology_node)
    workflow.add_node("xai_validator", xai_validator_node)
    workflow.add_node("finish", finish_node)

    # 2. Define the Entry Point
    workflow.set_entry_point("router")

    # 3. Define Conditional Edges from the Master Router
    # The router decides which specialist microservice to invoke
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "cardiology": "cardiology",
            "chronic_disease": "finish",  # To be implemented
            "pathology": "finish",        # To be implemented
            "finish": "finish"
        }
    )

    # 4. Specialist to XAI Revalidation Handshake
    # Every diagnosis must pass through the XAI layer for ethical check
    workflow.add_edge("cardiology", "xai_validator")

    # 5. The Revised Decision-Making Loop
    # If XAI fails, the system loops back to the router to revisit the data
    workflow.add_conditional_edges(
        "xai_validator",
        lambda state: "finish" if state.get("is_validated") else "router",
        {
            "finish": "finish",
            "router": "router"  # The feedback loop for autonomous correction
        }
    )

    # 6. Termination
    workflow.add_edge("finish", END)

    return workflow.compile()