from typing import Annotated, List, Union
from typing_extensions import TypedDict
import operator

class AgentState(TypedDict):
    # 'messages' tracks the conversation history and agent outputs
    messages: Annotated[List[str], operator.add]
    # 'next_node' tells the graph which service to call next
    next_node: str
    patient_id: str
    symptoms: str
    current_diagnosis: str
    is_validated: bool