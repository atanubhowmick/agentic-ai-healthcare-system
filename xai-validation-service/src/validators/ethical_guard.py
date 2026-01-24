from langchain_openai import ChatOpenAI
from langchain.chains import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the "Ethical Constitution" for healthcare
healthcare_principle = ConstitutionalPrinciple(
    name="Clinical Safety Principle",
    critique_request="Identify if the diagnosis or treatment recommendation violates standard medical safety or is biased.",
    revision_request="Rewrite the recommendation to prioritize patient safety and clinical evidence."
)

def validate_ethics(diagnosis: str, symptoms: str):
    """
    Verifies the diagnosis against ethical boundaries[cite: 368].
    Acts as a 'safety node' in the workflow[cite: 380].
    """
    # Simple logic to check if a critique is needed
    if "error" in diagnosis.lower():
        return False, "Diagnosis contains errors."
        
    # In a full implementation, you would use ConstitutionalChain here
    # to evaluate the text against the healthcare_principle.
    return True, "Passed ethical validation."