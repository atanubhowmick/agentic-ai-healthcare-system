from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def get_heart_data_analysis(query: str) -> str:
    """Analyzes heart-related symptoms and metrics using LLM medical reasoning."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a specialist cardiology analyst. "
         "Analyze the provided heart-related symptoms or metrics and give a precise clinical assessment. "
         "Cite specific cardiac indicators such as Troponin levels, blood pressure, ECG patterns, "
         "and other relevant biomarkers in your response."),
        ("human", "{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})

    return response.content
