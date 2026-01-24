import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class ClinicalRAG:
    def __init__(self, database_path: str):
        # Using OpenAI Embeddings for the clinical text
        self.embeddings = OpenAIEmbeddings()
        # Load the pre-indexed clinical guideline vector store
        # This contains guidelines like medication dosages and contraindications
        self.vector_store = FAISS.load_local(
            database_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature = 0)

    def get_treatment_guidelines(self, diagnosis: str):
        """
        Retrieves evidence-based guidelines for a specific diagnosis.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        query = f"Provide standard treatment and medication guidelines for: {diagnosis}"
        return qa_chain.run(query)
    