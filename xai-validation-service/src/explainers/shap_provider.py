"""
Diagnosis explainability module.

Current implementation uses LLM-based explanation to identify the top contributing
clinical factors for a diagnosis. This approach requires no pre-trained ML model.

Extension point: when a trained sklearn/XGBoost model becomes available, replace
`DiagnosisExplainer.explain_diagnosis` with the commented SHAP implementation below.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from log.logger import logger

_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical AI explainability specialist.
Given a patient's symptoms and the resulting diagnosis, identify the top 3 clinical
factors that most contributed to the diagnosis decision.

Respond ONLY with valid JSON (no markdown fences):
{{
    "top_factors": [
        {{"factor": "Cardiac arrest possibility", "importance": 0.92, "direction": "increases_risk"}},
        {{"factor": "Infection in the stomach", "importance": 0.78, "direction": "increases_risk"}},
        {{"factor": "Normal ECG baseline", "importance": 0.45, "direction": "decreases_risk"}}
    ]
}}
direction must be one of: increases_risk, decreases_risk, neutral
importance is a float between 0.0 and 1.0"""),
    ("human", "Patient symptoms: {symptoms}\nDiagnosis summary: {diagnosis_summary}"),
])

_explanation_chain = _EXPLANATION_PROMPT | _llm


class DiagnosisExplainer:
    """
    Provides explainability for AI diagnosis decisions via LLM-generated
    feature importance analysis.

    # SHAP extension (uncomment when a trained model is available):
    # from shap import Explainer as SHAPExplainer
    # import pickle, pandas as pd
    # def load_model(self, path): ...
    # def explain_with_shap(self, features: dict): ...
    """

    def explain_diagnosis(self, symptoms: str, diagnosis_summary: str) -> list:
        """Return top 3 contributing clinical factors for the diagnosis."""
        try:
            result = _explanation_chain.invoke({
                "symptoms": symptoms,
                "diagnosis_summary": diagnosis_summary,
            })
            content = result.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            raw = json.loads(content.strip())
            factors = raw.get("top_factors", [])
            logger.debug("Explainability: %d factors identified", len(factors))
            return factors
        except Exception as e:
            logger.warning("Explainer LLM error (non-blocking): %s", str(e))
            return []
