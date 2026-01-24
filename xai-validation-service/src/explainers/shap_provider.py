import shap
import pickle
import pandas as pd

class HeartSHAPExplainer:
    def __init__(self, model_path: str):
        # Load the same model used by the Cardiology Agent
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        # Initialize SHAP explainer
        self.explainer = shap.Explainer(self.model)

    def explain_diagnosis(self, patient_features: dict):
        """
        Calculates feature importance using SHAP[cite: 374].
        Identifies which features (e.g., BP, Troponin) influenced the decision[cite: 374].
        """
        df = pd.DataFrame([patient_features])
        shap_values = self.explainer(df)
        
        # Extract top 3 contributing features
        feature_importance = dict(zip(df.columns, shap_values.values[0]))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_features[:3]
    