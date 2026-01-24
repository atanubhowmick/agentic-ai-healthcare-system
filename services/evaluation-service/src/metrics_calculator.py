import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, accuracy_score

class AgentEvaluator:
    """
    Evaluates Agent performance using standard ML metrics. [cite: 211, 408]
    """
    @staticmethod
    def calculate_agent_metrics(y_true, y_scores, y_pred):
        # 1. ROC-AUC: Robust measure of discriminative ability [cite: 413]
        roc_auc = roc_auc_score(y_true, y_scores)
        
        # 2. PR-AUC: Evaluates model with imbalanced datasets [cite: 420]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # 3. F1 Score: Balanced precision and recall [cite: 426]
        f1 = f1_score(y_true, y_pred)
        
        # 4. Accuracy: Correctly predicted outcomes [cite: 423]
        accuracy = accuracy_score(y_true, y_pred)

        # Expected value checks as per Section 7.7.1
        return {
            "roc_auc": roc_auc, # Expected > 0.8 [cite: 419]
            "pr_auc": pr_auc,   # Value > 0.8 is balanced [cite: 422]
            "f1_score": f1,     # > 0.85 is considered good [cite: 428]
            "accuracy": accuracy
        }
        