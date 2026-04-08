import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, accuracy_score


class AgentEvaluator:
    """Standard ML metrics for binary classification tasks."""

    @staticmethod
    def calculate_agent_metrics(y_true, y_scores, y_pred):
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "roc_auc":  roc_auc,
            "pr_auc":   pr_auc,
            "f1_score": f1,
            "accuracy": accuracy,
        }
