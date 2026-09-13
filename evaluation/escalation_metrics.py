import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List

class EscalationMetrics:
    """
    Evaluates safety escalation decisions (AUTO_HANDLE vs ESCALATE).
    Specifically tracks False Auto-Handle (missed escalation) as a critical safety metric.
    """
    
    @staticmethod
    def evaluate(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        labels = ["AUTO_HANDLE", "ESCALATE"]
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate for ESCALATE class as positive
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label="ESCALATE", average="binary", zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # cm structure:
        # [ [True AutoHandle, False Escalate (Over-escalation)],
        #   [False AutoHandle (Missed Escalation!), True Escalate] ]
        tn = int(cm[0][0])
        fp = int(cm[0][1])
        fn = int(cm[1][0]) # False Auto-Handle
        tp = int(cm[1][1])
        
        false_auto_handle_rate = round(fn / max(fn + tp, 1), 4)
        over_escalation_rate = round(fp / max(tn + fp, 1), 4)
        
        return {
            "accuracy": round(float(acc), 4),
            "escalate_precision": round(float(p), 4),
            "escalate_recall": round(float(r), 4),
            "escalate_f1": round(float(f1), 4),
            "true_auto_handle": tn,
            "over_escalations_fp": fp,
            "false_auto_handles_fn": fn,
            "true_escalations_tp": tp,
            "false_auto_handle_rate": false_auto_handle_rate,
            "over_escalation_rate": over_escalation_rate,
            "confusion_matrix": cm.tolist()
        }

    @staticmethod
    def format_table(results: Dict[str, Any], title: str = "Escalation Policy Evaluation") -> str:
        return f"""### {title}
- **Accuracy**: {results['accuracy']:.4f}
- **Escalation Precision**: {results['escalate_precision']:.4f}
- **Escalation Recall**: {results['escalate_recall']:.4f}
- **Escalation F1**: {results['escalate_f1']:.4f}
- **False Auto-Handles (Missed Escalations - CRITICAL SAFETY RISK)**: {results['false_auto_handles_fn']} ({results['false_auto_handle_rate']*100:.1f}%)
- **Over-Escalations (Safe but higher agent load)**: {results['over_escalations_fp']} ({results['over_escalation_rate']*100:.1f}%)

| Actual \ Predicted | Predicted AUTO_HANDLE | Predicted ESCALATE |
|---|---|---|
| **Actual AUTO_HANDLE** | {results['true_auto_handle']} (True Neg) | {results['over_escalations_fp']} (False Pos) |
| **Actual ESCALATE** | **{results['false_auto_handles_fn']} (False Neg)** | {results['true_escalations_tp']} (True Pos) |
"""
