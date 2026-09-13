import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Any, List

class IntentMetrics:
    """
    Standard evaluation metrics for multi-class intent classification:
    Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1,
    and Per-Class breakdown.
    """
    
    @staticmethod
    def evaluate(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict[str, Any]:
        if labels is None:
            labels = sorted(list(set(y_true).union(set(y_pred))))
            
        acc = accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        
        per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        per_class_dict = {}
        for i, lbl in enumerate(labels):
            per_class_dict[lbl] = {
                "precision": round(float(per_class_p[i]), 4),
                "recall": round(float(per_class_r[i]), 4),
                "f1": round(float(per_class_f1[i]), 4),
                "support": int(per_class_supp[i])
            }
            
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "labels": labels
        }

    @staticmethod
    def format_table(results: Dict[str, Any], title: str = "Intent Classification Results") -> str:
        lines = [
            f"### {title}",
            "",
            f"- **Accuracy**: {results['accuracy']:.4f}",
            f"- **Macro Precision**: {results['macro_precision']:.4f}",
            f"- **Macro Recall**: {results['macro_recall']:.4f}",
            f"- **Macro F1**: {results['macro_f1']:.4f}",
            f"- **Weighted F1**: {results['weighted_f1']:.4f}",
            "",
            "| Intent Class | Precision | Recall | F1 Score | Support |",
            "|---|---|---|---|---|"
        ]
        for lbl, stats in results['per_class'].items():
            lines.append(f"| `{lbl}` | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {stats['support']} |")
            
        return "\n".join(lines)
