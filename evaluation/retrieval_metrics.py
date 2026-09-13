import numpy as np
import pandas as pd
from typing import List, Dict, Any

class RetrievalMetrics:
    """
    Evaluates historical dialogue retrieval relevance and ranking quality.
    Measures semantic alignment, Top-k intent hit rate, and MRR.
    """
    
    @staticmethod
    def evaluate_retrieval(
        queries: List[str],
        true_intents: List[str],
        retriever,
        classifier,
        k_list: List[int] = [1, 3, 5]
    ) -> Dict[str, Any]:
        max_k = max(k_list)
        hits_at_k = {k: 0 for k in k_list}
        reciprocal_ranks = []
        similarity_scores = []
        
        n_queries = len(queries)
        
        for q, true_intent in zip(queries, true_intents):
            results = retriever.retrieve(q, top_k=max_k)
            if not results:
                reciprocal_ranks.append(0.0)
                continue
                
            similarity_scores.append(results[0]['score'])
            
            # Check intent match of retrieved historical items
            # Predict intent of retrieved historical items
            first_hit_rank = None
            for r in results:
                # Classify retrieved historical customer issue
                retrieved_pred = classifier.predict(r['historical_customer_issue'])['intent']
                if retrieved_pred == true_intent:
                    if first_hit_rank is None:
                        first_hit_rank = r['rank']
                    for k in k_list:
                        if r['rank'] <= k:
                            hits_at_k[k] += 1
                    break
                    
            if first_hit_rank:
                reciprocal_ranks.append(1.0 / first_hit_rank)
            else:
                reciprocal_ranks.append(0.0)
                
        metrics = {
            "num_evaluated": n_queries,
            "avg_top1_similarity": round(float(np.mean(similarity_scores)), 4) if similarity_scores else 0.0,
            "mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
        }
        for k in k_list:
            metrics[f"recall_at_{k}"] = round(hits_at_k[k] / max(n_queries, 1), 4)
            
        return metrics

    @staticmethod
    def format_table(results: Dict[str, Any], title: str = "Historical Retrieval Performance") -> str:
        lines = [
            f"### {title}",
            "",
            f"- **Evaluated Queries**: {results['num_evaluated']}",
            f"- **Average Top-1 Cosine Similarity**: {results['avg_top1_similarity']:.4f}",
            f"- **Mean Reciprocal Rank (MRR)**: {results['mrr']:.4f}",
            f"- **Recall@1**: {results.get('recall_at_1', 0):.4f}",
            f"- **Recall@3**: {results.get('recall_at_3', 0):.4f}",
            f"- **Recall@5**: {results.get('recall_at_5', 0):.4f}"
        ]
        return "\n".join(lines)
