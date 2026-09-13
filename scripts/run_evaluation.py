import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import time
import pandas as pd
from evaluation.intent_metrics import IntentMetrics
from evaluation.baselines import MajorityBaselineClassifier, TfidfLogisticBaselineClassifier
from evaluation.escalation_metrics import EscalationMetrics
from evaluation.retrieval_metrics import RetrievalMetrics
from evaluation.judge_agreement import evaluate_human_judge_agreement
from src.intent_classifier import SentenceTransformerIntentClassifier
from src.retriever import HistoricalSupportRetriever
from src.escalation import EscalationPolicy
from src.weak_labeler import RuleBasedIntentLabeler

RESULTS_FILE = os.path.join("experiments", "evaluation_results.json")

def run_complete_evaluation():
    print("="*70)
    print("     HIVER CUSTOMER SUPPORT AGENT — AUTOMATED EVALUATION HARNESS")
    print("="*70)
    
    t0 = time.time()
    
    # 1. Load Golden Set
    gold_df = pd.read_csv("data/golden_set.csv")
    X_gold = gold_df['message'].tolist()
    y_gold_intent = gold_df['intent'].tolist()
    y_gold_action = gold_df['expected_action'].tolist()
    labels = sorted(list(set(y_gold_intent)))
    print(f"Loaded {len(gold_df)} frozen golden evaluation records.")
    
    # 2. Load and train baselines
    train_df = pd.read_parquet("data/sample/train.parquet")
    train_df['intent'], train_df['conf'] = zip(*train_df['customer_text_clean'].apply(RuleBasedIntentLabeler.predict_intent))
    confident_train = train_df[train_df['conf'] >= 0.60].copy()
    X_train = confident_train['customer_text_clean'].tolist()
    y_train = confident_train['intent'].tolist()
    
    print("\n[1/5] Evaluating Baseline 1: Majority Class...")
    maj = MajorityBaselineClassifier()
    maj.fit(y_train)
    maj_preds = maj.predict(X_gold)
    maj_metrics = IntentMetrics.evaluate(y_gold_intent, maj_preds, labels=labels)
    
    print("\n[2/5] Evaluating Baseline 2: TF-IDF + Logistic Regression...")
    tfidf = TfidfLogisticBaselineClassifier()
    tfidf.fit(X_train, y_train)
    tfidf_preds = tfidf.predict(X_gold)
    tfidf_metrics = IntentMetrics.evaluate(y_gold_intent, tfidf_preds, labels=labels)
    
    print("\n[3/5] Evaluating Main Model: Sentence Transformers + Classifier...")
    main_model = SentenceTransformerIntentClassifier.load()
    main_preds = main_model.predict_batch(X_gold)
    main_metrics = IntentMetrics.evaluate(y_gold_intent, main_preds, labels=labels)
    
    print("\n[4/5] Evaluating Historical Retrieval Engine (FAISS)...")
    retriever = HistoricalSupportRetriever.load()
    retrieval_metrics = RetrievalMetrics.evaluate_retrieval(
        queries=X_gold,
        true_intents=y_gold_intent,
        retriever=retriever,
        classifier=main_model,
        k_list=[1, 3, 5]
    )
    
    print("\n[5/5] Evaluating Escalation Policy Engine...")
    escalation_policy = EscalationPolicy()
    esc_preds = []
    for _, row in gold_df.iterrows():
        eval_res = escalation_policy.evaluate(
            message=row['message'],
            intent=row['intent'],
            intent_confidence=0.85
        )
        esc_preds.append(eval_res['decision'])
    escalation_metrics = EscalationMetrics.evaluate(y_gold_action, esc_preds)
    
    print("\n[Bonus] Loading Judge Agreement Benchmark Statistics...")
    with open("data_analysis/judge_human_agreement.json", "r") as f:
        judge_stats = json.load(f)

    elapsed = round(time.time() - t0, 1)

    # Master results compilation
    master_results = {
        "benchmark_date": "2026-09-13",
        "elapsed_seconds": elapsed,
        "evaluation_samples": len(gold_df),
        "intent_classification": {
            "majority_baseline": {
                "accuracy": maj_metrics["accuracy"],
                "macro_f1": maj_metrics["macro_f1"],
                "weighted_f1": maj_metrics["weighted_f1"]
            },
            "tfidf_logistic_baseline": {
                "accuracy": tfidf_metrics["accuracy"],
                "macro_f1": tfidf_metrics["macro_f1"],
                "weighted_f1": tfidf_metrics["weighted_f1"]
            },
            "main_sentence_transformer": {
                "accuracy": main_metrics["accuracy"],
                "macro_f1": main_metrics["macro_f1"],
                "weighted_f1": main_metrics["weighted_f1"],
                "macro_precision": main_metrics["macro_precision"],
                "macro_recall": main_metrics["macro_recall"],
                "per_class": main_metrics["per_class"]
            }
        },
        "retrieval": retrieval_metrics,
        "escalation": escalation_metrics,
        "judge_agreement": judge_stats
    }
    
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "="*70)
    print("                     OFFICIAL RESULTS SUMMARY TABLE")
    print("="*70)
    print(f"{'System':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 73)
    print(f"{'Majority Baseline':<35} | {maj_metrics['accuracy']:<10.4f} | {maj_metrics['macro_f1']:<10.4f} | {maj_metrics['weighted_f1']:<10.4f}")
    print(f"{'TF-IDF + Logistic Regression':<35} | {tfidf_metrics['accuracy']:<10.4f} | {tfidf_metrics['macro_f1']:<10.4f} | {tfidf_metrics['weighted_f1']:<10.4f}")
    print(f"{'Main System (Sentence Transformer)':<35} | {main_metrics['accuracy']:<10.4f} | {main_metrics['macro_f1']:<10.4f} | {main_metrics['weighted_f1']:<10.4f}")
    print("-" * 73)
    print(f"\nComponent Performance Metrics:")
    print(f"- Intent Classification Macro F1: {main_metrics['macro_f1']:.4f}")
    print(f"- Retrieval Recall@5:             {retrieval_metrics['recall_at_5']:.4f}")
    print(f"- Retrieval MRR:                  {retrieval_metrics['mrr']:.4f}")
    print(f"- Escalation F1:                  {escalation_metrics['escalate_f1']:.4f}")
    print(f"- Escalation False Auto-Handle:   {escalation_metrics['false_auto_handle_rate']*100:.1f}% ({escalation_metrics['false_auto_handles_fn']}/{len(gold_df)})")
    print(f"- LLM Judge Average Score:        {judge_stats['average_judge_score']} / 5.0")
    print(f"- Judge-Human Within-1-Pt Agree:  {judge_stats['within_one_point_pct']}%")
    print(f"\nExecution Time: {elapsed} seconds (Fully reproducible in < 15 minutes).")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_complete_evaluation()
