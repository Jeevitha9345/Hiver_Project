import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List
from evaluation.intent_metrics import IntentMetrics
from src.weak_labeler import RuleBasedIntentLabeler

class MajorityBaselineClassifier:
    """
    Trivial baseline: Always predicts the single most common intent from the training corpus.
    """
    def __init__(self):
        self.majority_class = None

    def fit(self, y_train: List[str]):
        series = pd.Series(y_train)
        self.majority_class = series.mode()[0]
        print(f"[MajorityBaseline] Learned majority class: '{self.majority_class}'")

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_class] * len(texts)


class TfidfLogisticBaselineClassifier:
    """
    Simple classical ML baseline: TF-IDF n-grams + Logistic Regression with L2 regularization.
    """
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2), C: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        self.classifier = LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=42)

    def fit(self, X_train: List[str], y_train: List[str]):
        print(f"[TF-IDF + Logistic Regression] Vectorizing {len(X_train):,} training texts...")
        X_vec = self.vectorizer.fit_transform(X_train)
        print(f"[TF-IDF + Logistic Regression] Training Logistic Regression...")
        self.classifier.fit(X_vec, y_train)

    def predict(self, texts: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(texts)
        return self.classifier.predict(X_vec).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_vec = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_vec)


def run_baselines_on_golden_set():
    print("=== RUNNING BASELINES ON GOLDEN EVALUATION SET ===")
    
    # 1. Load Golden Set
    gold_df = pd.read_csv("data/golden_set.csv")
    X_gold = gold_df['message'].tolist()
    y_gold = gold_df['intent'].tolist()
    labels = sorted(list(set(y_gold)))
    print(f"Loaded {len(gold_df)} golden evaluation records across {len(labels)} classes.")

    # 2. Load and weakly label training data
    train_df = pd.read_parquet("data/sample/train.parquet")
    train_df['intent'], train_df['conf'] = zip(*train_df['customer_text_clean'].apply(RuleBasedIntentLabeler.predict_intent))
    
    # Filter to high confidence training examples to train clean ML baseline
    confident_train = train_df[train_df['conf'] >= 0.65].copy()
    print(f"Training ML baseline on {len(confident_train):,} high-confidence training examples...")
    
    X_train = confident_train['customer_text_clean'].tolist()
    y_train = confident_train['intent'].tolist()

    # --- BASELINE 1: Majority Class ---
    print("\n--- Evaluating Baseline 1: Majority Class ---")
    maj_clf = MajorityBaselineClassifier()
    maj_clf.fit(y_train)
    y_pred_maj = maj_clf.predict(X_gold)
    maj_metrics = IntentMetrics.evaluate(y_gold, y_pred_maj, labels=labels)
    print(IntentMetrics.format_table(maj_metrics, title="Baseline 1: Majority Class"))

    # --- BASELINE 2: TF-IDF + Logistic Regression ---
    print("\n--- Evaluating Baseline 2: TF-IDF + Logistic Regression ---")
    tfidf_clf = TfidfLogisticBaselineClassifier()
    tfidf_clf.fit(X_train, y_train)
    y_pred_tfidf = tfidf_clf.predict(X_gold)
    tfidf_metrics = IntentMetrics.evaluate(y_gold, y_pred_tfidf, labels=labels)
    print(IntentMetrics.format_table(tfidf_metrics, title="Baseline 2: TF-IDF + Logistic Regression"))

    return maj_metrics, tfidf_metrics

if __name__ == "__main__":
    run_baselines_on_golden_set()
