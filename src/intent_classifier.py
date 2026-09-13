import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "intent_classifier.joblib")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class SentenceTransformerIntentClassifier:
    """
    Production-grade Intent Classifier using Sentence Transformers (all-MiniLM-L6-v2)
    embeddings with a balanced Logistic Regression head.
    Combines dense semantic representations with fast, explainable linear decision boundaries.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced', random_state=42)
        self.classes_ = []

    def fit(self, X_train: List[str], y_train: List[str]):
        print(f"[IntentClassifier] Encoding {len(X_train):,} training examples via {self.model_name}...")
        embeddings = self.encoder.encode(X_train, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        print("[IntentClassifier] Training classifier head...")
        self.clf.fit(embeddings, y_train)
        self.classes_ = self.clf.classes_.tolist()
        print(f"[IntentClassifier] Classes: {self.classes_}")

    def predict(self, text: str) -> Dict[str, Any]:
        emb = self.encoder.encode([text], normalize_embeddings=True)
        probs = self.clf.predict_proba(emb)[0]
        best_idx = np.argmax(probs)
        best_intent = self.classes_[best_idx]
        confidence = float(probs[best_idx])
        
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, probs)}
        
        return {
            "intent": best_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        }

    def predict_batch(self, texts: List[str]) -> List[str]:
        embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return self.clf.predict(embeddings).tolist()

    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return self.clf.predict_proba(embeddings)

    def save(self, path: str = MODEL_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model_name": self.model_name,
            "classes": self.classes_,
            "clf": self.clf
        }, path)
        print(f"[IntentClassifier] Saved model artifacts to {path}")

    @classmethod
    def load(cls, path: str = MODEL_FILE) -> "SentenceTransformerIntentClassifier":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}. Run training first.")
        data = joblib.load(path)
        instance = cls(model_name=data["model_name"])
        instance.clf = data["clf"]
        instance.classes_ = data["classes"]
        print(f"[IntentClassifier] Successfully loaded model from {path}")
        return instance
