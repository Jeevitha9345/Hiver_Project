import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.intent_classifier import SentenceTransformerIntentClassifier
from src.weak_labeler import RuleBasedIntentLabeler

def train_main_intent_model():
    print("=== TRAINING MAIN INTENT CLASSIFIER ===")
    train_path = "data/sample/train.parquet"
    train_df = pd.read_parquet(train_path)
    
    # Label train set with weak supervision rules
    train_df['intent'], train_df['conf'] = zip(*train_df['customer_text_clean'].apply(RuleBasedIntentLabeler.predict_intent))
    
    # Select confident examples for high quality embedding representation
    confident_df = train_df[train_df['conf'] >= 0.60].copy()
    print(f"Training on {len(confident_df):,} confident examples from {len(train_df):,} total train set...")
    print("Class distribution:")
    print(confident_df['intent'].value_counts())
    
    X_train = confident_df['customer_text_clean'].tolist()
    y_train = confident_df['intent'].tolist()
    
    model = SentenceTransformerIntentClassifier()
    model.fit(X_train, y_train)
    model.save("models/intent_classifier.joblib")
    print("Model training complete.")

if __name__ == "__main__":
    train_main_intent_model()
