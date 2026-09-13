import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.retriever import HistoricalSupportRetriever

def build_retrieval_index():
    print("=== BUILDING HISTORICAL SUPPORT FAISS RETRIEVAL INDEX ===")
    train_path = "data/sample/train.parquet"
    train_df = pd.read_parquet(train_path)
    print(f"Loaded {len(train_df):,} historical dialogues from {train_path}")
    
    retriever = HistoricalSupportRetriever()
    retriever.build_index(train_df, text_col="customer_text_clean")
    retriever.save()
    print("FAISS index built and saved successfully.")

if __name__ == "__main__":
    build_retrieval_index()
