import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from src.preprocessing import TextPreprocessor
from src.dataset_split import DatasetSplitter

PROCESSED_PATH = os.path.join("data", "processed", "amazon_conversations.parquet")
SAMPLE_DIR = os.path.join("data", "sample")

def prepare_sample(sample_size: int = 10000, random_seed: int = 42):
    print(f"Preparing reproducible sample of {sample_size:,} conversations (Seed: {random_seed})...")
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {len(df):,} total conversations.")
    
    english_stopwords = {'the', 'to', 'and', 'my', 'i', 'is', 'it', 'for', 'you', 'in', 'on', 'with', 'me', 'have', 'was', 'at', 'this'}
    def has_english_words(text):
        if not isinstance(text, str): return False
        words = set(text.lower().split())
        return len(words.intersection(english_stopwords)) >= 2
        
    df['is_confirmed_en'] = df['customer_text_clean'].apply(has_english_words)
    df_en = df[df['is_confirmed_en']].copy()
    print(f"Confirmed English conversations: {len(df_en):,}")
    
    splitter = DatasetSplitter(random_seed=random_seed)
    sample_df = splitter.create_reproducible_sample(df_en, sample_size=sample_size)
    
    train_df, val_df, test_df = splitter.split_conversations(sample_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    sample_path = os.path.join(SAMPLE_DIR, "conversations_sample.parquet")
    train_path = os.path.join(SAMPLE_DIR, "train.parquet")
    val_path = os.path.join(SAMPLE_DIR, "val.parquet")
    test_path = os.path.join(SAMPLE_DIR, "test.parquet")
    
    sample_df.to_parquet(sample_path, index=False)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved sample dataset to {sample_path} ({len(sample_df):,} rows)")
    print(f"Train split: {train_path} ({len(train_df):,} rows)")
    print(f"Val split:   {val_path} ({len(val_df):,} rows)")
    print(f"Test split:  {test_path} ({len(test_df):,} rows)")

if __name__ == "__main__":
    prepare_sample()
