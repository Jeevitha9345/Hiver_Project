import os
import pandas as pd
from typing import Optional

class DataLoader:
    """
    Unified interface to load raw, processed, sample, and split datasets.
    """
    
    RAW_PATH = os.path.join("data", "raw", "twcs.csv")
    PROCESSED_PATH = os.path.join("data", "processed", "amazon_conversations.parquet")
    SAMPLE_PATH = os.path.join("data", "sample", "conversations_sample.parquet")
    GOLDEN_PATH = os.path.join("data", "golden_set.csv")
    
    @classmethod
    def load_processed_conversations(cls, path: Optional[str] = None) -> pd.DataFrame:
        target = path or cls.PROCESSED_PATH
        if not os.path.exists(target):
            raise FileNotFoundError(f"Processed conversations not found at {target}. Run scripts/prepare_sample.py first.")
        return pd.read_parquet(target)

    @classmethod
    def load_sample_conversations(cls, path: Optional[str] = None) -> pd.DataFrame:
        target = path or cls.SAMPLE_PATH
        if not os.path.exists(target):
            raise FileNotFoundError(f"Sample conversations not found at {target}. Run scripts/prepare_sample.py first.")
        return pd.read_parquet(target)

    @classmethod
    def load_golden_set(cls, path: Optional[str] = None) -> pd.DataFrame:
        target = path or cls.GOLDEN_PATH
        if not os.path.exists(target):
            raise FileNotFoundError(f"Golden evaluation set not found at {target}.")
        return pd.read_csv(target)
