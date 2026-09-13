import os
import pandas as pd
import numpy as np
from typing import Tuple

class DatasetSplitter:
    """
    Splits conversation records into train, validation, and test splits
    at the strict CONVERSATION level, preventing any dialogue leakage.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def split_conversations(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        # Ensure deterministic shuffle by conversation_id
        np.random.seed(self.random_seed)
        conv_ids = df['conversation_id'].unique()
        shuffled_ids = np.random.permutation(conv_ids)
        
        n_total = len(shuffled_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = set(shuffled_ids[:n_train])
        val_ids = set(shuffled_ids[n_train:n_train + n_val])
        test_ids = set(shuffled_ids[n_train + n_val:])
        
        train_df = df[df['conversation_id'].isin(train_ids)].copy()
        val_df = df[df['conversation_id'].isin(val_ids)].copy()
        test_df = df[df['conversation_id'].isin(test_ids)].copy()
        
        print(f"Split complete: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
        return train_df, val_df, test_df

    def create_reproducible_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 5000,
        stratify_col: str = None
    ) -> pd.DataFrame:
        """Samples a representative subset for rapid iteration and offline evaluation."""
        np.random.seed(self.random_seed)
        if len(df) <= sample_size:
            return df.copy()
            
        if stratify_col and stratify_col in df.columns:
            sampled = df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), random_state=self.random_seed)
            )
            return sampled.reset_index(drop=True)
        else:
            return df.sample(n=sample_size, random_state=self.random_seed).reset_index(drop=True)
