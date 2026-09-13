import os
import re
import pandas as pd
import numpy as np
from typing import Optional
from src.preprocessing import TextPreprocessor

class ConversationBuilder:
    """
    Reconstructs 2-turn dialog conversations (Customer Problem -> Brand Support Reply)
    for a target brand from twcs.csv, applying deduplication and language filtering.
    """
    
    def __init__(self, raw_csv_path: str = os.path.join("data", "raw", "twcs.csv")):
        self.raw_csv_path = raw_csv_path
        self.preprocessor = TextPreprocessor()

    def build_brand_conversations(
        self, 
        brand: str = "AmazonHelp",
        min_customer_words: int = 3,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        print(f"Loading raw dataset from {self.raw_csv_path} for brand '{brand}'...")
        # Read raw CSV
        df = pd.read_csv(self.raw_csv_path, nrows=max_rows, low_memory=False)
        
        # 1. Identify support replies from this brand
        support_df = df[(df['author_id'] == brand) & (df['inbound'] == False) & (df['in_response_to_tweet_id'].notnull())].copy()
        print(f"Found {len(support_df):,} support reply tweets for {brand}.")
        
        support_df['in_response_to_tweet_id'] = support_df['in_response_to_tweet_id'].astype(np.int64, errors='ignore')
        
        # 2. Extract customer tweets
        cust_ids = set(support_df['in_response_to_tweet_id'].unique())
        customer_df = df[df['tweet_id'].isin(cust_ids) & (df['inbound'] == True)].copy()
        print(f"Found {len(customer_df):,} matching customer inquiry tweets.")
        
        # 3. Merge customer and support
        pairs = pd.merge(
            customer_df[['tweet_id', 'author_id', 'created_at', 'text']],
            support_df[['tweet_id', 'author_id', 'created_at', 'text', 'in_response_to_tweet_id']],
            left_on='tweet_id',
            right_on='in_response_to_tweet_id',
            suffixes=('_cust', '_supp')
        )
        print(f"Constructed {len(pairs):,} initial pairs.")
        
        # 4. Filter English / ASCII
        pairs['is_english'] = pairs['text_cust'].apply(self.preprocessor.is_english_ascii)
        pairs = pairs[pairs['is_english']].copy()
        print(f"After English/ASCII filter: {len(pairs):,} pairs.")
        
        # 5. Clean texts
        pairs['customer_text_clean'] = pairs['text_cust'].apply(self.preprocessor.clean_customer_query)
        pairs['support_text_clean'] = pairs['text_supp'].apply(self.preprocessor.clean_support_reply)
        
        # 6. Filter by word count
        pairs['cust_word_count'] = pairs['customer_text_clean'].apply(lambda t: len(t.split()))
        pairs = pairs[pairs['cust_word_count'] >= min_customer_words].copy()
        
        # 7. Deduplicate exact duplicate customer queries (keep first)
        pairs = pairs.drop_duplicates(subset=['customer_text_clean']).copy()
        print(f"After deduplication: {len(pairs):,} unique conversation pairs.")
        
        # 8. Create unified records
        pairs['conversation_id'] = [f"conv_{i:06d}" for i in range(len(pairs))]
        
        result = pairs[[
            'conversation_id',
            'tweet_id_cust',
            'tweet_id_supp',
            'created_at_cust',
            'created_at_supp',
            'text_cust',
            'customer_text_clean',
            'text_supp',
            'support_text_clean',
            'cust_word_count'
        ]].rename(columns={
            'tweet_id_cust': 'customer_tweet_id',
            'tweet_id_supp': 'support_tweet_id',
            'created_at_cust': 'customer_created_at',
            'created_at_supp': 'support_created_at',
            'text_cust': 'customer_text_raw',
            'text_supp': 'support_text_raw'
        })
        
        return result

if __name__ == "__main__":
    builder = ConversationBuilder()
    convs = builder.build_brand_conversations(brand="AmazonHelp")
    out_path = os.path.join("data", "processed", "amazon_conversations.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    convs.to_parquet(out_path, index=False)
    print(f"Saved {len(convs):,} processed conversations to {out_path}")
