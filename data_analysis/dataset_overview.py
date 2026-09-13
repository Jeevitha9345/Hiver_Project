import os
import json
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
OUTPUT_PATH = os.path.join("data_analysis", "dataset_overview.json")

def inspect_dataset():
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    # Load dataset with low_memory=False to handle mixed types cleanly
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    total_rows = len(df)
    cols = df.columns.tolist()
    
    print(f"Total rows: {total_rows:,}")
    print(f"Columns: {cols}")
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / total_rows * 100, 2) for k, v in missing.items()}
    
    # Inbound distribution
    inbound_counts = df['inbound'].value_counts().to_dict()
    # Handle boolean keys in json
    inbound_counts_clean = {str(k): int(v) for k, v in inbound_counts.items()}
    
    # Unique authors
    unique_authors = df['author_id'].nunique()
    
    # Brand handles (companies replying, i.e., inbound == False)
    brands = df[df['inbound'] == False]['author_id'].value_counts()
    top_brands = brands.head(20).to_dict()
    total_brands = len(brands)
    
    # Timestamp range
    # Sample parse timestamps to avoid parsing 3M dates if slow
    sample_dates = pd.to_datetime(df['created_at'].iloc[[0, -1]], errors='coerce')
    date_min = str(sample_dates.iloc[0])
    date_max = str(sample_dates.iloc[-1])
    
    # Thread links: how many tweets have in_response_to_tweet_id
    has_in_reply_to = df['in_response_to_tweet_id'].notnull().sum()
    has_response_tweet_id = df['response_tweet_id'].notnull().sum()
    
    overview = {
        "total_rows": total_rows,
        "columns": cols,
        "missing_values": missing,
        "missing_percentage": missing_pct,
        "inbound_distribution": inbound_counts_clean,
        "unique_authors": unique_authors,
        "total_brands": total_brands,
        "top_20_brands": top_brands,
        "date_sample_start": date_min,
        "date_sample_end": date_max,
        "has_in_reply_to": int(has_in_reply_to),
        "has_in_reply_to_pct": round(has_in_reply_to / total_rows * 100, 2),
        "has_response_tweet_id": int(has_response_tweet_id),
        "has_response_tweet_id_pct": round(has_response_tweet_id / total_rows * 100, 2)
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
        
    print(f"\nSaved overview to {OUTPUT_PATH}")
    print("\n--- Summary ---")
    print(f"Total Rows: {total_rows:,}")
    print(f"Customer tweets (inbound=True): {inbound_counts.get(True, 0):,} ({inbound_counts.get(True, 0)/total_rows*100:.1f}%)")
    print(f"Support tweets (inbound=False): {inbound_counts.get(False, 0):,} ({inbound_counts.get(False, 0)/total_rows*100:.1f}%)")
    print(f"Unique Brands: {total_brands}")
    print(f"Top 5 Brands by Support Tweets:")
    for b, count in list(top_brands.items())[:5]:
        print(f"  - {b}: {count:,} tweets")

if __name__ == "__main__":
    inspect_dataset()
