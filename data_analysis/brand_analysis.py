import os
import json
import pandas as pd
import numpy as np
import re

RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
OUTPUT_JSON = os.path.join("data_analysis", "brand_analysis.json")
OUTPUT_MD = os.path.join("data_analysis", "brand_ranking.md")

def analyze_brands():
    print("Loading data for brand analysis...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    # Identify brands
    brand_counts = df[df['inbound'] == False]['author_id'].value_counts()
    top_candidate_names = brand_counts.head(10).index.tolist()
    
    print(f"Top 10 brands by support tweet count: {top_candidate_names}")
    
    # Map tweet_id to row for fast lookup of customer-support pairs
    # Filter only relevant tweets: support tweets from top brands and tweets they replied to
    support_df = df[df['author_id'].isin(top_candidate_names) & (df['inbound'] == False)].copy()
    
    # support tweets have in_response_to_tweet_id pointing to the customer tweet
    support_replies = support_df[support_df['in_response_to_tweet_id'].notnull()].copy()
    # Convert in_response_to_tweet_id to int/str
    support_replies['in_response_to_tweet_id'] = support_replies['in_response_to_tweet_id'].astype(np.int64, errors='ignore')
    
    # Merge with original tweets to get the customer message
    # Keep essential columns from df
    customer_tweets = df[['tweet_id', 'author_id', 'text', 'created_at', 'inbound']].copy()
    
    print("Joining support replies with customer tweets...")
    pairs = pd.merge(
        support_replies[['tweet_id', 'author_id', 'text', 'in_response_to_tweet_id', 'created_at']],
        customer_tweets,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_support', '_customer')
    )
    
    # Only keep where customer tweet is inbound == True
    pairs = pairs[pairs['inbound'] == True].copy()
    
    print(f"Total reconstructed Customer-Support pairs across top brands: {len(pairs):,}")
    
    brand_stats = []
    
    # Keywords indicating deflection to private messages / DM
    dm_regex = re.compile(r'\b(dm|direct message|private message|pm)\b', re.IGNORECASE)
    
    for brand in top_candidate_names:
        b_pairs = pairs[pairs['author_id_support'] == brand]
        b_support_total = int(brand_counts[brand])
        pair_count = len(b_pairs)
        
        if pair_count == 0:
            continue
            
        cust_texts = b_pairs['text_customer'].dropna().tolist()
        supp_texts = b_pairs['text_support'].dropna().tolist()
        
        # Word counts
        cust_word_lens = [len(t.split()) for t in cust_texts]
        avg_cust_len = float(np.mean(cust_word_lens)) if cust_word_lens else 0.0
        
        # Deflection rate (how many support answers just say "DM us")
        dm_count = sum(1 for t in supp_texts if dm_regex.search(t))
        dm_rate = round(dm_count / pair_count * 100, 2)
        
        # Issue vocabulary diversity (unique words / total words in customer queries)
        all_cust_words = [w.lower() for t in cust_texts for w in re.findall(r'[a-z]{3,}', t.lower())]
        vocab_size = len(set(all_cust_words))
        diversity_ratio = round(vocab_size / max(len(all_cust_words), 1) * 100, 3)
        
        # Calculate a suitability score
        # High volume of pairs (+), good length of customer questions (+), reasonable variety (+), 
        # but penalize excessive deflection where agent can't do anything public
        suitability_score = round(
            (np.log1p(pair_count) * 20) + 
            (min(avg_cust_len, 25) * 1.5) + 
            (min(vocab_size / 500, 20)) - 
            (dm_rate * 0.2), 
            2
        )
        
        brand_stats.append({
            "brand": brand,
            "total_support_tweets": b_support_total,
            "reconstructed_pairs": pair_count,
            "avg_customer_msg_words": round(avg_cust_len, 1),
            "dm_deflection_rate_pct": dm_rate,
            "customer_vocabulary_size": vocab_size,
            "suitability_score": suitability_score
        })
        
    # Sort by suitability score descending
    brand_stats.sort(key=lambda x: x['suitability_score'], reverse=True)
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(brand_stats, f, indent=2)
        
    # Generate Markdown report
    md_lines = [
        "# Brand Selection and Analysis Report",
        "",
        "| Rank | Brand | Total Support Tweets | Paired Dialogues | Avg Cust Words | DM Deflection % | Vocab Size | Suitability Score |",
        "|---|---|---|---|---|---|---|---|"
    ]
    for i, b in enumerate(brand_stats, 1):
        md_lines.append(
            f"| {i} | **{b['brand']}** | {b['total_support_tweets']:,} | {b['reconstructed_pairs']:,} | "
            f"{b['avg_customer_msg_words']} | {b['dm_deflection_rate_pct']}% | {b['customer_vocabulary_size']:,} | {b['suitability_score']} |"
        )
        
    recommended = brand_stats[0]['brand']
    md_lines.extend([
        "",
        f"### Recommendation: `{recommended}`",
        f"Based on paired dialogue volume, issue depth, customer query length, and substantive support resolutions, **{recommended}** is the top candidate for our customer-support AI agent."
    ])
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"Saved brand analysis to {OUTPUT_JSON} and {OUTPUT_MD}")
    print("\nTop 5 Ranked Brands:")
    for b in brand_stats[:5]:
        print(f"  {b['brand']}: {b['reconstructed_pairs']:,} pairs, {b['dm_deflection_rate_pct']}% DM rate, Score: {b['suitability_score']}")

if __name__ == "__main__":
    analyze_brands()
