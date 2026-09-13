import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
import re

PROCESSED_PATH = os.path.join("data", "processed", "amazon_conversations.parquet")

def explore_intents():
    print("Loading processed conversations for intent discovery...")
    df = pd.read_parquet(PROCESSED_PATH)
    
    # Sample 25,000 customer messages for robust topic discovery
    sample = df.sample(n=min(25000, len(df)), random_state=42)
    texts = sample['customer_text_clean'].tolist()
    
    print(f"Analyzing {len(texts):,} customer inquiries...")
    
    # 1. Frequent keywords and bigrams
    stopwords = set([
        'to', 'the', 'i', 'my', 'and', 'is', 'it', 'a', 'for', 'in', 'of', 'on', 'have',
        'you', 'that', 'this', 'me', 'with', 'be', 'was', 'so', 'but', 'not', 'are',
        'amazon', 'help', 'brand', 'user', 'url', 'please', 'can', 'get', 'at', 'just',
        'from', 'do', 'what', 'when', 'if', 'been', 'no', 'as', 'will', 'an', 'up',
        'out', 'how', 'they', 'your', 'has', 'now', 'all', 'one', 'would', 'there',
        'had', 'why', 'got', 'said', 'we', 'by', 'or', 'more', 'about', 'still', 'am'
    ])
    
    def get_ngrams(text_list, n=2):
        ngrams = []
        for t in text_list:
            words = [w.lower() for w in re.findall(r'[a-z]{2,}', t) if w.lower() not in stopwords]
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i+n]))
        return Counter(ngrams)
    
    bigrams = get_ngrams(texts, 2).most_common(30)
    print("\nTop 20 Bigrams in Customer Messages:")
    for bg, count in bigrams[:20]:
        print(f"  - {bg}: {count}")

    # 2. Topic Modeling via NMF (Non-negative Matrix Factorization)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(stopwords), ngram_range=(1, 2), min_df=5)
    tfidf = vectorizer.fit_transform(texts)
    
    n_topics = 8
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=200)
    nmf.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\nDiscovered {n_topics} Latent Topics:")
    topics_summary = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics_summary.append({"topic_id": topic_idx, "top_words": top_features})
        print(f"  Topic {topic_idx + 1}: {', '.join(top_features)}")

    out_file = os.path.join("data_analysis", "discovered_topics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(topics_summary, f, indent=2)
    print(f"\nSaved discovered topics to {out_file}")

if __name__ == "__main__":
    explore_intents()
