import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

INDEX_PATH = os.path.join("data", "processed", "faiss_index.bin")
METADATA_PATH = os.path.join("data", "processed", "retrieval_metadata.parquet")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class HistoricalSupportRetriever:
    """
    RAG Retrieval Engine using Sentence Transformers and FAISS.
    Finds top-k historically resolved customer issues and their verified brand resolutions.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[pd.DataFrame] = None

    def build_index(self, conversations_df: pd.DataFrame, text_col: str = "customer_text_clean"):
        print(f"[Retriever] Building FAISS index for {len(conversations_df):,} historical dialogues...")
        texts = conversations_df[text_col].tolist()
        
        # 1. Encode all customer queries
        embeddings = self.encoder.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # 2. Build FAISS Inner Product (Cosine Similarity) index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[Retriever] Added {self.index.ntotal} vectors of dimension {dim} to FAISS index.")
        
        # 3. Retain metadata
        self.metadata = conversations_df[[
            'conversation_id',
            'customer_text_clean',
            'support_text_clean',
            'customer_created_at',
            'support_created_at'
        ]].copy().reset_index(drop=True)

    def save(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        self.metadata.to_parquet(metadata_path, index=False)
        print(f"[Retriever] Saved index to {index_path} and metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH, model_name: str = EMBEDDING_MODEL_NAME) -> "HistoricalSupportRetriever":
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("FAISS index or metadata not found. Run scripts/build_index.py first.")
            
        instance = cls(model_name=model_name)
        instance.index = faiss.read_index(index_path)
        instance.metadata = pd.read_parquet(metadata_path)
        print(f"[Retriever] Loaded FAISS index ({instance.index.ntotal} items) and metadata from disk.")
        return instance

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.35) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical customer issues and historical support replies.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Retriever not initialized with index.")
            
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata.iloc[idx]
            results.append({
                "rank": rank,
                "score": round(float(score), 4),
                "is_confident": bool(score >= min_similarity),
                "conversation_id": meta['conversation_id'],
                "historical_customer_issue": meta['customer_text_clean'],
                "historical_support_reply": meta['support_text_clean']
            })
            
        return results
