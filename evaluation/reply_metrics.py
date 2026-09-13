import re
from typing import List, Dict, Any
from collections import Counter
import numpy as np

class ReplyMetrics:
    """
    Evaluates generated response quality against historical ground-truth replies:
    Token overlap (ROUGE-L approximation, BLEU-1/2), length alignment,
    and hallucinated claim detection.
    """
    
    @staticmethod
    def compute_ngram_overlap(pred_tokens: List[str], ref_tokens: List[str], n: int = 1) -> float:
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        pred_ngrams = Counter([" ".join(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)])
        ref_ngrams = Counter([" ".join(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        overlap = sum((pred_ngrams & ref_ngrams).values())
        return overlap / max(len(pred_tokens) - n + 1, 1)

    @staticmethod
    def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @classmethod
    def evaluate_replies(cls, generated_replies: List[str], reference_replies: List[str]) -> Dict[str, Any]:
        assert len(generated_replies) == len(reference_replies), "Must have equal length"
        
        bleu1_scores = []
        bleu2_scores = []
        rouge_l_scores = []
        lengths = []
        unsupported_claim_count = 0
        
        fake_promise_regex = re.compile(r'(\$\d+|refunded immediately|guarantee within \d+ hours)', re.IGNORECASE)
        
        for gen, ref in zip(generated_replies, reference_replies):
            gen_tokens = [w.lower() for w in re.findall(r'\w+', gen)]
            ref_tokens = [w.lower() for w in re.findall(r'\w+', ref)]
            
            b1 = cls.compute_ngram_overlap(gen_tokens, ref_tokens, n=1)
            b2 = cls.compute_ngram_overlap(gen_tokens, ref_tokens, n=2)
            
            lcs = cls.longest_common_subsequence(gen_tokens, ref_tokens)
            rl = (2 * lcs / (len(gen_tokens) + len(ref_tokens))) if (len(gen_tokens) + len(ref_tokens)) > 0 else 0.0
            
            bleu1_scores.append(b1)
            bleu2_scores.append(b2)
            rouge_l_scores.append(rl)
            lengths.append(len(gen_tokens))
            
            if fake_promise_regex.search(gen):
                unsupported_claim_count += 1
                
        return {
            "num_evaluated": len(generated_replies),
            "bleu_1": round(float(np.mean(bleu1_scores)), 4),
            "bleu_2": round(float(np.mean(bleu2_scores)), 4),
            "rouge_l": round(float(np.mean(rouge_l_scores)), 4),
            "avg_word_count": round(float(np.mean(lengths)), 1),
            "unsupported_claims_rate": round(unsupported_claim_count / max(len(generated_replies), 1), 4)
        }
