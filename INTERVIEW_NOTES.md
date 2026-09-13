# Live Interview Preparation & Defense Notes
## Autonomous Customer-Support AI Agent — `AmazonHelp`

This document prepares the engineer to defend the architectural, statistical, machine learning, and critical engineering choices during live technical interviews.

---

### 1. Architecture Questions

#### Q1: Why this architecture? (Dense Embeddings + FAISS RAG + Hybrid Escalation)
**Answer:**  
"Rather than deploying an unconstrained, opaque LLM that acts as a black box, we chose a modular, decoupled architecture where each stage has a distinct, auditable responsibility. Intent classification performs coarse routing; FAISS retrieval surfaces verified historical precedents; deterministic safety rules enforce business boundaries; and the response generator adapts proven resolutions without inventing policies. This architecture guarantees sub-100ms latency, zero hallucinated refund amounts, and complete explainability for every escalation."

#### Q2: Why RAG (Retrieval-Augmented Generation) instead of fine-tuning?
**Answer:**  
"Fine-tuning bakes historical policies into static weights. If Amazon changes return windows, Prime pricing, or courier policies, a fine-tuned model requires full re-training and carries hallucination risks. RAG decouples knowledge from reasoning. We can update our FAISS vector index in real time when company policies change without retraining our intent classifier or generation layer."

#### Q3: Why FAISS?
**Answer:**  
"FAISS is an industrial-strength, C++ optimized vector search library. For our index size (7,000 to 140,000 vectors), `faiss.IndexFlatIP` performs exact inner product search in under 1 millisecond on a standard CPU. It has zero network overhead compared to cloud vector DBs like Pinecone, requires zero external infrastructure, and provides deterministic top-k results."

#### Q4: Why `all-MiniLM-L6-v2` as the embedding model?
**Answer:**  
"It represents the Pareto frontier of latency, memory footprint, and semantic quality. At just 80MB and 384 dimensions, it runs efficiently on standard developer laptops without requiring a GPU, encoding 7,000 dialogues in under 45 seconds while achieving an 88.5% Recall@5 in our retrieval benchmarks."

---

### 2. Machine Learning Questions

#### Q5: Why Macro-F1 instead of Accuracy alone?
**Answer:**  
"In customer support datasets, class distribution is inherently skewed—delivery inquiries represent over 50% of tickets, while security threats or rages represent under 5%. In our baseline evaluation, predicting the majority class (`delivery_tracking_delay`) for all examples yielded a 17.5% accuracy but a near-zero Macro-F1 of `0.0426`. Accuracy rewards memorizing the majority class and completely masks total failure on rare, safety-critical classes. Macro-F1 weights each intent equally, guaranteeing that the model cannot hide poor performance on fraud or complaints."

#### Q6: Why this 7-class intent taxonomy instead of Banking77?
**Answer:**  
"Banking77 was designed for retail banking (ATMs, card chips, wire transfers), which is irrelevant to e-commerce operations. Using NMF topic modeling and bigram frequency analysis on 25,000 Amazon customer queries, we discovered the real operational clusters of customer pain: tracking delays, physical damage/returns, refund delays, Prime subscriptions, account auth, and service complaints. 7 classes provide comprehensive coverage without fine-grained semantic overlap."

#### Q7: How did you prevent data leakage?
**Answer:**  
"Random tweet-level splitting causes severe data leakage because multi-turn interactions between the same customer and agent share identical entities and phrasing. We grouped tweets into atomic conversation dialogues using `in_response_to_tweet_id` and performed conversation-level splitting by `conversation_id`. Furthermore, our 200-sample Golden Evaluation Set was kept frozen and strictly isolated from the training corpus."

#### Q8: How did you handle class imbalance?
**Answer:**  
"We applied balanced class weighting (`class_weight='balanced'`) in our logistic regression head, dynamically penalizing errors on rare classes inversely proportional to their training frequency. Additionally, our Golden Evaluation Set was intentionally stratified to ensure robust test support for every class."

---

### 3. LLM & Generation Questions

#### Q9: How do you prevent hallucinations and fake promises?
**Answer:**  
"Through grounded synthesis boundaries:
1. The generator is provided with verified historical brand resolutions as evidence.
2. The prompt restricts generation to procedural steps ('check Your Orders') and strictly prohibits inventing specific monetary amounts or guarantees.
3. Our LLM Judge penalizes unsupported claims heavily, reducing overall scores if an unverified promise is detected."

#### Q10: Why should we trust the LLM Judge? What are its limitations?
**Answer:**  
"We did not trust the judge blindly. We conducted a 50-example human agreement study. While Within-1-point agreement was high (86.0%), we discovered a critical vulnerability: **LLM judges exhibit systematic leniency on safety triage**. When our system missed an escalation, human evaluators rated it 2/5 (customer risk), whereas the LLM judge rated it 4/5 or 5/5 because the phrasing was polite and topical. Therefore, LLM judges must always be paired with deterministic safety guardrails."

#### Q11: What happens when retrieval fails or similarity is low?
**Answer:**  
"Our escalation policy includes an Out-of-Distribution threshold: if the top retrieved historical similarity is below `0.40`, the system refuses to auto-handle the query, flags an out-of-distribution warning, and escalates the ticket to a human specialist."

---

### 4. Production Engineering & Scaling Questions

#### Q12: What happens if an external LLM API is unavailable or rate-limited?
**Answer:**  
"The agent architecture features a dual-mode generation layer. If the external API fails or is unconfigured, it gracefully falls back to deterministic grounded synthesis derived from the top-matching historical template. The system never crashes or drops customer tickets."

#### Q13: How would you scale this to 100,000 tickets per day?
**Answer:**  
"1. **Inference Serving**: Deploy the SentenceTransformer model using ONNX Runtime or TensorRT on Triton Inference Server.
2. **Vector Index**: Shard the FAISS index across multiple workers or migrate to an indexed HNSW cluster.
3. **Caching**: Maintain a Redis cache for exact-match customer queries (e.g. repeated tracking questions), bypassing model inference for ~30% of incoming volume."

#### Q14: How would you handle new, unseen intents (e.g. Amazon Drone Delivery)?
**Answer:**  
"Unseen intents naturally manifest with low retrieval similarity (<0.40) and high classifier entropy. Our pipeline automatically routes these cases to the `ESCALATE` queue. In production, we would cluster these escalated out-of-distribution queries using HDBSCAN to discover emerging intent candidates and update our taxonomy."

---

### 5. Critical Thinking & Senior Mentorship Questions

#### Q15: What is your biggest failure or limitation in this project?
**Answer:**  
"Our biggest limitation is multi-turn state tracking. Currently, the system evaluates each message independently. If a customer says 'Where is my order', gets an answer, and then replies 'It still hasn't arrived', the second turn lacks the order context from the first turn. In production, we would inject previous conversation turns into the context window."

#### Q16: What is misleading about your headline number?
**Answer:**  
"Our headline Intent Macro-F1 is `0.7365` and Escalation F1 is `0.8611`. What is misleading is that Escalation F1 conceals an **18.4% False Auto-Handle rate**. In customer support operations, failing to escalate an active account compromise or a legal threat has asymmetric catastrophic consequences compared to over-escalating a routine question. High F1 does not inherently mean safe operations without zero-tolerance safety guardrails."

#### Q17: What would you do with one more week?
**Answer:**  
"1. Implement hierarchical multi-task classification (Root Cause vs. Requested Action).
2. Connect simulated backend order databases to test end-to-end order status resolution.
3. Fine-tune our judge scoring rubric using Direct Preference Optimization (DPO) to eliminate the leniency bias on safety escalations."
