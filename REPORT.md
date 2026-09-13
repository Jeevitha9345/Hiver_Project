# Comprehensive Evaluation & System Report
## Autonomous Customer-Support AI Agent for `AmazonHelp`

**Author:** SDE Intern Candidate  
**Date:** September 2026  
**Repository:** `Hiver_Project`  
**Execution Runtime:** Python 3.12, PyTorch 2.9, FAISS, Sentence-Transformers  

---

## 1. Problem Framing

### What Does "Good" Mean for AmazonHelp?
Customer support for Amazon on Twitter operates under unique, high-velocity constraints:
1. **Actionable Resolution Over Deflection**: Unlike telecom or banking brands where 50–80% of tweets simply state *"Please DM us your phone number"*, Amazon customers expect immediate, concrete answers: order tracking status, return drop-off guidelines, refund processing timelines, and Prime subscription cancellation steps.
2. **Strict Safety Triage**: An e-commerce agent must never hallucinate financial commitments (*"We have refunded $500 to your card"*), promise unattainable delivery guarantees, or fail to escalate account compromises, legal litigation threats, or severe service failures.
3. **High Intent Separation**: Real customer tickets are messy, terse, and informal (*"pkg late af"*, *"driver threw box over fence"*). A "good" agent must reliably disentangle logistics delays from product return requests and customer service rages.

### What We Chose NOT to Build
* **No Unconstrained Chatbot**: We explicitly avoided building a free-form conversational chatbot that attempts to answer every inquiry with ungrounded generation.
* **No Complex Microservices or Kubernetes Overhead**: The entire system is built with modular, typed Python components runnable on a single developer machine in under 15 minutes.
* **No Unnecessary Large Fine-Tuning**: Rather than fine-tuning a 7B parameter LLM on noisy Twitter tweets, we paired dense semantic representations (`all-MiniLM-L6-v2`) with a fast linear decision boundary and FAISS nearest-neighbor retrieval, ensuring 100% reproducibility and explainability.

---

## 2. Dataset & Data Engineering

### Dataset Verification & Statistics
The system was developed and validated on the primary Kaggle dataset (`thoughtvector/customer-support-on-twitter`):
* **Total Records Profiled**: `2,811,774` tweets across 108 brands.
* **Customer Inbound Queries**: `1,537,843` (54.7%)
* **Support Brand Replies**: `1,273,931` (45.3%)
* **Missing Value Analysis**: `0` missing in text, author, or timestamps. `in_response_to_tweet_id` missing in `28.25%` of records, representing the root customer inquiry starting each thread.

### Empirical Brand Selection Rationale
Through quantitative analysis (`data_analysis/brand_ranking.md`), `AmazonHelp` was selected as the optimal target:
1. **Substantive Volume**: 169,840 support tweets and 168,814 reconstructed 1-to-1 customer-support dialogue pairs.
2. **Lowest DM Deflection Rate in the Dataset (0.82%)**: In stark contrast to `AppleSupport` (52.56% deflection), `Uber_Support` (35.15%), `comcastcares` (71.89%), and `TMobileHelp` (82.12%), Amazon support publicly answers queries with real policy information, tracking links, and resolution workflows.
3. **Vocabulary Depth**: 91,987 unique customer vocabulary terms covering shipping, physical goods damage, digital subscriptions, and payment security.

### Leakage Prevention & Conversation Splitting
* **Conversation-Level Splitting**: Random tweet-level train/test splitting causes disastrous data leakage because multiple tweets within the same conversation share identical entity identifiers, customer complaints, and support agent phrasing.
* **Methodology**: Dialogues were linked via `in_response_to_tweet_id` into atomic 2-turn conversation records. The dataset was deterministically partitioned strictly at the conversation boundary into Train (70%, 7,000 sample), Validation (15%, 1,500 sample), and Test (15%, 1,500 sample).
* **Language Sanitation**: Amazon operates global accounts. A dual filter (ASCII character ratio > 0.85 and English stopword verification) eliminated non-English queries (Spanish, French, Italian, Japanese).

---

## 3. System Architecture

```
[Customer Twitter Message]
           │
           ▼
[Text Preprocessor (Handles, URLs, ASCII)]
           │
     ┌─────┴────────────────────────────────┐
     ▼                                      ▼
[Dense Semantic Embedding]         [Rule-Based Safety Filter]
 (all-MiniLM-L6-v2, 384-dim)        (Legal, Fraud, Safety, Rude)
     │                                      │
     ├───────────────────────┐              │
     ▼                       ▼              │
[Intent Classifier]    [FAISS Vector Index] │
(Balanced LogReg Head)  (Top-k Brand Pairs) │
     │                       │              │
     ▼                       ▼              │
[Intent Label + Conf]  [Retrieved Resolutions]│
     │                       │              │
     └──────────────┬────────┘              │
                    ▼                       ▼
         [Multi-Tier Escalation Policy Engine]
         (AUTO_HANDLE vs ESCALATE + Decision Reason)
                    │
                    ▼
     [Grounded Response Generation Engine]
     (Strict Brand Guidelines + RAG Evidence)
                    │
                    ▼
       [Structured JSON Response]
```

### Component Details
1. **Intent Classifier (`src/intent_classifier.py`)**:
   Drives customer triage across the 7-class taxonomy: `delivery_tracking_delay`, `return_and_replacement`, `refund_and_cancellation`, `prime_and_subscription`, `account_and_payment_security`, `service_complaint_escalation`, and `general_product_inquiry`. Employs normalized 384-dimensional embeddings with class-weighted L2-regularized logistic regression.
2. **Historical Retriever (`src/retriever.py`)**:
   Maintains a FAISS `IndexFlatIP` storing 7,000 verified historical Amazon support dialogues. Computes exact cosine similarity in <1ms to ground responses in proven historical resolutions.
3. **Escalation Engine (`src/escalation.py`)**:
   Evaluates risk via a 4-tier waterfall: (1) Hard legal, fraud, and harassment rules, (2) Intent risk priors, (3) Classifier uncertainty (<0.35 confidence), and (4) Out-of-distribution retrieval gating (<0.40 similarity).
4. **Response Generator (`src/response_generator.py`)**:
   Synthesizes concise, courteous Amazon replies. In case of escalation, drafts an empathetic routing notice explaining why a human specialist is intervening.

---

## 4. Empirical Evaluation & Headline Results

### Frozen Golden Evaluation Set
The evaluation was performed against a frozen, manually verified **200-sample Golden Set** (`data/golden_set.csv`) strictly separated from the training corpus. The set purposefully incorporates diverse operational realities:
* **Common Queries**: 91 examples
* **High-Distress Complaints**: 62 examples
* **Short/Terse Messages**: 11 examples
* **Informal Slang / Abbreviations**: 8 examples
* **Edge Cases & Ambiguities**: 13 examples
* **Rare Topics & Policies**: 15 examples

### Official Benchmark Comparison Table

| System | Accuracy | Macro F1 | Weighted F1 | Description / Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Baseline** | **0.1750** | **0.0426** | **0.0521** | Trivial baseline; predicts `delivery_tracking_delay` for all queries. |
| **TF-IDF + Logistic Regression** | **0.6750** | **0.6718** | **0.6777** | Classical n-gram ML baseline trained on 2,093 high-confidence examples. |
| **Main System (Sentence Transformer)** | **0.7350** | **0.7365** | **0.7382** | `all-MiniLM-L6-v2` dense embeddings + balanced linear classifier head. |

### End-to-End Component Performance Table

| Component | Target Objective | Core Metric | Empirical Result |
| :--- | :--- | :--- | :---: |
| **Intent Classification** | Multi-class domain triage | **Macro F1** | **0.7365** |
| **Intent Classification** | Overall subset accuracy | **Accuracy** | **0.7350** |
| **Historical Retrieval** | Top-5 relevant dialogue recall | **Recall@5** | **0.8850** |
| **Historical Retrieval** | Ranking position quality | **MRR (Mean Reciprocal Rank)** | **0.7507** |
| **Safety Escalation** | Critical case identification | **Escalation F1** | **0.8611** |
| **Safety Escalation** | Escalation precision | **Precision** | **0.9118** |
| **Safety Escalation** | Missed escalations (Critical safety) | **False Auto-Handle Rate** | **18.4% (14 / 86)** |
| **Reply Quality** | Holistic response quality | **Average LLM Judge Score** | **4.80 / 5.0** |
| **Judge Validation** | Human-Judge alignment | **Within-1-Point Agreement** | **86.0%** |

### Per-Class Intent Breakdown (Main System)

| Intent Class | Precision | Recall | F1-Score | Golden Support |
| :--- | :---: | :---: | :---: | :---: |
| `prime_and_subscription` | **1.0000** | 0.8400 | **0.9130** | 25 |
| `refund_and_cancellation` | 0.8387 | 0.8667 | **0.8525** | 30 |
| `delivery_tracking_delay` | 0.7838 | 0.8286 | **0.8056** | 35 |
| `account_and_payment_security` | 0.6774 | 0.8400 | **0.7500** | 25 |
| `service_complaint_escalation` | 0.9000 | 0.6000 | **0.7200** | 30 |
| `return_and_replacement` | 0.6522 | 0.5000 | **0.5660** | 30 |
| `general_product_inquiry` | 0.4595 | 0.6800 | **0.5484** | 25 |

---

## 5. LLM-as-a-Judge & Human Agreement Validation

### Evaluation Rubric
The automated judge (`src/judge.py`) grades replies across 6 explicit criteria on a 1–5 scale:
1. **Relevance**: Direct alignment with customer problem statement.
2. **Helpfulness**: Clear, actionable self-service or escalation routing.
3. **Groundedness**: Supported strictly by retrieved historical brand evidence or policy.
4. **Brand Consistency**: Professional, courteous, empathetic Amazon tone.
5. **Unsupported Claims**: Penalty factor (1 = No false promises, 5 = Severe fabricated refund).
6. **Overall Quality**: Holistic customer experience rating.

### Human Agreement Empirical Results (50 Sample Benchmark)
To validate the reliability of the judge, 50 diverse examples were evaluated side-by-side with a human expert evaluator:
* **Exact Agreement**: `58.0%`
* **Within-1-Point Agreement**: `86.0%`
* **Pearson Correlation ($r$)**: `0.1618`
* **Spearman Rank Correlation ($\rho$)**: `0.1784`
* **Cohen's Kappa ($\kappa$)**: `0.0455`
* **Average Judge Rating**: `4.80 / 5.0`
* **Average Human Rating**: `4.34 / 5.0`

### Key Finding on Judge Reliability:
The LLM judge exhibited **systematic leniency** on missed escalations. While the human evaluator severely penalized cases where the agent attempted to auto-handle an inquiry requiring human review (giving 2/5 for customer safety risk), the LLM judge evaluated purely on polite phrasing and topical relevance, giving 4/5 or 5/5. This demonstrates that **LLM judges cannot be trusted blindly on safety triage without human-correlated alignment penalties**.

---

## 6. Real Failure Mode Analysis (Top 5 Failures)

Rather than hypothesizing theoretical errors, the following 5 failure modes were extracted directly from the empirical evaluation run:

### 1. Hybrid Delivery-Damage Boundary Collision
* **Real Example (`gold_030`)**: *"driver threw box over 8ft fence and broke ceramic inside"*
* **Expected Intent**: `delivery_tracking_delay` (Driver conduct investigation)
* **Actual Output**: `return_and_replacement` (Confidence: 0.7256)
* **Why It Failed**: The strong token *"broke ceramic"* pulled the dense embedding into the replacement cluster, ignoring the courier misconduct.
* **Fix**: Implement hierarchical classification: First determine Root Cause (Courier Fault), then identify Desired Action (Replacement).

### 2. Multi-Item Split Shipment vs. Missing Item Confusion
* **Real Example (`gold_024`)**: *"Ordered 5 items together, 4 arrived today but 1 is missing from the box."*
* **Expected Intent**: `delivery_tracking_delay` (Split shipment in transit)
* **Actual Output**: `return_and_replacement` (Confidence: 0.6271)
* **Why It Failed**: Semantic vector space placed *"missing from box"* closer to return claims for incomplete parts than to carrier split tracking.
* **Fix**: Inject customer order metadata (checking if an order has multiple tracking sub-packages).

### 3. Misdelivery to Neighbor Privacy Breach (False Auto-Handle)
* **Real Example (`gold_025`)**: *"My parcel was delivered to a neighbor two houses down without permission."*
* **Expected Action**: `ESCALATE` (Neighbor dispute and privacy breach)
* **Actual Action**: `AUTO_HANDLE` (Standard tracking completion reply)
* **Why It Failed**: The phrasing lacked aggressive complaint tokens (*"worst"*, *"terrible"*), matching the carrier delivery notification template.
* **Fix**: Add specific regex trigger for unauthorized neighbor/off-property drops.

### 4. High-Value In-Transit Theft Attributed to Credential Fraud
* **Real Example (`gold_040`)**: *"I received an empty box with just brown paper inside! The phone was stolen!"*
* **Expected Intent**: `return_and_replacement` (Empty parcel received / transit theft)
* **Actual Output**: `account_and_payment_security` (Confidence: 0.5161)
* **Why It Failed**: The token *"stolen"* carries heavy statistical weight towards account fraud in the linear classification head.
* **Fix**: Explicitly decouple physical package pilfering (*"empty box"*, *"tampered tape"*) from account credential compromise (*"password"*, *"hacked"*).

### 5. Policy Inquiries with Carrier Pricing Attributes
* **Real Example (`gold_028`)**: *"Does Sunday delivery cost extra?"*
* **Expected Intent**: `delivery_tracking_delay` (Carrier shipping schedules)
* **Actual Output**: `general_product_inquiry` (Confidence: 0.3439)
* **Why It Failed**: The phrase *"cost extra?"* pushed the embedding into general pricing inquiry.
* **Fix**: Expand shipping FAQ retrieval templates to bridge logistics pricing queries.

---

## 7. Mandatory Section: "What is Misleading About My Headline Number?"

### Headline Metric: `Intent Classification Macro-F1 = 0.7365`

While achieving a 73.65% Macro-F1 across 7 distinct classes on noisy social media customer data is strong, treating this number as proof of production readiness is dangerously misleading for several reasons:

1. **Class Distribution Skew in Production**:
   Our Golden Set was balanced with 25–35 examples per intent to rigorously measure rare and safety-critical classes. In real Twitter production streams, `delivery_tracking_delay` represents over 50% of inbound tickets, while `service_complaint_escalation` represents ~5%. The unweighted Macro-F1 masks the fact that a system could perform poorly on the 50% high-volume traffic while maintaining a respectable unweighted macro average.
2. **The Illusion of Offline Test Isolation**:
   In Twitter support, customers engage in multi-turn dialogues. A customer whose initial message was *"Where is my stuff"* will follow up with *"You didn't answer me"*. Offline evaluation on isolated 2-turn dialog pairs fails to capture context degradation across turns 3, 4, and 5.
3. **Binary Triage Conceals High-Cost False Negatives**:
   The headline Escalation F1 of `0.8611` appears high. However, our system still had a **False Auto-Handle rate of 18.4% (14 out of 86 cases)**. In customer operations, a False Auto-Handle on an active account takeover or a regulatory lawsuit threat can result in catastrophic legal liability. An executive would be misled into believing 86% F1 equates to safe autonomous operation.
4. **Retrieval Overlap and Template Memorization**:
   High retrieval Recall@5 (88.5%) benefits from standard e-commerce phrasing. In production, customers often describe novel hardware defects or esoteric delivery edge cases not represented in the historical FAISS index.

---

## 8. System Limitations

1. **Single-Turn Triage Boundary**: Currently optimized for 2-turn interactions (Inquiry $\rightarrow$ Resolution). Multi-turn context tracking across long threads requires conversational state tracking.
2. **Twitter Character Limit & Information Scarcity**: Many customer tweets are terse (*"help pls"*, *"not working"*). Without access to internal backend order databases (CRM/ERP), no NLP system can definitively resolve ambiguous intents without clarification.
3. **Static Vector Index**: The FAISS index is built on historical training interactions. If Amazon changes its return policy (e.g. holiday return window extension), the index must be rebuilt.

---

## 9. "One More Week" Roadmap

If given one additional week of engineering time, the following improvements would be prioritized:
1. **Active Learning & Negative Mining Loop**: Automatically sample low-confidence predictions (<0.40) into an annotation queue for weekly retraining.
2. **CRM / Order DB API Mocking**: Augment the RAG pipeline by looking up simulated customer order status by tracking ID, allowing the agent to provide exact package coordinates.
3. **Hierarchical Multi-Task Intent Head**: Transition to a two-level classifier: Tier 1 (Logistics vs. Billing vs. Security) $\rightarrow$ Tier 2 (Specific Action).
4. **Calibrated Judge Alignment**: Retrain the LLM judge rubric using preference optimization (DPO) on human ratings to eliminate the 0.46-point leniency bias on safety escalations.
