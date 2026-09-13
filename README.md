# Hiver Customer Support AI Agent (`AmazonHelp`)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproduce-under%2015%20min-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evaluation-focused, production-grade AI customer support agent designed for **`AmazonHelp`** on Twitter. Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## 1. Executive Summary

This repository implements a complete, grounded, and rigorously evaluated autonomous customer-support system:
* **Target Brand**: `AmazonHelp` (selected through empirical ranking across 108 brands).
* **Intent Classification**: 7-class operational taxonomy tailored to e-commerce support.
* **Retrieval-Augmented Generation (RAG)**: FAISS vector retrieval grounded in 7,000+ historical support resolutions.
* **Safety Escalation Engine**: Multi-tiered decision layer (Auto-Handle vs. Escalate) with explicit audit reasons.
* **Evaluation Harness**: Benchmark comparison against Majority and TF-IDF baselines on a frozen 200-sample Golden Set.
* **LLM-as-a-Judge**: Reply quality scoring with human agreement validation and leniency analysis.

---

## 2. Headline Evaluation Results

All numbers are **100% empirical** and reproducible in under 1 minute via `python scripts/run_evaluation.py`:

| System | Accuracy | Macro F1 | Weighted F1 | Core Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Baseline** | `0.1750` | `0.0426` | `0.0521` | Always predicts `delivery_tracking_delay` |
| **TF-IDF + Logistic Regression** | `0.6750` | `0.6718` | `0.6777` | N-gram sublinear TF-IDF + balanced linear classifier |
| **Main System (Sentence Transformer)** | **`0.7350`** | **`0.7365`** | **`0.7382`** | `all-MiniLM-L6-v2` dense embeddings + balanced Logistic Head |

### Component Performance Summary

| Component | Metric | Score | Key Takeaway |
| :--- | :--- | :---: | :--- |
| **Intent Classifier** | Macro F1 | **`0.7365`** | +6.5% absolute over classical TF-IDF |
| **FAISS Retrieval** | Recall@5 | **`0.8850`** | 88.5% of queries find relevant resolution in top 5 |
| **FAISS Retrieval** | MRR | **`0.7507`** | First relevant hit typically at Rank 1 or 2 |
| **Escalation Policy** | Escalation F1 | **`0.8611`** | High recall on security and customer distress |
| **Escalation Policy** | Precision | **`0.9118`** | Minimal unnecessary agent overhead |
| **Escalation Policy** | False Auto-Handle | **`18.4%`** | Critical safety metric actively tracked |
| **LLM Reply Judge** | Overall Score | **`4.80 / 5.0`** | Evaluated on 6-factor rubric |
| **Judge Validation** | Within-1 Agreement | **`86.0%`** | Discovered systematic LLM judge leniency |

---

## 3. Architecture

```mermaid
flowchart TD
    A[Customer Twitter Message] --> B[Text Preprocessor]
    B --> C[all-MiniLM-L6-v2 Embedding Engine]
    B --> D[Deterministic Safety Rule Engine]
    
    C --> E[Intent Classifier Head]
    C --> F[FAISS Vector Index (7k Resolutions)]
    
    E -->|Intent & Confidence| G[Multi-Layer Escalation Policy]
    F -->|Top-k Historical Matches| G
    D -->|Safety & Legal Triggers| G
    
    G -->|Decision: AUTO_HANDLE or ESCALATE| H[Grounded Response Generator]
    F -->|Historical Brand Evidence| H
    
    H --> I[Structured JSON Output]
```

---

## 4. Empirical Brand Selection (`AmazonHelp`)

From the 2.8M row Kaggle dataset, `AmazonHelp` was selected over 108 brands based on hard quantitative evidence (`data_analysis/brand_ranking.md`):

| Rank | Brand | Total Support Tweets | Reconstructed Dialogue Pairs | DM Deflection Rate | Vocab Size | Suitability Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`AmazonHelp`** | **169,840** | **168,814** | **0.82%** | **91,987** | **289.51** |
| 2 | `AppleSupport` | 106,860 | 106,646 | 52.56% | 48,188 | 269.26 |
| 3 | `Uber_Support` | 56,270 | 56,160 | 35.15% | 30,475 | 263.41 |
| 4 | `Delta` | 42,253 | 42,114 | 17.12% | 26,191 | 259.05 |
| 5 | `AmericanAir` | 36,764 | 36,531 | 16.78% | 28,889 | 257.69 |

**Crucial Insight:** Brands like AppleSupport (52.6% deflection) and TMobileHelp (82.1% deflection) largely post canned deflection tweets (*"Please DM us your info"*), providing zero historical grounding. Amazon resolves issues with substantive public guidance, tracking tools, and policy links.

---

## 5. Intent Taxonomy (`data/intent_taxonomy.json`)

Derived directly from Amazon customer topic discovery:
1. `delivery_tracking_delay`: Late deliveries, tracking numbers, missing drop-offs.
2. `return_and_replacement`: Damaged units, wrong items, return labels, replacements.
3. `refund_and_cancellation`: Order cancellation, missing refunds, billing reversals.
4. `prime_and_subscription`: Prime fees, Video/Music streaming errors, Prime cancellation.
5. `account_and_payment_security`: 2FA/OTP issues, password resets, account compromise, fraud.
6. `service_complaint_escalation`: Severe frustration, supervisor demands, legal/regulatory threats.
7. `general_product_inquiry`: Restock dates, warranty, product compatibility, promotions.

---

## 6. Reproduction in Under 15 Minutes

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-org/hiver-support-agent.git
cd hiver-support-agent
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### Step 2: Prepare Sample Dataset & Build Retrieval Index (1–2 minutes)
```bash
python scripts/prepare_sample.py
python scripts/train.py
python scripts/build_index.py
```

### Step 3: Run Automated Evaluation Harness (30 seconds)
```bash
python scripts/run_evaluation.py
```

### Step 4: Run Unit & Integration Tests
```bash
pytest tests/ -v
```

### Step 5: Launch Interactive Web Demo (Optional)
```bash
streamlit run app/streamlit_app.py
```

---

## 7. Structured JSON Output Schema

```json
{
  "message": "Where is my package? The tracking has not updated for 3 days.",
  "cleaned_message": "Where is my package? The tracking has not updated for 3 days.",
  "intent": {
    "label": "delivery_tracking_delay",
    "confidence": 0.87,
    "probabilities": {
      "delivery_tracking_delay": 0.87,
      "refund_and_cancellation": 0.04,
      "return_and_replacement": 0.03
    }
  },
  "retrieved_examples": [
    {
      "rank": 1,
      "score": 0.76,
      "conversation_id": "conv_008124",
      "historical_customer_issue": "Tracking TBA892019 hasn't updated since Monday",
      "historical_support_reply": "You can view the latest courier scans in Your Orders..."
    }
  ],
  "decision": "AUTO_HANDLE",
  "decision_reason": "Standard operational inquiry with established resolution guidelines and high confidence.",
  "escalation_trigger": "STANDARD_AUTO_HANDLE",
  "reply": "We're sorry to hear about the delay with your delivery! You can view the real-time tracking status under 'Your Orders'...",
  "evidence": ["Historical Conversation conv_008124 (Similarity: 0.76)"],
  "warnings": [],
  "confidence": 0.86
}
```

---

## 8. Failure Analysis Summary (Top 5 Real Failures)

Extracted directly from empirical benchmark logs (`scripts/analyze_failures.py`):
1. **Hybrid Delivery-Damage Overlap** (`gold_030`): Driver threw package over fence breaking ceramic inside $\rightarrow$ Misclassified as replacement instead of courier conduct.
2. **Split Shipment Confusion** (`gold_024`): Missing item from multi-item box $\rightarrow$ Misclassified as damaged return instead of split shipment tracking.
3. **Misdelivery to Neighbor Privacy Breach** (`gold_025`): Driver dropped parcel two houses down $\rightarrow$ False Auto-Handle due to polite customer phrasing.
4. **Stolen In-Transit High-Value Goods** (`gold_040`): Empty box received without phone $\rightarrow$ Misclassified as account fraud instead of shipping theft.
5. **Shipping Policy Pricing Inquiries** (`gold_028`): *"Does Sunday delivery cost extra?"* $\rightarrow$ Clustered under general product pricing.

---

## 9. Project Structure

```
Hiver_Project/
├── data/
│   ├── raw/                      # Downloaded twcs.csv (516MB)
│   ├── sample/                   # 10k reproducible sample & splits
│   ├── golden_set.csv            # 200 manually curated evaluation records
│   ├── golden_set_documentation.md
│   └── intent_taxonomy.json      # 7-intent definition & criteria
├── src/
│   ├── preprocessing.py          # Text sanitation & language filter
│   ├── conversation_builder.py   # Dialogue pair reconstruction
│   ├── dataset_split.py          # Conversation-level splitter
│   ├── data_loader.py            # Unified data loading interface
│   ├── weak_labeler.py           # Domain rule weak supervision
│   ├── intent_classifier.py      # SentenceTransformer + LogReg
│   ├── retriever.py              # FAISS RAG retrieval engine
│   ├── response_generator.py     # Grounded reply synthesizer
│   ├── escalation.py             # Safety escalation engine
│   ├── judge.py                  # LLM-as-a-judge rubric evaluator
│   └── agent.py                  # End-to-end unified agent
├── evaluation/
│   ├── baselines.py              # Majority & TF-IDF baselines
│   ├── intent_metrics.py         # Multi-class & per-class metrics
│   ├── retrieval_metrics.py      # Recall@k, MRR, similarity
│   ├── escalation_metrics.py     # Escalation F1 & False Auto-Handle
│   └── judge_agreement.py       # Human vs Judge correlation
├── scripts/
│   ├── download_dataset.py       # Automated dataset downloader
│   ├── prepare_sample.py         # 10k sample preparation
│   ├── train.py                  # Model training
│   ├── build_index.py            # FAISS index generation
│   ├── run_evaluation.py         # Automated evaluation harness
│   └── analyze_failures.py       # Failure mode diagnostic
├── tests/                        # Pytest unit & integration tests
├── experiments/                  # Experiment logs & evaluation results
├── data_analysis/                # Dataset profiling & brand ranking
├── app/
│   └── streamlit_app.py          # Interactive web UI demo
├── README.md
├── REPORT.md                     # Comprehensive 6-page report
├── DECISION_LOG.md               # 12 non-obvious engineering decisions
├── INTERVIEW_NOTES.md            # Live interview preparation
├── requirements.txt
└── .env.example
```
