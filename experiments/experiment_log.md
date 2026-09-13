# Experiment Log

Every experiment is recorded here with parameters, split details, actual metrics, and observations.

---

### Experiment 0: Dataset Profiling & Brand Suitability Benchmark
- **Date**: 2026-09-13
- **Objective**: Profile raw `twcs.csv` (2,811,774 rows) and rank top 10 brands for AI support agent feasibility.
- **Data Source**: `data/raw/twcs.csv` (516.5 MB)
- **Results**:
  - Total rows: 2,811,774 (Customer: 1,537,843, Support: 1,273,931)
  - Unique brands: 108
  - Top candidate ranking:
    1. `AmazonHelp`: 168,814 dialogue pairs | 0.82% DM deflection | Score: 289.51
    2. `AppleSupport`: 106,646 dialogue pairs | 52.56% DM deflection | Score: 269.26
    3. `Uber_Support`: 56,160 dialogue pairs | 35.15% DM deflection | Score: 263.41
    4. `Delta`: 42,114 dialogue pairs | 17.12% DM deflection | Score: 259.05
    5. `AmericanAir`: 36,531 dialogue pairs | 16.78% DM deflection | Score: 257.69
- **Observations**: `AmazonHelp` demonstrates unmatched data volume and substantive public resolution content (under 1% deflection to DM). Selected as the single target brand for the project.

---

### Experiment 1: Baseline Intent Classifiers on Golden Evaluation Set
- **Date**: 2026-09-13
- **Objective**: Establish empirical performance lower-bounds on the 200-sample Golden Set.
- **Data Split**: Trained on 2,093 high-confidence examples from `data/sample/train.parquet`; evaluated on `data/golden_set.csv` (200 records).
- **Results**:
  - **Baseline 1 (Majority Class - `delivery_tracking_delay`)**:
    - Accuracy: **0.1750** (17.5%)
    - Macro F1: **0.0426**
    - Weighted F1: **0.0521**
  - **Baseline 2 (TF-IDF + Logistic Regression)**:
    - Accuracy: **0.6750** (67.5%)
    - Macro Precision: **0.6958**
    - Macro Recall: **0.6646**
    - Macro F1: **0.6718**
    - Weighted F1: **0.6777**
- **Observations**: The majority class baseline reveals that class imbalance renders accuracy completely uninformative (17.5% accuracy with near-zero Macro-F1). TF-IDF + Logistic Regression achieves 67.18% Macro-F1, struggling primarily on `general_product_inquiry` (F1 = 0.4074) due to lexical vocabulary variance and lack of semantic generalization.
