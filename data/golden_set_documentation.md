# Golden Evaluation Dataset Documentation

## 1. Overview
The Golden Evaluation Dataset comprises **200 manually curated and verified customer inquiries** designed to evaluate the Hiver AI Customer Support Agent for `AmazonHelp`.

* **Total Examples**: 200
* **Storage**: `data/golden_set.csv`
* **Target Brand**: `AmazonHelp`
* **Version**: 1.0 (Strictly frozen for reproducible benchmarking)

## 2. Intent Distribution
| Intent | Count | Percentage |
|---|---|---|
| `delivery_tracking_delay` | 35 | 17.5% |
| `return_and_replacement` | 30 | 15.0% |
| `refund_and_cancellation` | 30 | 15.0% |
| `service_complaint_escalation` | 30 | 15.0% |
| `prime_and_subscription` | 25 | 12.5% |
| `account_and_payment_security` | 25 | 12.5% |
| `general_product_inquiry` | 25 | 12.5% |
| **Total** | **200** | **100.0%** |

## 3. Escalation Action Distribution
| Expected Action | Count | Percentage | Primary Rationale |
|---|---|---|---|
| `AUTO_HANDLE` | 114 | 57.0% | Standard informational inquiries, tracking, self-service return/cancellation links, general policies. |
| `ESCALATE` | 86 | 43.0% | Account/financial security, stolen items, severe repeat complaints, legal/regulatory threats, policy exceptions. |

## 4. Sampling Strategy & Coverage
To ensure realistic stress-testing rather than trivial evaluation:
* **Common (96 examples)**: Typical customer scenarios that represent high-frequency contact drivers.
* **Complaints (43 examples)**: High emotional valence, repeated unresolved tickets, demanding human management.
* **Short Messages (16 examples)**: Terse queries (e.g. "delivery late", "stock when?", "cancel order asap").
* **Informal / Slang (13 examples)**: Colloquial Twitter expressions ("pkg late af", "gimme my refund now", "trash support").
* **Edge Cases (16 examples)**: Complex situations (e.g. personal items returned by mistake, hazmat leaks, multi-box discrepancies).
* **Rare Topics (16 examples)**: Infrequent policy scenarios (APO/FPO shipping, bulk freight TVs, executive escalations).

## 5. Leakage Prevention
All golden evaluation examples are strictly isolated from the training corpus (`data/sample/train.parquet`). The retrieval index and ML classifiers never train on these examples.
