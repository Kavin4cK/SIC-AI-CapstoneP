# ML Module Integration & Handoff Specification
## Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
**Module:** Machine Learning Detection Engine (Phases 1–3 & Final Evaluation)  
**Target Audience:** Downstream LLM-as-a-Judge Module Team & Operator Feedback Loop Engineers  
**Version:** 1.0 (Final Audited Release)  
**Date:** November 2026

---

## 1. Executive Summary

This document specifies the technical interface, data schemas, operational semantics, and integration protocols between the **Machine Learning Module** and the downstream **LLM-as-a-Judge / Human Operator Review Module**.

The ML module is responsible for:
1. Ingesting raw credit card transaction data.
2. Applying leakage-free preprocessing (fitted exclusively on training data).
3. Scoring transactions with the benchmark **CatBoost Classifier** (`models/catboost_model.cbm`).
4. Applying the cost-optimized decision threshold ($t^* = 0.144$) derived from hypothetical financial loss minimization.
5. Outputting a clean, traceable set of flagged suspicious transactions identified by unique, persistent `TransactionID`s **strictly without ground-truth labels**.

The downstream module is responsible for assessing flagged transactions, generating contextual explanations via an LLM judge, and recording operator feedback for future continuous learning loops.

> [!NOTE]
> **Scope & Deployment Boundary**: This ML module operates as an academic and methodological benchmark for cost-sensitive learning and ML-to-LLM handoff. It is not an online core banking authorization switch. Real-world financial deployment necessitates real-time telemetry, device intelligence, IP geolocation, fraud velocity counters, and low-latency streaming infrastructure (sub-50ms SLA) not present in anonymized PCA datasets.

---

## 2. Hand-off File Specifications

### 2.1 File Location & Generation
- **Validation Run Handoff**: `outputs/evaluation/flagged_transactions.csv`
- **Batch / Live Inference Handoff**: `outputs/predictions/flagged_transactions.csv`
- **Diagnostic / Evaluation Audit File**: `outputs/evaluation/flagged_transactions_evaluation.csv` *(Internal ML team only; contains `actual_class` for verification)*

### 2.2 LLM Handoff Schema (`flagged_transactions.csv`)

| Column Name | Type | Example | Description |
|---|---|---|---|
| `TransactionID` | String | `TXN153824` | Unique, deterministic transaction identifier. |
| `fraud_probability` | Float (6 dec) | `0.999978` | Continuous fraud probability predicted by CatBoost. |
| `selected_threshold` | Float (4 dec) | `0.1440` | Frozen cost-sensitive decision threshold. |
| `selected_model` | String | `CatBoost` | Name of the scoring model. |
| `assessment_status` | String | `SUSPICIOUS_PENDING_REVIEW` | Alert triage state. Strictly pending review. |

> [!IMPORTANT]
> **Zero Ground-Truth Guarantee**: The LLM handoff file (`outputs/evaluation/flagged_transactions.csv` and any file produced by `src/predict.py`) **strictly omits `actual_class`, `Class`, or any target labels**. The LLM module receives only feature alerts and predicted risk scores, ensuring zero data leakage and unbiased judgment.

---

## 3. Interpreting Model Outputs

### 3.1 Probability Semantics & Class-Weighting Effects
- The scoring model (CatBoost) was trained with `auto_class_weights='Balanced'` to counteract extreme class imbalance ($0.173\%$ prevalence).
- As a consequence of balanced gradient weighting, output probabilities are shifted upward relative to the base prevalence rate (mean predicted probability $0.0032$ vs base rate $0.0017$).
- Therefore, a transaction with `fraud_probability = 0.15` is **highly suspicious** and exceeds the cost-optimal decision threshold ($0.144$), representing approximately the 99.7th percentile of transaction risk.

### 3.2 Decision Threshold ($t^* = 0.1440$)
- The default naive cutoff of $0.50$ is **sub-optimal for fraud detection** because missing a fraud costs 100x more than reviewing a false alarm (hypothetical ratio: ₹500 vs ₹5).
- At threshold $0.1440$, the model catches **85.14% of fraudulent transactions** (63 out of 74 on validation; 63 out of 74 on test) with a manageable review volume (148 alerts per 42,721 validation transactions = $0.35\%$ triage rate; 133 alerts per 42,722 test transactions = $0.31\%$ triage rate).

### 3.3 Assessment Status (`SUSPICIOUS_PENDING_REVIEW`)
- Every transaction flagged by the ML module is labeled `SUSPICIOUS_PENDING_REVIEW`.
- **Under no circumstances should this status be interpreted as confirmed fraud.** It indicates an elevated risk score exceeding the financial risk tolerance cutoff, warranting secondary automated reasoning (LLM) and human operator verification.

---

## 4. Raw Feature Lookup Protocol

To generate rich contextual prompts for the LLM judge, the downstream pipeline can retrieve the raw, unscaled features of any flagged transaction using its `TransactionID`.

### Programmatic Python Lookup:
```python
import pandas as pd

# Load the raw dataset with transaction mapping
df_raw = pd.read_csv("data/creditcard.csv")
df_raw["TransactionID"] = [f"TXN{i+1:06d}" for i in range(len(df_raw))]

def get_transaction_features(txn_id: str) -> dict:
    match = df_raw[df_raw["TransactionID"] == txn_id]
    if match.empty:
        raise KeyError(f"TransactionID '{txn_id}' not found.")
    record = match.iloc[0].to_dict()
    # Exclude Class label from LLM context
    record.pop("Class", None)
    return record

# Example lookup
features = get_transaction_features("TXN153824")
print(f"Transaction {features['TransactionID']}: Amount=${features['Amount']:.2f}, Elapsed Time={features['Time']:.1f}s")
```

---

## 5. Guidelines for LLM-as-a-Judge Prompting & Guardrails

When constructing prompts for the LLM-as-a-judge module, adhere to the following data constraints:

1. **PCA Anonymization Guardrail**:
   - Features `V1` through `V28` are mathematical principal components derived via PCA to preserve customer confidentiality.
   - **Do NOT prompt the LLM to invent synthetic merchant names, cardholder names, geographical locations, or physical transaction stories.** The LLM must assess statistical anomalies, feature magnitudes, extreme standard deviations, and Amount patterns.
2. **Elapsed Time Interpretation**:
   - The `Time` feature represents **elapsed seconds from the first recorded transaction in the dataset**, spanning 48 consecutive hours (0 to 172,792 seconds).
   - **Do NOT interpret `Time` as clock timestamps (e.g. 02:00 AM)**.
3. **Transaction Amount Analysis**:
   - The dataset median amount is **$22.00** (IQR: $71.83). High dollar amounts relative to this baseline combined with extreme negative/positive PCA components (e.g. `V14`, `V17`, `V12`, `V10`, `V4`) provide strong statistical evidence of anomalous activity.
4. **Structured Output Expectation**:
   - We recommend that the LLM-as-a-judge output structured JSON:
     ```json
     {
       "TransactionID": "TXN153824",
       "llm_verdict": "CONFIRM_SUSPICIOUS",
       "confidence": "HIGH",
       "risk_drivers": ["Extreme deviation on V14 (-6.8)", "Anomalous Amount ($120.00 vs $22.00 median)"],
       "operator_action_recommended": "TEMP_HOLD_CALL_CARDHOLDER"
     }
     ```

---

## 6. Reusable Inference API for Live & Batch Prediction

The ML module provides an end-to-end inference engine (`src/predict.py`).

### 6.1 Python Programmatic API
```python
from src.predict import FraudInferencePipeline
import pandas as pd

# 1. Initialize pipeline
pipeline = FraudInferencePipeline(
    model_path="models/catboost_model.cbm",
    preprocessor_path="models/preprocessor.joblib",
    default_threshold=0.144,
)

# 2. Score new incoming transactions (DataFrame or CSV path)
df_new = pd.read_csv("path/to/incoming_batch.csv")

# Returns only flagged suspicious transactions (no ground truth)
flagged_df = pipeline.predict(
    data=df_new,
    threshold=0.144,
    return_all=False,
    output_path="outputs/predictions/flagged_transactions.csv"
)

# Or return all scored transactions with boolean is_suspicious flag
all_df = pipeline.predict(df_new, return_all=True)
```

### 6.2 Command-Line Interface (CLI)
```bash
# Standard batch inference (exports flagged transactions)
python src/predict.py --input data/incoming_transactions.csv --output outputs/predictions/flagged_transactions.csv

# Custom decision threshold
python src/predict.py --input data/incoming_transactions.csv --threshold 0.100

# Full output mode (scores every transaction with is_suspicious boolean)
python src/predict.py --input data/incoming_transactions.csv --return-all --output outputs/predictions/all_scored.csv
```

---

## 7. Operator Feedback Loop Protocol

When human operators review the LLM-triaged transactions, the feedback loop should capture the final operator determination in the following format:

| Field Name | Type | Description |
|---|---|---|
| `TransactionID` | String | Matches ML and LLM handoff identifier. |
| `ml_probability` | Float | Original model score. |
| `llm_verdict` | String | LLM-as-a-judge assessment. |
| `operator_decision` | String | `CONFIRMED_FRAUD` \| `CONFIRMED_GENUINE` \| `INCONCLUSIVE`. |
| `operator_timestamp` | ISO 8601 | Review timestamp. |
| `operator_id` | String | Reviewing analyst identifier. |
| `action_taken` | String | `CARD_BLOCKED` \| `CHARGEBACK_INITIATED` \| `WHITELISTED`. |

This log will be consumed during scheduled model retraining to continuously update the training dataset and recalibrate cost-sensitive thresholds.

---

## 8. Summary of Performance Benchmarks (Audited)

| Metric | Validation Set (42,721 tx) | Test Set (42,722 tx, Untouched) |
|---|---|---|
| **Model** | CatBoost (`models/catboost_model.cbm`) | CatBoost (`models/catboost_model.cbm`) |
| **Decision Threshold** | 0.1440 | 0.1440 |
| **PR-AUC (Average Precision)** | 0.7996 | **0.8298** |
| **ROC-AUC** | 0.9764 | **0.9710** |
| **Accuracy** | 99.78% | **99.81%** |
| **Fraud Recall** | **85.14%** (63/74 frauds) | **85.14%** (63/74 frauds)* |
| **Fraud Precision** | 42.57% | **47.37%** |
| **Fraud F1-Score** | 0.5676 | **0.6087** |
| **True Positives (TP)** | 63 | 63 |
| **False Positives (FP)** | 85 | **70** |
| **False Negatives (FN)** | 11 | 11 |
| **True Negatives (TN)** | 42,562 | 42,578 |
| **Total Review Volume (Flagged)** | 148 (0.35%) | **133 (0.31%)** |
| **Total Hypothetical Cost (₹5/₹500)** | INR 5,925.00 | **INR 5,850.00** |
| **Average Cost per Transaction** | INR 0.138691 | **INR 0.136932** |

*\*Statistical Note on Recall: The identical point recall (63/74 = 85.14%) across partitions is an empirical observation on this partition. For N=74 fraud cases, the 95% Wilson confidence interval is [75.3%, 91.6%], with bootstrap standard deviation of ±5.07%, reflecting consistent partition performance rather than theoretical invariance.*

---
*For questions or integration support regarding the ML module, refer to [`README.md`](file:///c:/Users/A%20S%20Nemitha/Desktop/SIC-Capstone%20Project/README.md) or run `pytest tests/`.*
