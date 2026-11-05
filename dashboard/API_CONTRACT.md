# Operator Dashboard API Contract & Integration Guide

**Target Audience:** Downstream LLM-as-a-Judge and Model Retraining Engineering Teams  
**Host / Default Port:** `http://localhost:8000` (FastAPI) | **UI:** `http://localhost:5173` (React/Vite)  
**Specification Source:** `dashboard/backend/app.py`, `models.py`, `database.py`

---

## 1. REST API Endpoints & Verified Responses

### `GET /health`
Returns service status and active predictions file location.
- **Query Params / Body:** None
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Fraud Operator Dashboard API",
  "predictions_source": "/Users/salonisingh/SIC-AI-CapstoneP/outputs/predictions/flagged_transactions.csv"
}
```

### `GET /transactions`
Fetches scored transactions sorted strictly by `fraud_probability` descending, strictly isolated to the specified active `dataset_id`.
- **Query Params:**
  - `status` *(optional, string)*: `'pending'` (or `'suspicious_pending_review'`), `'fraud'` (`'confirmed_fraud'`), `'genuine'` (`'confirmed_genuine'`), or `'all'`. Default: all.
  - `dataset_id` *(optional, string)*: Unique dataset ID (e.g. `'default'`, `'ulb_benchmark'`, `'ds_...'`). If omitted, defaults to `'default'`. Only returns suspicious transactions (fraud_probability >= 0.144) or operator-labelled transactions.
- **Response (200 OK):**
```json
[
  {
    "TransactionID": "TEST_TXN_0001",
    "fraud_probability": 0.999913,
    "selected_threshold": 0.144,
    "selected_model": "CatBoost",
    "assessment_status": "SUSPICIOUS_PENDING_REVIEW",
    "amount": 1.0,
    "time": 7526.0,
    "operator_label": null,
    "operator": null,
    "labelled_at": null
  }
]
```
*Note: When a transaction is labelled, `assessment_status` transitions to `CONFIRMED_FRAUD` or `CONFIRMED_GENUINE`, and `operator_label`, `operator`, `labelled_at` are populated.*

### `GET /metrics`
Dynamically reads audited test/validation metrics from `outputs/evaluation/final_test_metrics.json` (fallback: `README.md`). Never hardcoded.
- **Query Params / Body:** None
- **Response (200 OK):**
```json
{
  "model_name": "CatBoost",
  "decision_threshold": 0.144,
  "recall": 0.851351,
  "precision": 0.473684,
  "accuracy": 0.998104,
  "f1_score": 0.608696,
  "pr_auc": 0.829777,
  "roc_auc": 0.971018,
  "total_hypothetical_cost": 5850.0,
  "average_cost_per_tx": 0.136932,
  "confusion_matrix": { "tp": 63, "fp": 70, "fn": 11, "tn": 42578, "flagged_transactions": 133 },
  "cost_matrix": { "c_fp": 5.0, "c_fn": 500.0, "c_tp": 0.0, "c_tn": 0.0, "disclaimer": "Academic hypothetical assumptions; not real institutional loss figures." },
  "source_file": "outputs/evaluation/final_test_metrics.json"
}
```

### `GET /flags-per-hour`
Hourly flagged alerts distribution across elapsed seconds (`int(time // 3600)`), isolated by dataset.
- **Query Params:**
  - `dataset_id` *(optional, string)*: Active dataset ID.
- **Response (200 OK):**
```json
[
  { "hour": 0, "flags": 1 },
  { "hour": 1, "flags": 2 },
  { "hour": 2, "flags": 6 }
]
```

### `POST /labels`
Persists operator or LLM adjudication. Never mutates input predictions CSV.
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "TransactionID": "TEST_TXN_0001",
  "label": "fraud",
  "operator": "llm_judge_v1",
  "dataset_id": "default"
}
```
*Validation:* `label` must be `"fraud"` or `"genuine"` (case-insensitive). Invalid values return `400 Bad Request`. `operator` defaults to `"operator_1"` if omitted.
- **Response (200 OK):**
```json
{
  "success": true,
  "message": "Transaction TEST_TXN_0001 marked as fraud.",
  "data": {
    "TransactionID": "TEST_TXN_0001",
    "label": "fraud",
    "operator": "llm_judge_v1",
    "created_at": "2026-11-04T18:49:23.506597+00:00"
  }
}
```

### `GET /labels`
Retrieves recorded operator decisions ordered by `created_at DESC`, filtered by active `dataset_id`.
- **Query Params:**
  - `dataset_id` *(optional, string)*: Filter decisions strictly to this dataset ID.
- **Response (200 OK):**
```json
[
  {
    "TransactionID": "TEST_TXN_0001",
    "label": "fraud",
    "operator": "llm_judge_v1",
    "created_at": "2026-11-04T18:49:23.506597+00:00"
  }
]
```

### `GET /labels/export.csv`
Streams recorded labels as standard RFC 4180 CSV attachment (`operator_labels.csv`), filtered by `dataset_id`.
- **Query Params:**
  - `dataset_id` *(optional, string)*: Filter exported labels strictly to this dataset.
- **Response (200 OK, text/csv):**
```csv
dataset_id,TransactionID,label,operator,created_at
default,TEST_TXN_0001,fraud,llm_judge_v1,2026-11-04T18:49:23.506597+00:00
```

### `POST /seed`
Idempotent trigger: if predictions CSV is absent, executes `src/predict.py` on `outputs/predictions/sample_input.csv`.
- **Response (200 OK):** `{"seeded": false, "message": "Predictions checked/seeded."}`

### `GET /datasets`
Lists all available datasets in the current session (default benchmark, original ULB dataset, and user uploads).
- **Response (200 OK):** Array of dataset objects containing `dataset_id`, `filename`, `total_rows`, `is_predicted`, `has_class`.

### `POST /datasets/upload`
Uploads and validates a credit card transactions CSV dataset (strictly 30 numerical features: Time, Amount, V1–V28; optional TransactionID and Class).
- **Form Data:** `file` (multipart CSV file up to 250 MB, <= 350,000 rows)
- **Response (200 OK):** `DatasetUploadResponse` with `dataset_id`, row counts, feature validation status, and 5 preview rows.

### `POST /datasets/use-existing`
Loads the original ULB benchmark dataset (`data/creditcard.csv`, 284,807 transactions).
- **Response (200 OK):** `DatasetUploadResponse` with `dataset_id: "ulb_benchmark"`.

### `POST /datasets/{dataset_id}/predict`
Executes batch inference using the frozen CatBoost model (`models/catboost_model.cbm`) and preprocessor at threshold 0.144.
- **Query Params:** `threshold` (default 0.144)
- **Response (200 OK):** `PredictionResponse` containing summary KPI counts, probability histogram & risk charts, and optional ground-truth metrics (TP, FP, FN, TN, precision, recall, F1, ROC-AUC, PR-AUC) if `Class` is present.

### `GET /datasets/{dataset_id}/results`
Fetches paginated, filtered, and sorted transaction predictions with operator feedback status overlaid.
- **Query Params:** `page`, `page_size`, `status` ('all', 'suspicious', 'not_flagged', 'fraud', 'genuine'), `search`, `sort_by`, `sort_order`.
- **Response (200 OK):** `DatasetResultsResponse`.

### `GET /datasets/{dataset_id}/download`
Streams the complete results CSV attachment containing all original columns plus `fraud_probability`, `selected_threshold`, `selected_model`, `prediction_label`, and `assessment_status`.
- **Response (200 OK, text/csv):** RFC 4180 CSV attachment preserving original row order.

### `GET /datasets/template.csv`
Streams a standard CSV template containing all 30 required model features plus sample rows.
- **Response (200 OK, text/csv):** Downloadable `credit_card_transactions_template.csv`.

---

## 2. Labels Store (Schema & Linking)

- **Storage Location:** SQLite file at `dashboard/data/labels.sqlite3`
- **Table Name:** `operator_labels`
- **DDL Schema:**
  ```sql
  CREATE TABLE IF NOT EXISTS operator_labels (
      transaction_id TEXT PRIMARY KEY,
      label TEXT NOT NULL CHECK(label IN ('fraud', 'genuine', 'CONFIRMED_FRAUD', 'CONFIRMED_GENUINE')),
      operator TEXT NOT NULL,
      created_at TEXT NOT NULL
  );
  ```
- **Transaction Linkage:** `transaction_id` matches the caller's unique `TransactionID` (e.g., `TXN153824` or `TEST_TXN_0001`). Updates use `ON CONFLICT(transaction_id) DO UPDATE` to guarantee idempotent writes.

---

## 3. End-to-End Data Flow

```
[Incoming Transactions]
        │
        ▼
[src/predict.py (CatBoost t*=0.144)]
        │  Exports flagged alerts strictly without ground-truth
        ▼
[outputs/predictions/flagged_transactions.csv]
        │
        ▼
[dashboard/backend/data_loader.py] ◄── Joins Amount & Time by TransactionID
        │                          ◄── Overlays labels from labels.sqlite3
        ▼
[GET /transactions?status=pending]
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
[Human Review Queue]      [LLM-as-a-Judge Module]
   │                               │
   └───────────────┬───────────────┘
                   │ POST /labels {TransactionID, label, operator}
                   ▼
         [dashboard/data/labels.sqlite3] (operator_labels table)
                   │
                   ▼
  [Scheduled Retraining Script & Audit Export]
```

---

## 4. Integration Points for Teammates

### A. LLM-as-a-Judge Integration
1. **Poll Pending Cases:** Call `GET http://localhost:8000/transactions?status=pending`. Each item includes `TransactionID`, `fraud_probability`, `selected_threshold`, `amount`, and `time`.
2. **Feature Hydration:** If PCA components ($V_1$–$V_{28}$) are needed for reasoning prompts, query `data/creditcard.csv` or `outputs/predictions/sample_input.csv` using `TransactionID` (see `ML_MODULE_HANDOFF.md` §4).
3. **Submit Verdict:** Post the assessment to `POST http://localhost:8000/labels`:
   ```json
   {
     "TransactionID": "TEST_TXN_0001",
     "label": "fraud",
     "operator": "llm_judge_gpt4o"
   }
   ```
   *Result:* Transaction immediately transitions out of `status=pending` into `CONFIRMED_FRAUD` / `CONFIRMED_GENUINE`.

### B. Retraining Pipeline Integration
1. **Fetch Training Labels:** Query `GET http://localhost:8000/labels` or directly read SQLite (`dashboard/data/labels.sqlite3`).
2. **What Retraining Page Expects:** The UI displays live feedback counts (`totalCount`, `fraudCount`, `genuineCount`) parsed from `/labels`. Retraining scripts should consume these labels, update training matrices, refit CatBoost weights, and log execution.

---

## 5. Non-Negotiable System Invariants

1. **Predictions Immutability:** `flagged_transactions.csv` is read-only for the dashboard. Operator decisions write **exclusively** to SQLite (`labels.sqlite3`).
2. **Zero Ground-Truth Leakage:** Model inference outputs and `/transactions` records **strictly omit** `Class` or `actual_class`.
3. **Alert Semantics:** Every unadjudicated row is `SUSPICIOUS_PENDING_REVIEW` (elevated statistical risk). It is **never** auto-declared as confirmed fraud.

---

## 6. Not Implemented Yet (Scope Boundaries)

The following capabilities are **not** present in the current codebase:
- **No Authentication / RBAC:** The API has no JWT/OAuth; any client can submit labels.
- **No Direct Retraining Trigger Endpoint:** There is no `POST /retrain` endpoint. Retraining is handled externally by scripts.
- **No LLM Prompt Generator Endpoint:** No `POST /judge` or automated LLM reasoning endpoint exists inside the dashboard backend.
- **No Streaming / WebSocket Bus:** Transactions are polled over HTTP; no Kafka or SSE bus is wired.
- **No Dynamic Threshold Tuning Endpoint:** The threshold $0.144$ is a frozen policy constant; no `PUT /threshold` endpoint exists.
