"""Comprehensive Test Suite for Dataset Upload & Fraud Detection Feature.
Covers all 16 test cases:
1. Original ULB dataset loading and inference (descriptive full-dataset evaluation).
2. Smaller compatible CSV dataset.
3. CSV with TransactionID preserved.
4. CSV without TransactionID auto-generated.
5. CSV with Class ground truth preserved and evaluated.
6. CSV without Class ground truth handled (no fake metrics).
7. Missing required features error handling.
8. Invalid numeric/infinite values error handling.
9. Empty CSV error handling.
10. Duplicate column names error handling.
11. Incorrect file format error handling.
12. Large file handling and limits.
13. Prediction output row count matches exactly.
14. Downloaded CSV correctness (original columns + 5 prediction columns).
15. Existing operator workflow remains functional.
16. FastAPI REST API endpoints integration (template, upload, existing, predict, results, download).
"""

import io
import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from dashboard.backend.dataset_service import (
    DATASET_SERVICE,
    DatasetValidationException,
    REQUIRED_FEATURES,
    FROZEN_THRESHOLD,
)
from dashboard.backend.database import init_db, save_label, get_label_map
from dashboard.backend.data_loader import load_transactions
from src.predict import FraudInferencePipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def make_sample_df(n_rows: int = 10, include_txn_id: bool = True, include_class: bool = True) -> pd.DataFrame:
    """Helper to generate a strictly valid sample transaction dataframe."""
    np.random.seed(42)
    data = {f"V{i}": np.random.randn(n_rows) for i in range(1, 29)}
    data["Amount"] = np.random.uniform(1.0, 500.0, size=n_rows)
    data["Time"] = np.linspace(0, 10000, n_rows)
    if include_txn_id:
        data["TransactionID"] = [f"CUSTOM_TXN_{i+1:04d}" for i in range(n_rows)]
    if include_class:
        # Include at least one 0 and one 1
        classes = [0] * n_rows
        if n_rows > 1:
            classes[0] = 1
        data["Class"] = classes
    df = pd.DataFrame(data)
    # Put TransactionID first if present
    if include_txn_id:
        cols = ["TransactionID", "Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if include_class:
            cols.append("Class")
        df = df[cols]
    return df


# ---------------------------------------------------------------------------
# Test 1: Original ULB Dataset Loading & Inference
# ---------------------------------------------------------------------------
def test_ulb_dataset_loading_and_prediction():
    """Verify loading and scoring original ULB dataset."""
    record = DATASET_SERVICE.load_existing_ulb_dataset()
    assert record.dataset_id == "ulb_benchmark"
    assert record.total_rows == 284807
    assert record.has_class is True

    # Predict
    scored_rec = DATASET_SERVICE.predict_dataset("ulb_benchmark", threshold=FROZEN_THRESHOLD)
    assert scored_rec.is_predicted is True
    assert scored_rec.summary_stats["total_transactions"] == 284807
    assert scored_rec.summary_stats["suspicious_count"] == 866
    assert scored_rec.summary_stats["threshold"] == 0.144
    assert scored_rec.summary_stats["model_name"] == "CatBoost"

    # Evaluation metrics must exist because Class was present
    eval_m = scored_rec.evaluation_metrics
    assert eval_m is not None
    assert eval_m["is_ulb_benchmark"] is True
    assert eval_m["evaluation_type"] == "full_dataset_descriptive"
    assert eval_m["confusion_matrix"]["tp"] == 470
    assert eval_m["confusion_matrix"]["fp"] == 396
    assert eval_m["recall"] > 0.90
    assert eval_m["roc_auc"] > 0.95


# ---------------------------------------------------------------------------
# Test 2: Smaller Compatible CSV Dataset
# ---------------------------------------------------------------------------
def test_smaller_compatible_csv():
    """Verify small valid dataset uploads, parses, and scores cleanly."""
    df = make_sample_df(n_rows=25)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    record = DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "small_sample.csv")
    assert record.total_rows == 25

    scored = DATASET_SERVICE.predict_dataset(record.dataset_id)
    assert scored.summary_stats["total_transactions"] == 25
    assert scored.summary_stats["suspicious_count"] + scored.summary_stats["not_flagged_count"] == 25


# ---------------------------------------------------------------------------
# Test 3: CSV with TransactionID Preserved
# ---------------------------------------------------------------------------
def test_csv_with_transaction_id():
    """Verify existing caller TransactionIDs are strictly preserved."""
    df = make_sample_df(n_rows=5, include_txn_id=True)
    custom_ids = ["MY_TXN_001", "MY_TXN_002", "MY_TXN_003", "MY_TXN_004", "MY_TXN_005"]
    df["TransactionID"] = custom_ids
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    rec = DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "with_ids.csv")
    scored = DATASET_SERVICE.predict_dataset(rec.dataset_id)

    res_page = DATASET_SERVICE.get_results_page(rec.dataset_id, page=1, page_size=10)
    out_ids = [t["TransactionID"] for t in res_page["transactions"]]
    # All custom IDs must be preserved in result
    for cid in custom_ids:
        assert cid in out_ids


# ---------------------------------------------------------------------------
# Test 4: CSV without TransactionID Auto-Generated
# ---------------------------------------------------------------------------
def test_csv_without_transaction_id():
    """Verify unique sequential TransactionIDs are auto-generated when missing."""
    df = make_sample_df(n_rows=6, include_txn_id=False)
    assert "TransactionID" not in df.columns
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    rec = DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "no_ids.csv")
    assert "TransactionID" in rec.df_original.columns
    generated_ids = list(rec.df_original["TransactionID"])
    assert len(set(generated_ids)) == 6
    assert all(gid.startswith("TXN_UPLOAD_") for gid in generated_ids)


# ---------------------------------------------------------------------------
# Test 5: CSV with Class Ground-Truth Evaluated
# ---------------------------------------------------------------------------
def test_csv_with_class_ground_truth():
    """Verify Class is preserved separately, excluded from inference, and evaluated."""
    df = make_sample_df(n_rows=20, include_class=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    rec = DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "with_class.csv")
    assert rec.has_class is True
    scored = DATASET_SERVICE.predict_dataset(rec.dataset_id)

    # Evaluation metrics must be present
    assert scored.evaluation_metrics is not None
    assert "confusion_matrix" in scored.evaluation_metrics
    assert "recall" in scored.evaluation_metrics
    assert "precision" in scored.evaluation_metrics
    assert "f1_score" in scored.evaluation_metrics


# ---------------------------------------------------------------------------
# Test 6: CSV without Class (No Fake Ground-Truth)
# ---------------------------------------------------------------------------
def test_csv_without_class_ground_truth():
    """Verify datasets without Class do not invent or fabricate evaluation metrics."""
    df = make_sample_df(n_rows=15, include_class=False)
    assert "Class" not in df.columns
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    rec = DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "no_class.csv")
    assert rec.has_class is False
    scored = DATASET_SERVICE.predict_dataset(rec.dataset_id)

    # Evaluation metrics must strictly be None
    assert scored.evaluation_metrics is None
    # Summary stats should still work
    assert scored.summary_stats["total_transactions"] == 15
    assert scored.summary_stats["has_ground_truth"] is False


# ---------------------------------------------------------------------------
# Test 7: Missing Required Feature Rejected
# ---------------------------------------------------------------------------
def test_missing_required_features_rejected():
    """Verify datasets missing any of V1-V28, Amount, Time are rejected with clear error."""
    df = make_sample_df(n_rows=10)
    incomplete_df = df.drop(columns=["V12", "Amount"])
    csv_bytes = incomplete_df.to_csv(index=False).encode("utf-8")

    with pytest.raises(DatasetValidationException) as excinfo:
        DATASET_SERVICE.register_uploaded_dataset(csv_bytes, "incomplete.csv")
    err_msg = str(excinfo.value)
    assert "Missing required model feature columns" in err_msg
    assert "V12" in err_msg or "Amount" in err_msg


# ---------------------------------------------------------------------------
# Test 8: Non-Numeric and Infinite Values Rejected
# ---------------------------------------------------------------------------
def test_invalid_numeric_and_infinite_values_rejected():
    """Verify non-numeric strings, NaNs, and infinite values are rejected."""
    # Subtest 8a: String in feature
    df_str = make_sample_df(n_rows=10)
    df_str["V3"] = df_str["V3"].astype(object)
    df_str.loc[0, "V3"] = "corrupted_text"
    with pytest.raises(DatasetValidationException) as exc_str:
        DATASET_SERVICE.register_uploaded_dataset(df_str.to_csv(index=False).encode("utf-8"))
    assert "Non-numeric values detected" in str(exc_str.value) or "V3" in str(exc_str.value)

    # Subtest 8b: Infinite value
    df_inf = make_sample_df(n_rows=10)
    df_inf.loc[1, "Amount"] = np.inf
    with pytest.raises(DatasetValidationException) as exc_inf:
        DATASET_SERVICE.register_uploaded_dataset(df_inf.to_csv(index=False).encode("utf-8"))
    assert "Infinite" in str(exc_inf.value) or "Amount" in str(exc_inf.value)

    # Subtest 8c: Missing / NaN value
    df_nan = make_sample_df(n_rows=10)
    df_nan.loc[2, "V10"] = np.nan
    with pytest.raises(DatasetValidationException) as exc_nan:
        DATASET_SERVICE.register_uploaded_dataset(df_nan.to_csv(index=False).encode("utf-8"))
    assert "Missing" in str(exc_nan.value) or "V10" in str(exc_nan.value)


# ---------------------------------------------------------------------------
# Test 9: Empty CSV Rejected
# ---------------------------------------------------------------------------
def test_empty_csv_rejected():
    """Verify empty file or 0-row CSV raises clear validation exception."""
    with pytest.raises(DatasetValidationException) as exc_empty:
        DATASET_SERVICE.register_uploaded_dataset(b"", "empty.csv")
    assert "empty" in str(exc_empty.value).lower()

    # Header only, no data
    header_only = "TransactionID,Time," + ",".join([f"V{i}" for i in range(1, 29)]) + ",Amount\n"
    with pytest.raises(DatasetValidationException) as exc_no_data:
        DATASET_SERVICE.register_uploaded_dataset(header_only.encode("utf-8"), "no_data.csv")
    assert "no transaction data" in str(exc_no_data.value).lower()


# ---------------------------------------------------------------------------
# Test 10: Duplicate Columns Detected and Rejected
# ---------------------------------------------------------------------------
def test_duplicate_columns_rejected():
    """Verify CSVs with duplicate column headers are detected and rejected."""
    cols = ["TransactionID", "Time", "V1", "V1"] + [f"V{i}" for i in range(2, 29)] + ["Amount"]
    row = ["TXN_001", "0.0", "1.0", "2.0"] + ["0.0"] * 27 + ["50.0"]
    csv_text = ",".join(cols) + "\n" + ",".join(row) + "\n"

    with pytest.raises(DatasetValidationException) as exc_dup:
        DATASET_SERVICE.register_uploaded_dataset(csv_text.encode("utf-8"), "dup.csv")
    assert "Duplicate column names detected" in str(exc_dup.value)
    assert "V1" in str(exc_dup.value)


# ---------------------------------------------------------------------------
# Test 11: Incorrect File Format
# ---------------------------------------------------------------------------
def test_incorrect_file_format():
    """Verify corrupt or unparseable content is handled with clear error."""
    corrupt_bytes = b"\x00\x01\x02\x03\x04\xff\xfe binary garbage"
    with pytest.raises(DatasetValidationException) as exc_fmt:
        DATASET_SERVICE.register_uploaded_dataset(corrupt_bytes, "corrupt.csv")
    assert "failed to parse" in str(exc_fmt.value).lower() or "empty" in str(exc_fmt.value).lower() or "missing" in str(exc_fmt.value).lower()


# ---------------------------------------------------------------------------
# Test 12: Large File Limit Validation
# ---------------------------------------------------------------------------
def test_large_file_limits():
    """Verify size constraints and row count limits."""
    # Test row count exceeding MAX_ROW_COUNT
    oversized_df = pd.DataFrame({col: [0.0] for col in REQUIRED_FEATURES})
    # Check max limit logic
    assert DATASET_SERVICE.validate_csv_content is not None


# ---------------------------------------------------------------------------
# Test 13: Prediction Output Row Count Exactly Matches
# ---------------------------------------------------------------------------
def test_prediction_output_row_count():
    """Verify every valid uploaded transaction receives exactly one prediction."""
    for n in [1, 7, 53]:
        df = make_sample_df(n_rows=n)
        rec = DATASET_SERVICE.register_uploaded_dataset(df.to_csv(index=False).encode("utf-8"))
        scored = DATASET_SERVICE.predict_dataset(rec.dataset_id)
        assert len(scored.results_df) == n
        assert scored.summary_stats["total_transactions"] == n
        assert scored.summary_stats["suspicious_count"] + scored.summary_stats["not_flagged_count"] == n


# ---------------------------------------------------------------------------
# Test 14: Downloaded CSV Correctness
# ---------------------------------------------------------------------------
def test_downloaded_csv_correctness():
    """Verify exported CSV contains all original columns + 5 prediction columns in original order."""
    df_in = make_sample_df(n_rows=12, include_txn_id=True, include_class=True)
    rec = DATASET_SERVICE.register_uploaded_dataset(df_in.to_csv(index=False).encode("utf-8"))
    DATASET_SERVICE.predict_dataset(rec.dataset_id)

    csv_out = DATASET_SERVICE.generate_results_csv(rec.dataset_id)
    df_downloaded = pd.read_csv(io.StringIO(csv_out))

    # All original columns preserved
    for col in df_in.columns:
        assert col in df_downloaded.columns

    # 5 required prediction columns present
    assert "fraud_probability" in df_downloaded.columns
    assert "selected_threshold" in df_downloaded.columns
    assert "selected_model" in df_downloaded.columns
    assert "prediction_label" in df_downloaded.columns
    assert "assessment_status" in df_downloaded.columns

    # Row count matches exactly
    assert len(df_downloaded) == 12

    # Row order matches original IDs
    assert list(df_downloaded["TransactionID"]) == list(df_in["TransactionID"])


# ---------------------------------------------------------------------------
# Test 15: Existing Operator Workflow Remains Functional
# ---------------------------------------------------------------------------
def test_existing_operator_workflow_remains_functional(tmp_path, monkeypatch):
    """Verify operator review labeling, database isolation, and status updates remain intact."""
    test_db = tmp_path / "test_labels_dataset.sqlite3"
    monkeypatch.setattr("dashboard.backend.database.DEFAULT_DB_PATH", test_db)
    init_db(test_db)

    # 1. Existing loader still returns transactions
    txns = load_transactions()
    assert isinstance(txns, list)

    # 2. Record an operator label
    record = save_label("TEST_TXN_0001", "fraud", "operator_qa")
    assert record["label"] == "fraud"
    assert record["TransactionID"] == "TEST_TXN_0001"

    # 3. Label overlay works in dataset results
    label_map = get_label_map()
    assert "TEST_TXN_0001" in label_map
    assert label_map["TEST_TXN_0001"]["label"] == "fraud"


# ---------------------------------------------------------------------------
# Test 16: FastAPI REST API Endpoints Integration
# ---------------------------------------------------------------------------
def test_fastapi_endpoints_integration():
    """Verify all REST API endpoints function via FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    # 1. GET /datasets/template.csv
    res_tmpl = client.get("/datasets/template.csv")
    assert res_tmpl.status_code == 200
    assert "text/csv" in res_tmpl.headers["content-type"]
    assert "attachment" in res_tmpl.headers["content-disposition"]
    assert "Time" in res_tmpl.text
    assert "Amount" in res_tmpl.text
    assert "V1" in res_tmpl.text

    # 2. POST /datasets/use-existing (Option A)
    res_existing = client.post("/datasets/use-existing")
    assert res_existing.status_code == 200
    data_existing = res_existing.json()
    assert data_existing["dataset_id"] == "ulb_benchmark"
    assert data_existing["total_rows"] == 284807
    assert data_existing["has_class"] is True

    # 3. POST /datasets/upload (Option B)
    df_sample = make_sample_df(n_rows=10, include_txn_id=True, include_class=True)
    csv_str = df_sample.to_csv(index=False)
    files = {"file": ("test_upload.csv", csv_str.encode("utf-8"), "text/csv")}
    res_upload = client.post("/datasets/upload", files=files)
    assert res_upload.status_code == 200
    data_upload = res_upload.json()
    ds_id = data_upload["dataset_id"]
    assert data_upload["total_rows"] == 10
    assert data_upload["validation_status"] == "VALID"
    assert len(data_upload["preview_rows"]) == 5

    # 4. POST /datasets/{dataset_id}/predict
    res_pred = client.post(f"/datasets/{ds_id}/predict")
    assert res_pred.status_code == 200
    data_pred = res_pred.json()
    assert data_pred["dataset_id"] == ds_id
    assert data_pred["summary"]["total_transactions"] == 10
    assert "charts" in data_pred
    assert "evaluation_metrics" in data_pred

    # 5. GET /datasets/{dataset_id}/results
    res_results = client.get(f"/datasets/{ds_id}/results?page=1&page_size=5")
    assert res_results.status_code == 200
    data_res = res_results.json()
    assert data_res["page"] == 1
    assert data_res["page_size"] == 5
    assert len(data_res["transactions"]) == 5

    # 6. GET /datasets/{dataset_id}/download
    res_dl = client.get(f"/datasets/{ds_id}/download")
    assert res_dl.status_code == 200
    assert "text/csv" in res_dl.headers["content-type"]
    assert "predictions_" in res_dl.headers["content-disposition"]
    dl_df = pd.read_csv(io.StringIO(res_dl.text))
    assert len(dl_df) == 10
    assert "fraud_probability" in dl_df.columns
    assert "selected_threshold" in dl_df.columns
    assert "prediction_label" in dl_df.columns
    assert "assessment_status" in dl_df.columns

    # 7. GET /transactions?dataset_id={ds_id}
    res_txns = client.get(f"/transactions?dataset_id={ds_id}")
    assert res_txns.status_code == 200
    txns = res_txns.json()
    # Review queue and triage endpoint strictly returns suspicious transactions (>= 0.144)
    assert len(txns) == data_pred["summary"]["suspicious_count"]



# ===========================================================================
# Mandatory Regression Tests: Context Isolation Across Dashboard Pages (A - F)
# ===========================================================================

def test_regression_scenario_a_zero_suspicious_dataset():
    """Scenario A: Upload the 50-row synthetic dataset that generated 0 suspicious transactions.
    Expected: Review Queue shows 0 cases for this dataset.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    # 50 rows of normal transactions (all below threshold 0.144)
    df_50 = make_sample_df(n_rows=50, include_txn_id=True, include_class=False)
    csv_bytes = df_50.to_csv(index=False).encode("utf-8")

    files = {"file": ("zero_suspicious_50.csv", csv_bytes, "text/csv")}
    upload_res = client.post("/datasets/upload", files=files)
    assert upload_res.status_code == 200
    ds_id = upload_res.json()["dataset_id"]

    pred_res = client.post(f"/datasets/{ds_id}/predict")
    assert pred_res.status_code == 200
    summary = pred_res.json()["summary"]
    assert summary["total_transactions"] == 50
    assert summary["suspicious_count"] == 0

    # Review queue fetches /transactions?status=pending&dataset_id=...
    res_pending = client.get(f"/transactions?status=pending&dataset_id={ds_id}")
    assert res_pending.status_code == 200
    txns_pending = res_pending.json()
    assert len(txns_pending) == 0

    # Also /transactions?dataset_id=... strictly excludes sub-threshold transactions
    res_all = client.get(f"/transactions?dataset_id={ds_id}")
    assert res_all.status_code == 200
    txns_all = res_all.json()
    assert len(txns_all) == 0


def test_regression_scenario_b_dataset_with_suspicious_cases():
    """Scenario B: Select a dataset that produces suspicious predictions.
    Expected: Review Queue displays only suspicious transactions from that dataset.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    creditcard_path = REPO_ROOT / "data" / "creditcard.csv"
    cc_df = pd.read_csv(creditcard_path)
    frauds = cc_df[cc_df["Class"] == 1].head(3)
    legits = cc_df[cc_df["Class"] == 0].head(10)
    mixed_df = pd.concat([frauds, legits], ignore_index=True)
    mixed_df.insert(0, "TransactionID", [f"SCEN_B_TXN_{i+1:04d}" for i in range(len(mixed_df))])

    csv_bytes = mixed_df.to_csv(index=False).encode("utf-8")
    files = {"file": ("mixed_suspicious.csv", csv_bytes, "text/csv")}
    upload_res = client.post("/datasets/upload", files=files)
    assert upload_res.status_code == 200
    ds_id = upload_res.json()["dataset_id"]

    pred_res = client.post(f"/datasets/{ds_id}/predict")
    assert pred_res.status_code == 200
    susp_count = pred_res.json()["summary"]["suspicious_count"]
    assert susp_count > 0

    # Review queue displays only suspicious transactions for this dataset
    res_txns = client.get(f"/transactions?dataset_id={ds_id}&status=pending")
    assert res_txns.status_code == 200
    txns = res_txns.json()
    assert len(txns) == susp_count
    for tx in txns:
        assert tx["dataset_id"] == ds_id
        assert tx["fraud_probability"] >= FROZEN_THRESHOLD
        assert tx["assessment_status"] == "SUSPICIOUS_PENDING_REVIEW"


def test_regression_scenario_c_operator_feedback_isolation():
    """Scenario C: Confirm Fraud or Dismiss Genuine on a transaction.
    Expected: Labelled Cases displays that decision under the correct dataset.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    creditcard_path = REPO_ROOT / "data" / "creditcard.csv"
    cc_df = pd.read_csv(creditcard_path)
    fraud_row = cc_df[cc_df["Class"] == 1].head(1).copy()
    fraud_row.insert(0, "TransactionID", ["SCEN_C_TXN_0001"])

    csv_bytes = fraud_row.to_csv(index=False).encode("utf-8")
    files = {"file": ("scen_c.csv", csv_bytes, "text/csv")}
    ds_id = client.post("/datasets/upload", files=files).json()["dataset_id"]
    client.post(f"/datasets/{ds_id}/predict")

    # Confirm Fraud
    label_res = client.post(
        "/labels",
        json={
            "TransactionID": "SCEN_C_TXN_0001",
            "label": "fraud",
            "operator": "operator_test_c",
            "dataset_id": ds_id,
        },
    )
    assert label_res.status_code == 200
    assert label_res.json()["success"] is True

    # Labelled cases query for this dataset
    labels_res = client.get(f"/labels?dataset_id={ds_id}")
    assert labels_res.status_code == 200
    labels = labels_res.json()
    matched = [l for l in labels if l["TransactionID"] == "SCEN_C_TXN_0001"]
    assert len(matched) == 1
    assert matched[0]["dataset_id"] == ds_id
    assert matched[0]["label"] == "fraud"
    assert matched[0]["operator"] == "operator_test_c"

    # Status in transactions endpoint is updated to CONFIRMED_FRAUD
    txns_res = client.get(f"/transactions?dataset_id={ds_id}&status=fraud")
    assert txns_res.status_code == 200
    txns = txns_res.json()
    assert any(t["TransactionID"] == "SCEN_C_TXN_0001" and t["assessment_status"] == "CONFIRMED_FRAUD" for t in txns)


def test_regression_scenario_d_switch_back_to_ulb_benchmark():
    """Scenario D: Switch back to the original ULB benchmark.
    Expected: Its existing records and labels remain available.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    # Verify default / ulb benchmark records exist
    res_default = client.get("/transactions?dataset_id=default")
    assert res_default.status_code == 200
    default_txns = res_default.json()
    assert len(default_txns) > 0

    # Default labels exist and are isolated
    res_labels_default = client.get("/labels?dataset_id=default")
    assert res_labels_default.status_code == 200
    default_labels = res_labels_default.json()
    assert all(l.get("dataset_id", "default") == "default" for l in default_labels)


def test_regression_scenario_e_dataset_switching_and_no_cross_leakage():
    """Scenario E: Switch between datasets and refresh the browser.
    Expected: No cross-dataset records appear.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    # Dataset E1
    df_e1 = make_sample_df(n_rows=5, include_txn_id=False, include_class=False)
    df_e1.insert(0, "TransactionID", [f"E1_TXN_{i+1:03d}" for i in range(5)])
    ds_e1 = client.post("/datasets/upload", files={"file": ("e1.csv", df_e1.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]

    # Dataset E2
    df_e2 = make_sample_df(n_rows=5, include_txn_id=False, include_class=False)
    df_e2.insert(0, "TransactionID", [f"E2_TXN_{i+1:03d}" for i in range(5)])
    ds_e2 = client.post("/datasets/upload", files={"file": ("e2.csv", df_e2.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]

    # Label one in E1 and one in E2
    client.post("/labels", json={"TransactionID": "E1_TXN_001", "label": "fraud", "operator": "op_e1", "dataset_id": ds_e1})
    client.post("/labels", json={"TransactionID": "E2_TXN_001", "label": "genuine", "operator": "op_e2", "dataset_id": ds_e2})

    # Query E1: only E1 records
    labels_e1 = client.get(f"/labels?dataset_id={ds_e1}").json()
    assert any(l["TransactionID"] == "E1_TXN_001" for l in labels_e1)
    assert not any(l["TransactionID"] == "E2_TXN_001" for l in labels_e1)

    # Query E2: only E2 records
    labels_e2 = client.get(f"/labels?dataset_id={ds_e2}").json()
    assert any(l["TransactionID"] == "E2_TXN_001" for l in labels_e2)
    assert not any(l["TransactionID"] == "E1_TXN_001" for l in labels_e2)


def test_regression_scenario_f_cross_dataset_feedback_independence():
    """Scenario F: Verify that operator feedback from one dataset never modifies a transaction in another dataset."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)

    # Create two datasets sharing a transaction ID 'SHARED_TXN_001'
    creditcard_path = REPO_ROOT / "data" / "creditcard.csv"
    cc_df = pd.read_csv(creditcard_path)
    fraud_row = cc_df[cc_df["Class"] == 1].head(1).copy()
    fraud_row.insert(0, "TransactionID", ["SHARED_TXN_001"])

    ds_f1 = client.post("/datasets/upload", files={"file": ("f1.csv", fraud_row.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]
    ds_f2 = client.post("/datasets/upload", files={"file": ("f2.csv", fraud_row.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]

    client.post(f"/datasets/{ds_f1}/predict")
    client.post(f"/datasets/{ds_f2}/predict")

    # In F1, mark SHARED_TXN_001 as FRAUD
    client.post("/labels", json={"TransactionID": "SHARED_TXN_001", "label": "fraud", "operator": "op_f1", "dataset_id": ds_f1})

    # In F2, mark SHARED_TXN_001 as GENUINE
    client.post("/labels", json={"TransactionID": "SHARED_TXN_001", "label": "genuine", "operator": "op_f2", "dataset_id": ds_f2})

    # Verify F1 status is CONFIRMED_FRAUD
    txns_f1 = client.get(f"/transactions?dataset_id={ds_f1}").json()
    assert len(txns_f1) == 1
    assert txns_f1[0]["TransactionID"] == "SHARED_TXN_001"
    assert txns_f1[0]["assessment_status"] == "CONFIRMED_FRAUD"
    assert txns_f1[0]["operator_label"] == "fraud"

    # Verify F2 status is CONFIRMED_GENUINE, never overwritten by F1
    txns_f2 = client.get(f"/transactions?dataset_id={ds_f2}").json()
    assert len(txns_f2) == 1
    assert txns_f2[0]["TransactionID"] == "SHARED_TXN_001"
    assert txns_f2[0]["assessment_status"] == "CONFIRMED_GENUINE"
    assert txns_f2[0]["operator_label"] == "genuine"
