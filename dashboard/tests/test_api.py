"""Pytest test suite for Operator Dashboard API.
Tests cover:
- GET /transactions (status filtering, probability sorting)
- GET /metrics (dynamic reading of recall, precision, cost, threshold)
- GET /flags-per-hour
- POST /labels and GET /labels
- Strict verification that predictions file is NEVER overwritten.
"""

import os
import hashlib
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.database import get_db_connection, init_db
from dashboard.backend.data_loader import get_active_predictions_file


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with an isolated test SQLite database."""
    test_db = tmp_path / "test_labels.sqlite3"
    monkeypatch.setattr("dashboard.backend.database.DEFAULT_DB_PATH", test_db)
    init_db(test_db)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_metrics(client):
    """Verify metrics are dynamically parsed from real evaluation files and match audited benchmarks."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "CatBoost"
    assert data["decision_threshold"] == 0.144
    # Real test recall: ~0.851351 (85.14%)
    assert pytest.approx(data["recall"], rel=1e-3) == 0.851351
    # Real test precision: ~0.473684 (47.37%)
    assert pytest.approx(data["precision"], rel=1e-3) == 0.473684
    # Real PR-AUC and ROC-AUC
    assert pytest.approx(data["pr_auc"], rel=1e-3) == 0.829777
    assert pytest.approx(data["roc_auc"], rel=1e-3) == 0.971018
    # Real cost: 5850.0
    assert data["total_hypothetical_cost"] == 5850.0
    # Confusion matrix
    assert data["confusion_matrix"]["tp"] == 63
    assert data["confusion_matrix"]["fp"] == 70
    assert data["confusion_matrix"]["fn"] == 11


def test_get_transactions_sorted_by_probability(client):
    """Verify transactions are returned sorted by fraud_probability descending."""
    response = client.get("/transactions")
    assert response.status_code == 200
    txns = response.json()
    assert isinstance(txns, list)
    if txns:
        probs = [t["fraud_probability"] for t in txns]
        assert probs == sorted(probs, reverse=True)
        # Check required schema
        first = txns[0]
        assert "TransactionID" in first
        assert "fraud_probability" in first
        assert "selected_threshold" in first
        assert "assessment_status" in first


def test_flags_per_hour(client):
    """Verify flags-per-hour returns structured hourly counts."""
    response = client.get("/flags-per-hour")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "hour" in data[0]
        assert "flags" in data[0]


def test_post_label_and_status_transition(client):
    """Verify submitting a label moves transaction status from Pending to Fraud/Genuine."""
    # 1. Fetch initial pending transactions
    res_init = client.get("/transactions?status=pending")
    assert res_init.status_code == 200
    pending = res_init.json()
    if not pending:
        pytest.skip("No pending transactions available in test environment.")

    target_id = pending[0]["TransactionID"]

    # 2. Submit label 'fraud'
    res_label = client.post(
        "/labels",
        json={"TransactionID": target_id, "label": "fraud", "operator": "test_analyst"},
    )
    assert res_label.status_code == 200
    data = res_label.json()
    assert data["success"] is True

    # 3. Verify in /labels
    res_labels_list = client.get("/labels")
    assert res_labels_list.status_code == 200
    all_labels = res_labels_list.json()
    matched = [l for l in all_labels if l["TransactionID"] == target_id]
    assert len(matched) == 1
    assert matched[0]["label"] == "fraud"
    assert matched[0]["operator"] == "test_analyst"

    # 4. Verify transaction assessment_status is now CONFIRMED_FRAUD
    res_after = client.get("/transactions")
    assert res_after.status_code == 200
    updated_tx = next(t for t in res_after.json() if t["TransactionID"] == target_id)
    assert updated_tx["assessment_status"] == "CONFIRMED_FRAUD"
    assert updated_tx["operator_label"] == "fraud"

    # 5. Verify it is no longer in pending filter
    res_pending_after = client.get("/transactions?status=pending")
    pending_ids = [t["TransactionID"] for t in res_pending_after.json()]
    assert target_id not in pending_ids

    # 6. Verify it is in fraud filter
    res_fraud_filter = client.get("/transactions?status=fraud")
    fraud_ids = [t["TransactionID"] for t in res_fraud_filter.json()]
    assert target_id in fraud_ids


def test_never_overwrite_predictions_file(client):
    """CRITICAL TEST: Ensure labels never touch, modify, or overwrite predictions CSV."""
    active_pred = get_active_predictions_file()
    if not active_pred or not active_pred.exists():
        pytest.skip("No active predictions file found.")

    # Compute sha256 and get mtime before
    stat_before = active_pred.stat()
    with open(active_pred, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    # Submit a label
    response = client.post(
        "/labels",
        json={"TransactionID": "TEST_AUDIT_TXN", "label": "genuine", "operator": "audit_runner"},
    )
    assert response.status_code == 200

    # Verify predictions file remains completely untouched
    stat_after = active_pred.stat()
    with open(active_pred, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert stat_before.st_mtime == stat_after.st_mtime
    assert hash_before == hash_after


def test_invalid_label_rejected(client):
    """Verify invalid labels are rejected with 400 Bad Request."""
    response = client.post(
        "/labels",
        json={"TransactionID": "TXN_ERR_001", "label": "maybe", "operator": "analyst"},
    )
    assert response.status_code == 400
