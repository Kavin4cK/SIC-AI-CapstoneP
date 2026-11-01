"""
Automated Test Suite for Fraud Detection ML Pipeline & Inference Engine
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Scope: ML Module Verification
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.predict import (
    FraudInferencePipeline,
    REQUIRED_FEATURES,
    EXPECTED_MODEL_FEATURES,
    TRAIN_AMOUNT_MEDIAN,
    TRAIN_AMOUNT_IQR,
    TRAIN_TIME_MEDIAN,
    TRAIN_TIME_IQR,
)


@pytest.fixture(scope="module")
def pipeline():
    """Shared FraudInferencePipeline instance."""
    return FraudInferencePipeline(
        model_path="models/catboost_model.cbm",
        preprocessor_path="models/preprocessor.joblib",
        default_threshold=0.144,
    )


@pytest.fixture
def sample_valid_df():
    """Generates a small valid transactions DataFrame with 5 rows."""
    np.random.seed(42)
    data = {f"V{i}": np.random.randn(5) for i in range(1, 29)}
    data["Amount"] = [10.0, 50.0, 120.0, 0.99, 1500.0]
    data["Time"] = [100.0, 2500.0, 45000.0, 85000.0, 150000.0]
    data["TransactionID"] = [f"TXN_TEST_{i:03d}" for i in range(5)]
    data["Class"] = [0, 0, 1, 0, 1]  # Ground truth should be ignored and excluded
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test 1: Model Loading & Artifact Integrity
# ---------------------------------------------------------------------------
def test_model_loading(pipeline):
    """Verify CatBoost model loads, is fitted, and expects 30 specific features."""
    assert pipeline.model is not None
    assert pipeline.model.is_fitted()
    assert len(pipeline.model.feature_names_) == 30
    assert pipeline.model.feature_names_ == EXPECTED_MODEL_FEATURES


# ---------------------------------------------------------------------------
# Test 2: Preprocessor Consistency & Scaling Parameters
# ---------------------------------------------------------------------------
def test_preprocessor_parameters(pipeline):
    """Verify that RobustScaler parameters strictly match training-data statistics."""
    prep = pipeline.preprocessor
    assert prep.is_fitted_

    # Amount scaler check
    assert hasattr(prep, "amount_scaler")
    assert np.isclose(prep.amount_scaler.center_[0], TRAIN_AMOUNT_MEDIAN, atol=1e-2)
    assert np.isclose(prep.amount_scaler.scale_[0], TRAIN_AMOUNT_IQR, atol=1e-2)

    # Time scaler check
    assert hasattr(prep, "time_scaler")
    assert np.isclose(prep.time_scaler.center_[0], TRAIN_TIME_MEDIAN, atol=1e-1)
    assert np.isclose(prep.time_scaler.scale_[0], TRAIN_TIME_IQR, atol=1e-1)


# ---------------------------------------------------------------------------
# Test 3: Feature Schema Validation & Missing Feature Errors
# ---------------------------------------------------------------------------
def test_feature_validation_success(pipeline, sample_valid_df):
    """Verify validation succeeds when all required features are present."""
    features, ids = pipeline.validate_input(sample_valid_df)
    assert features.shape == (5, 30)
    assert list(features.columns) == REQUIRED_FEATURES
    assert len(ids) == 5


def test_missing_features_raises_error(pipeline, sample_valid_df):
    """Verify that omitting required features raises an informative ValueError."""
    # Drop 'V5' and 'Amount'
    incomplete_df = sample_valid_df.drop(columns=["V5", "Amount"])
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_input(incomplete_df)
    
    err_msg = str(excinfo.value)
    assert "missing required feature columns" in err_msg
    assert "V5" in err_msg
    assert "Amount" in err_msg


def test_empty_dataframe_raises_error(pipeline):
    """Verify that passing an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=REQUIRED_FEATURES)
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_input(empty_df)
    assert "empty" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Test 4: Transaction ID Mapping & Isolation
# ---------------------------------------------------------------------------
def test_transaction_id_preservation(pipeline, sample_valid_df):
    """Verify input TransactionIDs are preserved exactly through inference."""
    results = pipeline.predict(sample_valid_df, return_all=True)
    expected_ids = sample_valid_df["TransactionID"].tolist()
    assert set(results["TransactionID"]).issubset(set(expected_ids))
    assert len(results) == len(expected_ids)


def test_transaction_id_auto_generation(pipeline, sample_valid_df):
    """Verify sequential synthetic TransactionIDs are generated if missing from input."""
    df_no_id = sample_valid_df.drop(columns=["TransactionID"])
    results = pipeline.predict(df_no_id, return_all=True)
    assert "TransactionID" in results.columns
    assert results["TransactionID"].iloc[0].startswith("TXN_INFER_")
    assert len(results["TransactionID"].unique()) == len(sample_valid_df)


def test_id_and_target_exclusion_from_features(pipeline, sample_valid_df):
    """Verify TransactionID and Class are NEVER passed to the preprocessor or model."""
    df_features_raw, _ = pipeline.validate_input(sample_valid_df)
    assert "TransactionID" not in df_features_raw.columns
    assert "Class" not in df_features_raw.columns

    X_infer = pipeline.preprocess(df_features_raw)
    assert X_infer.shape == (5, 30)
    assert "TransactionID" not in X_infer.columns
    assert "Class" not in X_infer.columns
    assert list(X_infer.columns) == EXPECTED_MODEL_FEATURES


# ---------------------------------------------------------------------------
# Test 5: Probability Bounds & Decision Threshold Logic
# ---------------------------------------------------------------------------
def test_probability_bounds(pipeline, sample_valid_df):
    """Verify all predicted fraud probabilities are bounded in [0, 1]."""
    results = pipeline.predict(sample_valid_df, return_all=True)
    probs = results["fraud_probability"].values
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert not np.isnan(probs).any()


def test_threshold_decision_logic(pipeline, sample_valid_df):
    """Verify that records are flagged if and only if probability >= threshold."""
    threshold = 0.144
    results_all = pipeline.predict(sample_valid_df, threshold=threshold, return_all=True)
    results_flagged = pipeline.predict(sample_valid_df, threshold=threshold, return_all=False)

    # In return_all mode, is_suspicious must match probability >= threshold
    for _, row in results_all.iterrows():
        expected_flag = row["fraud_probability"] >= threshold
        assert row["is_suspicious"] == expected_flag
        if expected_flag:
            assert row["assessment_status"] == "SUSPICIOUS_PENDING_REVIEW"
        else:
            assert row["assessment_status"] == "NORMAL"

    # In flagged-only mode, all returned transactions must have probability >= threshold
    assert len(results_flagged) == (results_all["fraud_probability"] >= threshold).sum()
    if len(results_flagged) > 0:
        assert (results_flagged["fraud_probability"] >= threshold).all()
        assert (results_flagged["assessment_status"] == "SUSPICIOUS_PENDING_REVIEW").all()


# ---------------------------------------------------------------------------
# Test 6: Ground Truth Exclusion from LLM Handoff
# ---------------------------------------------------------------------------
def test_ground_truth_exclusion(pipeline, sample_valid_df):
    """Verify output DataFrame strictly excludes any ground-truth or target columns."""
    results_flagged = pipeline.predict(sample_valid_df, return_all=False)
    results_all = pipeline.predict(sample_valid_df, return_all=True)

    forbidden_cols = ["Class", "class", "actual_class", "target", "y_true", "y_test", "ground_truth"]
    for col in forbidden_cols:
        assert col not in results_flagged.columns, f"Forbidden column '{col}' found in flagged output!"
        assert col not in results_all.columns, f"Forbidden column '{col}' found in all-results output!"

    # Verify exact required schema for LLM handoff
    assert list(results_flagged.columns) == [
        "TransactionID",
        "fraud_probability",
        "selected_threshold",
        "selected_model",
        "assessment_status",
    ]


# ---------------------------------------------------------------------------
# Test 7: End-to-End File Inference & CSV Persistence
# ---------------------------------------------------------------------------
def test_end_to_end_file_inference(pipeline, sample_valid_df):
    """Verify inference pipeline executes end-to-end reading from CSV and saving to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_csv = os.path.join(tmp_dir, "input.csv")
        output_csv = os.path.join(tmp_dir, "output_flagged.csv")

        # Save sample input to temporary CSV
        sample_valid_df.to_csv(input_csv, index=False)

        # Run file inference
        df_out = pipeline.predict(input_csv, output_path=output_csv, threshold=0.144)

        # Verify output file exists on disk
        assert os.path.exists(output_csv)

        # Verify output content matches
        df_disk = pd.read_csv(output_csv)
        assert len(df_disk) == len(df_out)
        assert list(df_disk.columns) == [
            "TransactionID",
            "fraud_probability",
            "selected_threshold",
            "selected_model",
            "assessment_status",
        ]
