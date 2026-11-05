"""Dataset Ingestion, Validation, Inference, and Evaluation Service.
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop

Handles:
- Option A: Loading existing ULB benchmark dataset (data/creditcard.csv)
- Option B: Custom CSV upload with strict schema validation
- Default benchmark dataset (outputs/predictions/sample_input.csv)
- Inference using existing FraudInferencePipeline (models/catboost_model.cbm)
- Ground-truth evaluation if Class column is present
- Multi-dataset isolation across predictions, results, and operator labels
- Structured summary metrics and charts computation
- Paginated results table with operator label overlay
- CSV export preserving original columns, row order, and IDs
- Downloadable CSV template generation
"""

import io
import os
import sys
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.predict import (
    FraudInferencePipeline,
    REQUIRED_FEATURES,
    EXPECTED_MODEL_FEATURES,
)
from dashboard.backend.database import get_label_map

# Repository root and key paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CREDITCARD_CSV = REPO_ROOT / "data" / "creditcard.csv"
SAMPLE_INPUT_CSV = REPO_ROOT / "outputs" / "predictions" / "sample_input.csv"
FLAGGED_PRIMARY = REPO_ROOT / "outputs" / "predictions" / "flagged_transactions.csv"
MODEL_PATH = REPO_ROOT / "models" / "catboost_model.cbm"
PREPROCESSOR_PATH = REPO_ROOT / "models" / "preprocessor.joblib"

# Operational constraints
FROZEN_THRESHOLD: float = 0.144
MAX_UPLOAD_SIZE_BYTES: int = 250 * 1024 * 1024  # 250 MB
MAX_ROW_COUNT: int = 350000


class DatasetValidationException(ValueError):
    """Raised when an uploaded dataset violates compatibility requirements."""
    pass


class DatasetRecord:
    """In-memory cache record for a validated dataset and its inference results."""

    def __init__(
        self,
        dataset_id: str,
        filename: str,
        df_original: pd.DataFrame,
        has_class: bool,
        has_transaction_id: bool,
    ) -> None:
        self.dataset_id = dataset_id
        self.filename = filename
        self.df_original = df_original
        self.has_class = has_class
        self.has_transaction_id = has_transaction_id
        self.total_rows = len(df_original)
        self.columns = list(df_original.columns)

        # Populated after prediction
        self.is_predicted: bool = False
        self.results_df: Optional[pd.DataFrame] = None
        self.summary_stats: Optional[Dict[str, Any]] = None
        self.charts_data: Optional[Dict[str, Any]] = None
        self.evaluation_metrics: Optional[Dict[str, Any]] = None


class DatasetService:
    """Thread-safe dataset storage, validation, and inference service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._datasets: Dict[str, DatasetRecord] = {}
        self._pipeline: Optional[FraudInferencePipeline] = None

    def _get_pipeline(self) -> FraudInferencePipeline:
        """Lazily initialize and cache the FraudInferencePipeline singleton."""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    if not MODEL_PATH.exists():
                        raise FileNotFoundError(f"CatBoost model artifact not found at {MODEL_PATH}")
                    self._pipeline = FraudInferencePipeline(
                        model_path=str(MODEL_PATH),
                        preprocessor_path=str(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None,
                        default_threshold=FROZEN_THRESHOLD,
                    )
        return self._pipeline

    def validate_csv_content(
        self,
        file_bytes: bytes,
        filename: str = "upload.csv",
    ) -> pd.DataFrame:
        """Strictly validate uploaded CSV data against all 30 model requirements."""
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise DatasetValidationException("Uploaded CSV file is empty. Please provide a valid transaction dataset.")

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise DatasetValidationException(
                f"File size exceeds maximum allowed limit ({MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB)."
            )

        # 1. Check duplicate headers by inspecting raw header text
        try:
            content_sample = file_bytes[:16384].decode("utf-8", errors="replace")
        except Exception:
            raise DatasetValidationException("Could not decode file content as UTF-8.")

        lines = content_sample.splitlines()
        if not lines:
            raise DatasetValidationException("Uploaded file contains no lines.")

        raw_header = lines[0]
        header_cols = [c.strip() for c in raw_header.split(",")]
        if not header_cols or all(c == "" for c in header_cols):
            raise DatasetValidationException("CSV header is empty or malformed.")

        # Duplicate header check
        seen_headers = set()
        duplicates = []
        for h in header_cols:
            if h in seen_headers:
                duplicates.append(h)
            seen_headers.add(h)
        if duplicates:
            raise DatasetValidationException(
                f"Duplicate column names detected: {sorted(list(set(duplicates)))}. Every column must have a unique name."
            )

        # 2. Parse CSV into DataFrame
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise DatasetValidationException(f"Failed to parse CSV file: {str(e)}")

        if df.empty:
            raise DatasetValidationException("CSV file contains headers but no transaction data rows.")

        if len(df) > MAX_ROW_COUNT:
            raise DatasetValidationException(
                f"Dataset row count ({len(df):,}) exceeds maximum allowed limit ({MAX_ROW_COUNT:,})."
            )

        # 3. Verify all 30 required model features are present
        missing_features = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing_features:
            raise DatasetValidationException(
                f"Missing required model feature columns: {missing_features}. "
                f"The trained CatBoost model strictly requires 30 features: Time, Amount, V1 through V28."
            )

        # 4. Strict check for non-numeric, null, or infinite values in required features
        cols_with_nulls = []
        cols_with_inf = []
        cols_non_numeric = []

        for col in REQUIRED_FEATURES:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                coerced = pd.to_numeric(series, errors="coerce")
                if coerced.isna().any():
                    cols_non_numeric.append(col)
                    continue
                df[col] = coerced

            if df[col].isna().any():
                cols_with_nulls.append(col)

            if np.isinf(df[col].to_numpy(dtype=float, na_value=np.nan)).any():
                cols_with_inf.append(col)

        if cols_non_numeric:
            raise DatasetValidationException(
                f"Non-numeric values detected in feature columns: {cols_non_numeric}. "
                "All 30 model features must be numerical."
            )

        if cols_with_nulls:
            raise DatasetValidationException(
                f"Missing (NaN/null) values detected in feature columns: {cols_with_nulls}. "
                "Silent replacement of missing features is strictly disabled. Please supply complete data."
            )

        if cols_with_inf:
            raise DatasetValidationException(
                f"Infinite (inf/-inf) values detected in feature columns: {cols_with_inf}. "
                "All feature values must be finite numerical values."
            )

        # 5. Handle TransactionID
        if "TransactionID" in df.columns:
            df["TransactionID"] = df["TransactionID"].astype(str)
        else:
            df.insert(
                0,
                "TransactionID",
                [f"TXN_UPLOAD_{i+1:06d}" for i in range(len(df))],
            )

        # 6. Validate Class column if present
        if "Class" in df.columns:
            if df["Class"].isna().any():
                raise DatasetValidationException(
                    "Optional 'Class' ground-truth column contains null values. Ground-truth must be complete (0 or 1)."
                )
            unique_classes = set(df["Class"].unique())
            try:
                df["Class"] = df["Class"].astype(int)
            except Exception:
                raise DatasetValidationException(
                    f"Optional 'Class' ground-truth column contains non-integer labels: {unique_classes}. Expected binary 0 (genuine) or 1 (fraud)."
                )

        return df

    def register_uploaded_dataset(
        self,
        file_bytes: bytes,
        filename: str = "upload.csv",
    ) -> DatasetRecord:
        """Validate and register a user-uploaded CSV dataset with a unique dataset_id."""
        df = self.validate_csv_content(file_bytes, filename)
        has_class = "Class" in df.columns
        has_txn_id = "TransactionID" in df.columns

        dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            df_original=df,
            has_class=has_class,
            has_transaction_id=has_txn_id,
        )

        with self._lock:
            self._datasets[dataset_id] = record

        return record

    def load_existing_ulb_dataset(self) -> DatasetRecord:
        """Option A: Load the existing ULB credit card fraud detection dataset."""
        dataset_id = "ulb_benchmark"

        with self._lock:
            if dataset_id in self._datasets:
                return self._datasets[dataset_id]

        if not CREDITCARD_CSV.exists():
            raise FileNotFoundError(f"Original ULB dataset not found at: {CREDITCARD_CSV}")

        df = pd.read_csv(CREDITCARD_CSV)
        if "TransactionID" not in df.columns:
            df.insert(0, "TransactionID", [f"TXN{i+1:06d}" for i in range(len(df))])

        record = DatasetRecord(
            dataset_id=dataset_id,
            filename="creditcard.csv (Original ULB Dataset)",
            df_original=df,
            has_class="Class" in df.columns,
            has_transaction_id=True,
        )

        with self._lock:
            self._datasets[dataset_id] = record

        return record

    def load_default_dataset(self) -> DatasetRecord:
        """Load default benchmark transactions (sample_input.csv / flagged_transactions.csv)."""
        dataset_id = "default"
        with self._lock:
            if dataset_id in self._datasets:
                return self._datasets[dataset_id]

        if SAMPLE_INPUT_CSV.exists():
            df = pd.read_csv(SAMPLE_INPUT_CSV)
        elif CREDITCARD_CSV.exists():
            df = pd.read_csv(CREDITCARD_CSV, nrows=60)
        else:
            raise FileNotFoundError("Default benchmark dataset files not found.")

        if "TransactionID" not in df.columns:
            df.insert(0, "TransactionID", [f"TEST_TXN_{i:04d}" for i in range(len(df))])

        record = DatasetRecord(
            dataset_id=dataset_id,
            filename="sample_input.csv (Default Benchmark)",
            df_original=df,
            has_class="Class" in df.columns,
            has_transaction_id=True,
        )

        with self._lock:
            self._datasets[dataset_id] = record

        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        """Retrieve a registered dataset by ID."""
        clean_did = (dataset_id or "default").strip()
        with self._lock:
            if clean_did not in self._datasets:
                if clean_did == "ulb_benchmark":
                    return self.load_existing_ulb_dataset()
                if clean_did == "default":
                    return self.load_default_dataset()
                raise KeyError(f"Dataset '{clean_did}' not found or session expired.")
            return self._datasets[clean_did]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets in the current session."""
        try:
            self.load_default_dataset()
        except Exception:
            pass
        try:
            self.load_existing_ulb_dataset()
        except Exception:
            pass

        results = []
        with self._lock:
            for did, rec in self._datasets.items():
                results.append({
                    "dataset_id": rec.dataset_id,
                    "filename": rec.filename,
                    "total_rows": rec.total_rows,
                    "is_predicted": rec.is_predicted,
                    "has_class": rec.has_class,
                })
        return results

    def predict_dataset(
        self,
        dataset_id: str,
        threshold: float = FROZEN_THRESHOLD,
    ) -> DatasetRecord:
        """Run ML inference on every row of the dataset using the saved CatBoost model."""
        clean_did = (dataset_id or "default").strip()
        record = self.get_dataset(clean_did)
        df_orig = record.df_original

        pipeline = self._get_pipeline()
        feature_df = df_orig[REQUIRED_FEATURES].copy()
        X_infer = pipeline.preprocess(feature_df)

        probs = pipeline.model.predict_proba(X_infer)[:, 1]
        probs_rounded = np.round(probs, 6)

        is_suspicious = probs >= threshold
        pred_label = np.where(is_suspicious, "SUSPICIOUS", "NOT_FLAGGED")
        assess_status = np.where(is_suspicious, "SUSPICIOUS_PENDING_REVIEW", "NOT_FLAGGED")

        results_df = pd.DataFrame({
            "dataset_id": clean_did,
            "TransactionID": df_orig["TransactionID"],
            "Amount": df_orig["Amount"],
            "Time": df_orig["Time"],
            "fraud_probability": probs_rounded,
            "selected_threshold": round(float(threshold), 4),
            "selected_model": "CatBoost",
            "prediction_label": pred_label,
            "assessment_status": assess_status,
        })

        if record.has_class:
            results_df["Class"] = df_orig["Class"]

        total_tx = len(results_df)
        suspicious_tx = int(is_suspicious.sum())
        not_flagged_tx = total_tx - suspicious_tx
        pct_suspicious = round((suspicious_tx / total_tx * 100), 2) if total_tx > 0 else 0.0

        summary = {
            "dataset_id": clean_did,
            "filename": record.filename,
            "total_transactions": total_tx,
            "suspicious_count": suspicious_tx,
            "not_flagged_count": not_flagged_tx,
            "suspicious_percentage": pct_suspicious,
            "threshold": threshold,
            "model_name": "CatBoost",
            "has_ground_truth": record.has_class,
        }

        status_dist = [
            {"label": "Not Flagged", "count": not_flagged_tx, "percentage": round(100 - pct_suspicious, 2), "color": "#047857"},
            {"label": "Suspicious", "count": suspicious_tx, "percentage": pct_suspicious, "color": "#B45309"},
        ]

        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
        bin_labels = ["0.0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"]
        hist_counts, _ = np.histogram(probs, bins=bins)
        prob_hist = [
            {"bin": label, "count": int(count), "contains_threshold": (i == 1)}
            for i, (label, count) in enumerate(zip(bin_labels, hist_counts))
        ]

        top_suspicious_df = results_df[results_df["prediction_label"] == "SUSPICIOUS"].sort_values(
            by="fraud_probability", ascending=False
        ).head(20)

        top_risk_viz = [
            {
                "TransactionID": row["TransactionID"],
                "fraud_probability": float(row["fraud_probability"]),
                "amount": float(row["Amount"]),
            }
            for _, row in top_suspicious_df.iterrows()
        ]

        charts = {
            "status_distribution": status_dist,
            "probability_histogram": prob_hist,
            "top_risk_transactions": top_risk_viz,
            "decision_threshold": threshold,
        }

        eval_metrics = None
        if record.has_class:
            y_true = df_orig["Class"].astype(int).to_numpy()
            y_pred = is_suspicious.astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

            unique_labels = np.unique(y_true)
            if len(unique_labels) > 1:
                roc_auc = float(roc_auc_score(y_true, probs))
                pr_auc = float(average_precision_score(y_true, probs))
            else:
                roc_auc = None
                pr_auc = None

            is_ulb = (clean_did == "ulb_benchmark")
            eval_metrics = {
                "has_ground_truth": True,
                "is_ulb_benchmark": is_ulb,
                "evaluation_type": "full_dataset_descriptive" if is_ulb else "uploaded_ground_truth",
                "disclaimer": (
                    "Full 284,807-transaction dataset contains training (70%), validation (15%), and test (15%) partitions. "
                    "These metrics represent descriptive whole-dataset batch scoring performance, not independent test-set generalization. "
                    "For audited out-of-sample test set benchmarks (Recall: 85.14%, Precision: 47.37%, ROC-AUC: 0.9710, PR-AUC: 0.8298), "
                    "refer to the 'Model and threshold' or 'Overview' tabs."
                    if is_ulb
                    else "Evaluated against uploaded ground-truth labels. Class column was strictly excluded from model inference features."
                ),
                "confusion_matrix": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "total_flagged": int(tp + fp),
                    "total_actual_fraud": int(tp + fn),
                    "total_actual_genuine": int(tn + fp),
                },
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            }

        with self._lock:
            record.is_predicted = True
            record.results_df = results_df
            record.summary_stats = summary
            record.charts_data = charts
            record.evaluation_metrics = eval_metrics

        return record

    def get_results_page(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 25,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "fraud_probability",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Fetch paginated, filtered, and sorted prediction results with operator review status overlaid."""
        clean_did = (dataset_id or "default").strip()
        record = self.get_dataset(clean_did)
        if not record.is_predicted or record.results_df is None:
            self.predict_dataset(clean_did)
            record = self.get_dataset(clean_did)

        df_res = record.results_df.copy()

        # Overlay operator labels for this specific dataset only
        label_map = get_label_map(dataset_id=clean_did)
        if label_map:
            for tid, op_info in label_map.items():
                mask = df_res["TransactionID"] == tid
                if mask.any():
                    new_status = "CONFIRMED_FRAUD" if op_info["label"].lower() == "fraud" else "CONFIRMED_GENUINE"
                    df_res.loc[mask, "assessment_status"] = new_status

        # 1. Search Filter
        if search and search.strip():
            clean_search = search.strip().lower()
            df_res = df_res[df_res["TransactionID"].astype(str).str.lower().str.contains(clean_search)]

        # 2. Status Filter
        if status_filter and status_filter.strip().lower() not in ("all", ""):
            clean_status = status_filter.strip().lower()
            if clean_status in ("suspicious", "suspicious_pending_review"):
                df_res = df_res[df_res["assessment_status"] == "SUSPICIOUS_PENDING_REVIEW"]
            elif clean_status in ("not_flagged", "not flagged", "normal"):
                df_res = df_res[df_res["assessment_status"] == "NOT_FLAGGED"]
            elif clean_status in ("fraud", "confirmed_fraud"):
                df_res = df_res[df_res["assessment_status"] == "CONFIRMED_FRAUD"]
            elif clean_status in ("genuine", "confirmed_genuine"):
                df_res = df_res[df_res["assessment_status"] == "CONFIRMED_GENUINE"]

        # 3. Sorting
        valid_sort_cols = ["fraud_probability", "Amount", "Time", "TransactionID"]
        target_sort = sort_by if sort_by in valid_sort_cols else "fraud_probability"
        ascending = (sort_order.strip().lower() == "asc")
        df_res = df_res.sort_values(by=target_sort, ascending=ascending)

        # 4. Pagination
        total_records = len(df_res)
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        total_pages = max(1, (total_records + page_size - 1) // page_size)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        page_df = df_res.iloc[start_idx:end_idx]

        items: List[Dict[str, Any]] = []
        for _, row in page_df.iterrows():
            tid = str(row["TransactionID"])
            op_info = label_map.get(tid)
            item = {
                "dataset_id": clean_did,
                "TransactionID": tid,
                "amount": float(row["Amount"]) if pd.notna(row["Amount"]) else None,
                "time": float(row["Time"]) if pd.notna(row["Time"]) else None,
                "fraud_probability": float(row["fraud_probability"]),
                "selected_threshold": float(row["selected_threshold"]),
                "selected_model": str(row["selected_model"]),
                "prediction_label": str(row["prediction_label"]),
                "assessment_status": str(row["assessment_status"]),
                "operator_label": op_info["label"] if op_info else None,
                "operator": op_info.get("operator") if op_info else None,
                "labelled_at": op_info.get("created_at") if op_info else None,
            }
            if record.has_class and "Class" in row:
                item["ground_truth_class"] = int(row["Class"])
            items.append(item)

        return {
            "dataset_id": clean_did,
            "summary": record.summary_stats,
            "charts": record.charts_data,
            "evaluation_metrics": record.evaluation_metrics,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_records": total_records,
            "transactions": items,
        }

    def generate_results_csv(self, dataset_id: str) -> str:
        """Export all original uploaded columns plus predictions, strictly preserving row order."""
        clean_did = (dataset_id or "default").strip()
        record = self.get_dataset(clean_did)
        if not record.is_predicted or record.results_df is None:
            self.predict_dataset(clean_did)
            record = self.get_dataset(clean_did)

        df_export = record.df_original.copy()

        df_export["dataset_id"] = clean_did
        df_export["fraud_probability"] = record.results_df["fraud_probability"]
        df_export["selected_threshold"] = record.results_df["selected_threshold"]
        df_export["selected_model"] = record.results_df["selected_model"]
        df_export["prediction_label"] = record.results_df["prediction_label"]
        df_export["assessment_status"] = record.results_df["assessment_status"]

        # Overlay any operator labels for this specific dataset
        label_map = get_label_map(dataset_id=clean_did)
        if label_map:
            for tid, op_info in label_map.items():
                mask = df_export["TransactionID"] == tid
                if mask.any():
                    new_status = "CONFIRMED_FRAUD" if op_info["label"].lower() == "fraud" else "CONFIRMED_GENUINE"
                    df_export.loc[mask, "assessment_status"] = new_status

        return df_export.to_csv(index=False)

    def generate_template_csv(self) -> str:
        """Generate a downloadable CSV template with all required columns and sample rows."""
        sample_file = REPO_ROOT / "outputs" / "predictions" / "sample_input.csv"
        if sample_file.exists():
            df_sample = pd.read_csv(sample_file, nrows=3)
            if "TransactionID" not in df_sample.columns:
                df_sample.insert(0, "TransactionID", [f"SAMPLE_TXN_{i+1:03d}" for i in range(len(df_sample))])
            if "Class" not in df_sample.columns:
                df_sample["Class"] = [0, 0, 1]
            return df_sample.to_csv(index=False)

        cols = ["TransactionID", "Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        rows = [
            ["SAMPLE_TXN_001", 0.0] + [0.0] * 28 + [149.62, 0],
            ["SAMPLE_TXN_002", 1.0] + [-0.5] * 28 + [2.69, 0],
            ["SAMPLE_TXN_003", 2.0] + [1.2] * 28 + [378.00, 1],
        ]
        df_template = pd.DataFrame(rows, columns=cols)
        return df_template.to_csv(index=False)


# Global singleton instance
DATASET_SERVICE = DatasetService()
