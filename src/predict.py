"""
Reusable Batch & Real-Time Fraud Inference Pipeline
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Scope: ML Module Inference Engine

This module provides a standalone, production-ready inference API and CLI:
1. Validates input schema (V1–V28, Amount, Time).
2. Preserves caller TransactionID or generates sequential synthetic IDs.
3. Excludes TransactionID and target labels (Class) from feature input.
4. Applies Phase 1 preprocessing (Amount and Time scaling using training parameters).
5. Generates fraud probabilities using the trained CatBoost model.
6. Flags transactions exceeding the frozen optimal threshold (0.144).
7. Exports clean LLM-handoff predictions strictly without ground-truth labels.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Union, Optional, Tuple, List

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set stdout to UTF-8 with replace to prevent cp1252 encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Set environment variables for clean headless execution
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler

from src.preprocessing import FraudDataPreprocessor

# Required raw features for inference
REQUIRED_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
EXPECTED_MODEL_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# Deterministic Phase 1 training parameters (RobustScaler fitted exclusively on training data)
TRAIN_AMOUNT_MEDIAN: float = 22.00
TRAIN_AMOUNT_IQR: float = 71.83
TRAIN_TIME_MEDIAN: float = 84845.5
TRAIN_TIME_IQR: float = 85111.25


class FraudInferencePipeline:
    """Production-grade inference pipeline for credit card fraud detection."""

    def __init__(
        self,
        model_path: str = "models/catboost_model.cbm",
        preprocessor_path: Optional[str] = "models/preprocessor.joblib",
        default_threshold: float = 0.144,
    ) -> None:
        """Initialize the inference pipeline with model and preprocessor.

        Parameters
        ----------
        model_path : str
            Path to the saved CatBoost model artifact.
        preprocessor_path : Optional[str]
            Path to the saved FraudDataPreprocessor artifact. If None or file
            does not exist, falls back to deterministic training parameters.
        default_threshold : float
            Decision threshold for flagging fraud (default 0.144 from Phase 3).
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.default_threshold = default_threshold

        # 1. Load CatBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        # 2. Load or construct preprocessor
        self.preprocessor = self._init_preprocessor(preprocessor_path)

    def _init_preprocessor(self, path: Optional[str]) -> FraudDataPreprocessor:
        """Initialize preprocessor from saved artifact or deterministic parameters."""
        if path and os.path.exists(path):
            try:
                prep = joblib.load(path)
                if isinstance(prep, FraudDataPreprocessor) and prep.is_fitted_:
                    return prep
            except Exception:
                pass  # Fall back to manual initialization

        # Fallback: construct fitted preprocessor using exact training parameters
        prep = FraudDataPreprocessor(amount_scaler_type="robust", time_scaler_type="robust")
        scaler_amount = RobustScaler()
        scaler_amount.center_ = np.array([TRAIN_AMOUNT_MEDIAN])
        scaler_amount.scale_ = np.array([TRAIN_AMOUNT_IQR])

        scaler_time = RobustScaler()
        scaler_time.center_ = np.array([TRAIN_TIME_MEDIAN])
        scaler_time.scale_ = np.array([TRAIN_TIME_IQR])

        prep.amount_scaler = scaler_amount
        prep.time_scaler = scaler_time
        prep.is_fitted_ = True
        return prep

    def validate_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate input DataFrame schema and extract transaction IDs.

        Parameters
        ----------
        df : pd.DataFrame
            Input transactions.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (df_features_raw, transaction_ids)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot perform inference.")

        # Check required features
        missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input data is missing required feature columns: {missing_cols}. "
                f"Expected features: {REQUIRED_FEATURES}"
            )

        # Extract or generate TransactionID
        if "TransactionID" in df.columns:
            txn_ids = df["TransactionID"].astype(str).reset_index(drop=True)
        else:
            txn_ids = pd.Series([f"TXN_INFER_{i+1:06d}" for i in range(len(df))], name="TransactionID")

        # Exclude TransactionID, Class, and any other non-feature columns
        feature_df = df[REQUIRED_FEATURES].copy().reset_index(drop=True)

        return feature_df, txn_ids

    def preprocess(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Apply Phase 1 transformations (scaling Amount and Time).

        Parameters
        ----------
        df_features : pd.DataFrame
            Raw feature matrix with V1–V28, Amount, Time.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix with Amount_scaled and Time_scaled.
        """
        transformed = self.preprocessor.transform(df_features)
        # Ensure column ordering strictly matches CatBoost expected feature names
        return transformed[EXPECTED_MODEL_FEATURES]

    def predict(
        self,
        data: Union[pd.DataFrame, str, Path],
        threshold: Optional[float] = None,
        return_all: bool = False,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform fraud scoring and extract flagged suspicious transactions.

        Parameters
        ----------
        data : Union[pd.DataFrame, str, Path]
            Input data as pandas DataFrame or path to CSV file.
        threshold : Optional[float]
            Decision threshold (defaults to self.default_threshold = 0.144).
        return_all : bool
            If True, returns all scored transactions with an 'is_suspicious' flag.
            If False (default), returns only transactions flagged as suspicious.
        output_path : Optional[str]
            Path to export results CSV. If provided, saves results to disk.

        Returns
        -------
        pd.DataFrame
            Predictions DataFrame ready for LLM module handoff or downstream review.
        """
        if threshold is None:
            threshold = self.default_threshold

        # Load from CSV path if string or Path
        if isinstance(data, (str, Path)):
            csv_path = str(data)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
            df_raw = pd.read_csv(csv_path)
        elif isinstance(data, pd.DataFrame):
            df_raw = data.copy()
        else:
            raise TypeError(f"Invalid input type: {type(data)}. Expected DataFrame or path string.")

        # 1. Validate features and extract IDs
        df_features_raw, txn_ids = self.validate_input(df_raw)

        # 2. Preprocess features (Amount & Time scaling)
        X_infer = self.preprocess(df_features_raw)

        # 3. Predict fraud probabilities
        probs = self.model.predict_proba(X_infer)[:, 1]

        # 4. Construct output DataFrame
        is_suspicious = probs >= threshold

        if return_all:
            df_out = pd.DataFrame({
                "TransactionID": txn_ids,
                "fraud_probability": np.round(probs, 6),
                "selected_threshold": round(float(threshold), 4),
                "selected_model": "CatBoost",
                "is_suspicious": is_suspicious,
                "assessment_status": np.where(is_suspicious, "SUSPICIOUS_PENDING_REVIEW", "NORMAL"),
            }).sort_values(by="fraud_probability", ascending=False).reset_index(drop=True)
        else:
            mask = is_suspicious
            df_out = pd.DataFrame({
                "TransactionID": txn_ids[mask],
                "fraud_probability": np.round(probs[mask], 6),
                "selected_threshold": round(float(threshold), 4),
                "selected_model": "CatBoost",
                "assessment_status": "SUSPICIOUS_PENDING_REVIEW",
            }).sort_values(by="fraud_probability", ascending=False).reset_index(drop=True)

        # 5. Export to CSV if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            df_out.to_csv(output_path, index=False)
            print(f"[+] Predictions saved to: {output_path}")

        return df_out


def main() -> None:
    """CLI entry point for batch fraud inference."""
    parser = argparse.ArgumentParser(
        description="Cost-Sensitive Fraud Detection Batch Inference Engine (Phase Final)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input transactions CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/predictions/flagged_transactions.csv",
        help="Path to save flagged suspicious transactions CSV.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.144,
        help="Cost-sensitive decision threshold.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/catboost_model.cbm",
        help="Path to CatBoost model artifact.",
    )
    parser.add_argument(
        "--preprocessor",
        "-p",
        type=str,
        default="models/preprocessor.joblib",
        help="Path to preprocessor artifact.",
    )
    parser.add_argument(
        "--return-all",
        action="store_true",
        help="Return all transactions with an 'is_suspicious' flag instead of only flagged transactions.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FRAUD DETECTION BATCH INFERENCE ENGINE")
    print("=" * 70)
    print(f"[*] Input File:         {args.input}")
    print(f"[*] Decision Threshold: {args.threshold:.4f}")
    print(f"[*] Model Artifact:     {args.model}")
    print(f"[*] Output File:        {args.output}")
    print(f"[*] Return Mode:        {'All Scored Transactions' if args.return_all else 'Flagged Suspicious Only'}")

    pipeline = FraudInferencePipeline(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        default_threshold=args.threshold,
    )

    df_out = pipeline.predict(
        data=args.input,
        threshold=args.threshold,
        return_all=args.return_all,
        output_path=args.output,
    )

    print("\n[+] Inference Execution Summary:")
    print(f"    - Records Output:       {len(df_out):,}")
    if args.return_all:
        flagged_n = int(df_out["is_suspicious"].sum())
        print(f"    - Suspicious Flagged:   {flagged_n:,} ({flagged_n/len(df_out)*100:.2f}%)")
    else:
        print(f"    - Suspicious Flagged:   {len(df_out):,}")
    print(f"    - LLM Handoff Columns:  {list(df_out.columns)}")
    print(f"    - Top Flagged Prob:     {df_out['fraud_probability'].iloc[0] if len(df_out) > 0 else 'N/A'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
