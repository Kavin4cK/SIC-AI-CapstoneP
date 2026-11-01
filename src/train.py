"""
Model Training Module
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Phase 2: Machine Learning Model Development

This script:
1. Reuses the Phase 1 preprocessing pipeline (70/15/15 stratified split, random_state=42).
2. Investigates and audits whether identical duplicate records overlap across train and validation splits.
3. Conducts class-weighting experiments for LightGBM and CatBoost to address severe class imbalance.
4. Trains two separate, non-ensembled models:
   - LightGBM Classifier
   - CatBoost Classifier
5. Evaluates model performance on the validation set.
6. Persists trained models to:
   - models/lightgbm_model.txt
   - models/catboost_model.cbm
7. Exports validation predictions with Transaction IDs, ground truth, predictions, and fraud probabilities to:
   - outputs/evaluation/lightgbm_val_predictions.csv
   - outputs/evaluation/catboost_val_predictions.csv
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to sys.path so 'src' is discoverable regardless of working directory
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variables for clean headless execution
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

# Import Phase 1 reusable preprocessing pipeline
from src.preprocessing import run_preprocessing_pipeline


def investigate_duplicate_overlap(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    ids_train: pd.Series,
    ids_val: pd.Series,
) -> Dict[str, Any]:
    """Investigate whether identical duplicate records occur across train and validation splits.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Transformed feature matrices.
    y_train, y_val : pd.Series
        Target labels.
    ids_train, ids_val : pd.Series
        Transaction IDs.

    Returns
    -------
    dict
        Detailed audit findings regarding duplicate overlap.
    """
    print("\n" + "=" * 70)
    print("DUPLICATE RECORD OVERLAP INVESTIGATION (TRAIN vs. VALIDATION)")
    print("=" * 70)

    # Convert feature vectors to hashable tuples
    train_tuples = list(map(tuple, X_train.values))
    val_tuples = list(map(tuple, X_val.values))

    train_set = set(train_tuples)
    val_set = set(val_tuples)

    # Overlapping unique feature profiles
    overlap_profiles = train_set.intersection(val_set)
    num_overlap_profiles = len(overlap_profiles)

    # Count affected validation records
    is_val_overlapping = [t in train_set for t in val_tuples]
    val_overlap_count = sum(is_val_overlapping)
    val_overlap_fraud = int(y_val[is_val_overlapping].sum())
    val_overlap_genuine = val_overlap_count - val_overlap_fraud

    # Count affected training records
    is_train_overlapping = [t in val_set for t in train_tuples]
    train_overlap_count = sum(is_train_overlapping)
    train_overlap_fraud = int(y_train[is_train_overlapping].sum())
    train_overlap_genuine = train_overlap_count - train_overlap_fraud

    print(f"Features evaluated for exact duplication: {X_train.shape[1]}")
    print(f"Unique feature vectors in Train:           {len(train_set):,} / {len(X_train):,}")
    print(f"Unique feature vectors in Validation:      {len(val_set):,} / {len(X_val):,}")
    print(f"\n[Findings]:")
    print(f"  - Overlapping unique feature profiles:   {num_overlap_profiles:,}")
    print(f"  - Validation rows matching a Train row:  {val_overlap_count:,} ({val_overlap_count/len(X_val)*100:.3f}% of Val)")
    print(f"    * Genuine validation duplicates:       {val_overlap_genuine:,}")
    print(f"    * Fraudulent validation duplicates:    {val_overlap_fraud:,} ({val_overlap_fraud}/{int(y_val.sum())} of Val fraud)")
    print(f"  - Training rows matching a Val row:      {train_overlap_count:,} ({train_overlap_count/len(X_train)*100:.3f}% of Train)")
    print(f"    * Genuine training duplicates:         {train_overlap_genuine:,}")
    print(f"    * Fraudulent training duplicates:      {train_overlap_fraud:,} ({train_overlap_fraud}/{int(y_train.sum())} of Train fraud)")

    print("\n[Audit Context & Decision]:")
    print("  In the full dataset (284,807 rows), 1,081 duplicate rows exist.")
    print("  Under the stratified 70/15/15 random split, identical rows naturally land across splits.")
    print("  As required, these records are NOT silently removed and the split is NOT altered.")
    print("  Because features V1-V28 are PCA projections without transaction metadata, their real-world")
    print("  meaning (distinct purchases with identical PCA, rapid retries, or logging duplicates)")
    print("  cannot be established.")

    return {
        "num_overlap_profiles": num_overlap_profiles,
        "val_overlap_count": val_overlap_count,
        "val_overlap_fraud": val_overlap_fraud,
        "val_overlap_genuine": val_overlap_genuine,
        "train_overlap_count": train_overlap_count,
        "train_overlap_fraud": train_overlap_fraud,
        "train_overlap_genuine": train_overlap_genuine,
    }


def run_class_weighting_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str = "outputs/evaluation",
) -> pd.DataFrame:
    """Experiment with different class-weighting configurations for LightGBM and CatBoost.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and labels.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation features and labels.
    output_dir : str
        Directory to export experiment results.

    Returns
    -------
    pd.DataFrame
        Tabular summary of class-weighting experimental results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("CLASS-WEIGHTING EXPERIMENTAL BENCHMARK")
    print("=" * 70)

    # Imbalance ratio: ~578.55 : 1
    imbalance_ratio = float((len(y_train) - y_train.sum()) / y_train.sum())
    print(f"Training Imbalance Ratio (Neg:Pos): {imbalance_ratio:.2f} : 1")

    experiments = []

    # 1. LightGBM Weighting Experiments
    lgb_weights = [
        ("LightGBM (Unweighted, spw=1.0)", 1.0),
        ("LightGBM (Moderate, spw=5.0)", 5.0),
        ("LightGBM (Moderate, spw=10.0)", 10.0),
        ("LightGBM (Balanced, spw=578.5)", imbalance_ratio),
    ]

    print("\n[LightGBM Class-Weighting Trials]:")
    for name, spw in lgb_weights:
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1]

        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", pos_label=1, zero_division=0)
        pr_auc = average_precision_score(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)

        print(f"  {name:<35} | Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
        experiments.append({
            "model_family": "LightGBM",
            "experiment": name,
            "weight_param": spw,
            "precision_fraud": round(p, 4),
            "recall_fraud": round(r, 4),
            "f1_fraud": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
        })

    # 2. CatBoost Weighting Experiments
    cb_weights = [
        ("CatBoost (Unweighted)", None),
        ("CatBoost (Balanced)", "Balanced"),
    ]

    print("\n[CatBoost Class-Weighting Trials]:")
    for name, weight_mode in cb_weights:
        cb = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            auto_class_weights=weight_mode,
            random_seed=42,
            thread_count=4,
            verbose=False,
            train_dir=os.path.join(tempfile.gettempdir(), "catboost_train"),
        )
        cb.fit(X_train, y_train)
        y_pred = cb.predict(X_val)
        y_proba = cb.predict_proba(X_val)[:, 1]

        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", pos_label=1, zero_division=0)
        pr_auc = average_precision_score(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)

        print(f"  {name:<35} | Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
        experiments.append({
            "model_family": "CatBoost",
            "experiment": name,
            "weight_param": str(weight_mode),
            "precision_fraud": round(p, 4),
            "recall_fraud": round(r, 4),
            "f1_fraud": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
        })

    exp_df = pd.DataFrame(experiments)
    exp_csv_path = os.path.join(output_dir, "class_weighting_experiments.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"\n[+] Class-weighting experiments saved to: {exp_csv_path}")

    return exp_df


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    scale_pos_weight: float = 5.0,
    random_state: int = 42,
    n_jobs: int = 4,
) -> LGBMClassifier:
    """Train LightGBM Classifier with configurable hyperparameters.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and labels.
    n_estimators : int
        Number of boosting trees (default: 300).
    learning_rate : float
        Boosting learning rate (default: 0.03).
    num_leaves : int
        Maximum tree leaves (default: 31).
    scale_pos_weight : float
        Positive class weight multiplier (default: 5.0).
    random_state : int
        Seed for reproducibility (default: 42).
    n_jobs : int
        Number of parallel CPU threads (default: 4).

    Returns
    -------
    LGBMClassifier
        Fitted LightGBM model.
    """
    print("\n[*] Training LightGBM Classifier...")
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print(f"[+] LightGBM training complete ({n_estimators} trees, scale_pos_weight={scale_pos_weight}).")
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iterations: int = 300,
    learning_rate: float = 0.05,
    depth: int = 6,
    auto_class_weights: str = "Balanced",
    random_seed: int = 42,
    thread_count: int = 4,
) -> CatBoostClassifier:
    """Train CatBoost Classifier with configurable hyperparameters.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and labels.
    iterations : int
        Number of boosting iterations (default: 300).
    learning_rate : float
        Boosting learning rate (default: 0.05).
    depth : int
        Tree depth (default: 6).
    auto_class_weights : str
        Class weighting mode ('Balanced' or None).
    random_seed : int
        Seed for reproducibility (default: 42).
    thread_count : int
        Number of parallel CPU threads (default: 4).

    Returns
    -------
    CatBoostClassifier
        Fitted CatBoost model.
    """
    print("\n[*] Training CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        auto_class_weights=auto_class_weights,
        random_seed=random_seed,
        thread_count=thread_count,
        verbose=False,
        train_dir=os.path.join(tempfile.gettempdir(), "catboost_final"),
    )
    model.fit(X_train, y_train)
    print(f"[+] CatBoost training complete ({iterations} iterations, auto_class_weights='{auto_class_weights}').")
    return model


def save_models_and_predictions(
    lgb_model: LGBMClassifier,
    cat_model: CatBoostClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    ids_val: pd.Series,
    models_dir: str = "models",
    output_dir: str = "outputs/evaluation",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Persist trained models and export validation predictions with Transaction IDs.

    Parameters
    ----------
    lgb_model, cat_model : LGBMClassifier, CatBoostClassifier
        Fitted models.
    X_val, y_val, ids_val : pd.DataFrame, pd.Series, pd.Series
        Validation set components.
    models_dir : str
        Directory to persist models.
    output_dir : str
        Directory to persist predictions.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Validation predictions DataFrames for LightGBM and CatBoost.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("MODEL PERSISTENCE & PREDICTION EXPORT")
    print("=" * 70)

    # 1. Save LightGBM model to models/lightgbm_model.txt
    lgb_model_path = os.path.join(models_dir, "lightgbm_model.txt")
    lgb_model.booster_.save_model(lgb_model_path)
    print(f"[+] Saved LightGBM model: {lgb_model_path}")

    # 2. Save CatBoost model to models/catboost_model.cbm
    cat_model_path = os.path.join(models_dir, "catboost_model.cbm")
    cat_model.save_model(cat_model_path)
    print(f"[+] Saved CatBoost model: {cat_model_path}")

    # 3. Generate and export LightGBM validation predictions
    lgb_pred = lgb_model.predict(X_val)
    lgb_proba = lgb_model.predict_proba(X_val)[:, 1]

    df_lgb_val = pd.DataFrame({
        "TransactionID": ids_val.values,
        "y_true": y_val.values,
        "y_pred": lgb_pred,
        "fraud_probability": lgb_proba,
    })
    lgb_pred_path = os.path.join(output_dir, "lightgbm_val_predictions.csv")
    df_lgb_val.to_csv(lgb_pred_path, index=False)
    print(f"[+] Saved LightGBM validation predictions: {lgb_pred_path} ({len(df_lgb_val):,} rows)")

    # 4. Generate and export CatBoost validation predictions
    cat_pred = cat_model.predict(X_val)
    cat_proba = cat_model.predict_proba(X_val)[:, 1]

    df_cat_val = pd.DataFrame({
        "TransactionID": ids_val.values,
        "y_true": y_val.values,
        "y_pred": cat_pred,
        "fraud_probability": cat_proba,
    })
    cat_pred_path = os.path.join(output_dir, "catboost_val_predictions.csv")
    df_cat_val.to_csv(cat_pred_path, index=False)
    print(f"[+] Saved CatBoost validation predictions: {cat_pred_path} ({len(df_cat_val):,} rows)")

    # 5. Sanity verifications
    assert (df_lgb_val["fraud_probability"] >= 0.0).all() and (df_lgb_val["fraud_probability"] <= 1.0).all(), "LightGBM probabilities out of bounds!"
    assert (df_cat_val["fraud_probability"] >= 0.0).all() and (df_cat_val["fraud_probability"] <= 1.0).all(), "CatBoost probabilities out of bounds!"
    assert len(df_lgb_val) == len(X_val), "LightGBM prediction row count mismatch!"
    assert len(df_cat_val) == len(X_val), "CatBoost prediction row count mismatch!"
    assert (df_lgb_val["TransactionID"] == ids_val.values).all(), "LightGBM TransactionID alignment error!"
    assert (df_cat_val["TransactionID"] == ids_val.values).all(), "CatBoost TransactionID alignment error!"

    print("[+] All prediction integrity checks passed (IDs verified, probabilities in [0, 1]).")

    return df_lgb_val, df_cat_val


def run_training_pipeline(
    data_path: str = "data/creditcard.csv",
    models_dir: str = "models",
    output_dir: str = "outputs/evaluation",
) -> Dict[str, Any]:
    """Execute end-to-end Phase 2 model training workflow.

    Parameters
    ----------
    data_path : str
        Path to creditcard.csv.
    models_dir : str
        Target models directory.
    output_dir : str
        Target evaluation directory.

    Returns
    -------
    dict
        Training artifacts, models, and predictions.
    """
    print("=" * 70)
    print("PHASE 2: MACHINE LEARNING MODEL DEVELOPMENT")
    print("=" * 70)

    # 1. Load preprocessed splits from Phase 1 pipeline
    pipeline_data = run_preprocessing_pipeline(data_path=data_path)
    X_train = pipeline_data["X_train"]
    y_train = pipeline_data["y_train"]
    ids_train = pipeline_data["ids_train"]
    X_val = pipeline_data["X_val"]
    y_val = pipeline_data["y_val"]
    ids_val = pipeline_data["ids_val"]
    X_test = pipeline_data["X_test"]
    y_test = pipeline_data["y_test"]
    ids_test = pipeline_data["ids_test"]

    # 2. Duplicate record overlap investigation
    dup_findings = investigate_duplicate_overlap(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        ids_train=ids_train,
        ids_val=ids_val,
    )

    # 3. Class-weighting experiments
    exp_df = run_class_weighting_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=output_dir,
    )

    # 4. Train final LightGBM model
    lgb_model = train_lightgbm(
        X_train=X_train,
        y_train=y_train,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        scale_pos_weight=5.0,
        random_state=42,
        n_jobs=4,
    )

    # 5. Train final CatBoost model
    cat_model = train_catboost(
        X_train=X_train,
        y_train=y_train,
        iterations=300,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=4,
    )

    # 6. Save models and validation predictions
    df_lgb_val, df_cat_val = save_models_and_predictions(
        lgb_model=lgb_model,
        cat_model=cat_model,
        X_val=X_val,
        y_val=y_val,
        ids_val=ids_val,
        models_dir=models_dir,
        output_dir=output_dir,
    )

    return {
        "lgb_model": lgb_model,
        "cat_model": cat_model,
        "df_lgb_val": df_lgb_val,
        "df_cat_val": df_cat_val,
        "dup_findings": dup_findings,
        "class_weighting_experiments": exp_df,
        "pipeline_data": pipeline_data,
    }


if __name__ == "__main__":
    run_training_pipeline()
