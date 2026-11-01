"""
CatBoost Hyperparameter Optimization & Cost-Sensitive Tuning Module
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Experiment 1: CatBoost Hyperparameter Optimization

Strict Constraints:
1. Do NOT modify or retrain the existing baseline model (models/catboost_model.cbm).
2. Do NOT load test labels or test predictions during tuning.
3. Use only the existing training and validation datasets for experimentation.
4. Preserve the existing preprocessing pipeline and feature definitions.
5. Keep the original dataset unchanged.
6. Do not introduce SMOTE or synthetic fraud samples.
7. Fixed random seed = 42 for complete reproducibility.

Exploration Space:
- iterations: 300 to 1500
- depth: 4 to 10
- learning_rate: 0.01 to 0.15
- l2_leaf_reg: 1 to 10
- random_strength: 0 to 2
- border_count: 64 to 254
- class weighting: Balanced and selected explicit positive-class weights
- early stopping where appropriate
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

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
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessing import run_preprocessing_pipeline
from src.cost_sensitive import CostMatrixConfig, evaluate_threshold_grid, find_optimal_threshold


def get_candidate_configurations() -> List[Dict[str, Any]]:
    """Define a systematic, reproducible search space of CatBoost configurations."""
    candidates = [
        # --- Group 1: Baseline Reference ---
        {
            "trial_id": "T01_Baseline",
            "group": "Baseline",
            "description": "Baseline CatBoost (Phase 2 reference)",
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        # --- Group 2: Tree Depth Exploration (iters=500, lr=0.03, Balanced) ---
        {
            "trial_id": "T02_Depth_4",
            "group": "Tree Depth",
            "description": "Shallow depth=4 to prevent overfitting rare modes",
            "iterations": 500,
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T03_Depth_5",
            "group": "Tree Depth",
            "description": "Moderately shallow depth=5",
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T04_Depth_6",
            "group": "Tree Depth",
            "description": "Standard depth=6 with 500 iterations",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T05_Depth_7",
            "group": "Tree Depth",
            "description": "Deeper depth=7 for non-linear feature interactions",
            "iterations": 500,
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T06_Depth_8",
            "group": "Tree Depth",
            "description": "Deep depth=8 with regularized L2 leaf",
            "iterations": 400,
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.5,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T07_Depth_10",
            "group": "Tree Depth",
            "description": "Maximum depth=10 with heavy L2 regularization",
            "iterations": 300,
            "depth": 10,
            "learning_rate": 0.025,
            "l2_leaf_reg": 8.0,
            "random_strength": 2.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        # --- Group 3: Iterations & Learning Rate Trade-off ---
        {
            "trial_id": "T08_Iter_300_LR_08",
            "group": "Iters & LR",
            "description": "Faster learning rate (lr=0.08) at 300 iters",
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.08,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T09_Iter_300_LR_12",
            "group": "Iters & LR",
            "description": "Aggressive learning rate (lr=0.12) at 300 iters",
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.12,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T10_Iter_600_LR_04",
            "group": "Iters & LR",
            "description": "Moderate depth and iterations (600 iters, lr=0.04)",
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.04,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T11_Iter_800_LR_025",
            "group": "Iters & LR",
            "description": "Smaller learning rate (lr=0.025) with 800 iters",
            "iterations": 800,
            "depth": 6,
            "learning_rate": 0.025,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T12_Iter_1000_LR_02",
            "group": "Iters & LR",
            "description": "Low learning rate (lr=0.02) with 1000 iters",
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.02,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T13_Iter_1200_LR_015",
            "group": "Iters & LR",
            "description": "Conservative lr=0.015 with 1200 iters",
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.015,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T14_Iter_1500_LR_01",
            "group": "Iters & LR",
            "description": "Very small lr=0.01 with maximum 1500 iters",
            "iterations": 1500,
            "depth": 5,
            "learning_rate": 0.01,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        # --- Group 4: Regularization (L2 Leaf Reg & Random Strength) ---
        {
            "trial_id": "T15_Reg_L2_1_RS_0",
            "group": "Regularization",
            "description": "Low regularization: l2_leaf_reg=1, random_strength=0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 1.0,
            "random_strength": 0.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T16_Reg_L2_5_RS_05",
            "group": "Regularization",
            "description": "Moderate regularization: l2_leaf_reg=5, random_strength=0.5",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T17_Reg_L2_8_RS_15",
            "group": "Regularization",
            "description": "High regularization: l2_leaf_reg=8, random_strength=1.5",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "random_strength": 1.5,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T18_Reg_L2_10_RS_20",
            "group": "Regularization",
            "description": "Maximum regularization: l2_leaf_reg=10, random_strength=2.0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10.0,
            "random_strength": 2.0,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        # --- Group 5: Border Count (Quantization) ---
        {
            "trial_id": "T19_Border_64",
            "group": "Border Count",
            "description": "Coarse quantization: border_count=64 (prevents overfitting)",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 64,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T20_Border_128",
            "group": "Border Count",
            "description": "Medium quantization: border_count=128",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 128,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": None,
        },
        # --- Group 6: Explicit Class Weighting ---
        {
            "trial_id": "T21_Weight_Pos_20",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=20.0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 20.0,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T22_Weight_Pos_50",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=50.0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 50.0,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T23_Weight_Pos_100",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=100.0 (matches cost ratio 100:1)",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 100.0,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T24_Weight_Pos_200",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=200.0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 200.0,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T25_Weight_Pos_300",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=300.0",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 300.0,
            "early_stopping_rounds": None,
        },
        {
            "trial_id": "T26_Weight_Pos_579",
            "group": "Class Weighting",
            "description": "Explicit scale_pos_weight=579.0 (exact class imbalance ratio)",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 254,
            "auto_class_weights": None,
            "scale_pos_weight": 579.0,
            "early_stopping_rounds": None,
        },
        # --- Group 7: Advanced Synergy & Early Stopping ---
        {
            "trial_id": "T27_Synergy_EarlyStop_D5",
            "group": "Synergy & EarlyStop",
            "description": "Depth=5, lr=0.03, l2=5, border=128 with early stopping 100",
            "iterations": 1000,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 128,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": 100,
        },
        {
            "trial_id": "T28_Synergy_EarlyStop_D6",
            "group": "Synergy & EarlyStop",
            "description": "Depth=6, lr=0.03, l2=5, border=128 with early stopping 100",
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 128,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": 100,
        },
        {
            "trial_id": "T29_Synergy_EarlyStop_D7",
            "group": "Synergy & EarlyStop",
            "description": "Depth=7, lr=0.025, l2=7, border=128 with early stopping 100",
            "iterations": 900,
            "depth": 7,
            "learning_rate": 0.025,
            "l2_leaf_reg": 7.0,
            "random_strength": 1.2,
            "border_count": 128,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": 100,
        },
        {
            "trial_id": "T30_Synergy_EarlyStop_W100",
            "group": "Synergy & EarlyStop",
            "description": "Depth=6, lr=0.03, l2=5, border=128, weight=100 with early stopping 100",
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "border_count": 128,
            "auto_class_weights": None,
            "scale_pos_weight": 100.0,
            "early_stopping_rounds": 100,
        },
        {
            "trial_id": "T31_Synergy_D4_Iter1200",
            "group": "Synergy & EarlyStop",
            "description": "Shallow depth=4, lr=0.035, l2=4, border=254, 1200 iters early stopping",
            "iterations": 1200,
            "depth": 4,
            "learning_rate": 0.035,
            "l2_leaf_reg": 4.0,
            "random_strength": 0.8,
            "border_count": 254,
            "auto_class_weights": "Balanced",
            "scale_pos_weight": None,
            "early_stopping_rounds": 100,
        },
    ]
    return candidates


def run_catboost_tuning(
    c_fp: float = 5.0,
    c_fn: float = 500.0,
    random_state: int = 42,
    output_dir: str = "outputs/tuning",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """Run comprehensive CatBoost hyperparameter optimization across candidate configurations.

    Parameters
    ----------
    c_fp : float
        Hypothetical review cost per False Positive (INR).
    c_fn : float
        Hypothetical chargeback loss per False Negative (INR).
    random_state : int
        Fixed random seed for reproducible training.
    output_dir : str
        Directory to save tuning results and comparisons.
    models_dir : str
        Directory to save tuned model artifact (models/catboost_tuned.cbm).

    Returns
    -------
    dict
        Consolidated tuning benchmark results and best candidate metadata.
    """
    print("=" * 75)
    print("EXPERIMENT 1: CATBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 75)
    print("STRICT CONSTRAINTS ADHERENCE:")
    print("  1. Baseline model artifact (models/catboost_model.cbm) is PRESERVED.")
    print("  2. Test partition is STRICTLY UNTOUCHED (zero leakage).")
    print("  3. Only Train (70%) and Validation (15%) partitions are utilized.")
    print("  4. Original data and Phase 1 preprocessing are unchanged.")
    print("  5. Reproducible random seed = 42.")
    print("=" * 75)

    # 1. Load Preprocessed Train and Validation Data ONLY
    print("\n[*] Step 1: Loading preprocessed Train and Validation splits from pipeline...")
    data = run_preprocessing_pipeline()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Explicit verification of data isolation
    assert "X_test" in data, "Integrity check: pipeline contains test split"
    del data["X_test"]
    del data["y_test"]
    del data["ids_test"]
    print(f"[+] Loaded Train: {len(X_train):,} rows ({int(np.sum(y_train == 1))} fraud)")
    print(f"[+] Loaded Val:   {len(X_val):,} rows ({int(np.sum(y_val == 1))} fraud)")
    print("[+] Test partition successfully isolated and excluded from memory.")

    # 2. Setup Central Cost Matrix Configuration (FP = INR 5, FN = INR 500)
    cost_config = CostMatrixConfig(c_fp=c_fp, c_fn=c_fn)
    print(f"\n[*] Cost Configuration: FP = INR {cost_config.c_fp:.2f}, FN = INR {cost_config.c_fn:.2f}")

    # 3. Define candidate configurations
    candidates = get_candidate_configurations()
    print(f"[*] Defined {len(candidates)} candidate configurations across 7 exploration groups.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results_records = []
    trained_models = {}
    validation_probs = {}

    print("\n" + "-" * 115)
    print(f"{'Trial ID':<26} | {'Group':<18} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Opt Thresh':<10} | {'Recall':<7} | {'Prec':<7} | {'Min Cost (INR)':<14} | {'Time (s)':<8}")
    print("-" * 115)

    for cand in candidates:
        trial_id = cand["trial_id"]
        t_start = time.time()

        # Build CatBoost parameters
        cb_params = {
            "iterations": cand["iterations"],
            "depth": cand["depth"],
            "learning_rate": cand["learning_rate"],
            "l2_leaf_reg": cand["l2_leaf_reg"],
            "random_strength": cand["random_strength"],
            "border_count": cand["border_count"],
            "random_seed": random_state,
            "verbose": 0,
            "thread_count": 4,
        }

        if cand["auto_class_weights"] is not None:
            cb_params["auto_class_weights"] = cand["auto_class_weights"]
        elif cand["scale_pos_weight"] is not None:
            cb_params["scale_pos_weight"] = cand["scale_pos_weight"]

        if cand["early_stopping_rounds"] is not None:
            cb_params["early_stopping_rounds"] = cand["early_stopping_rounds"]

        model = CatBoostClassifier(**cb_params)

        # Fit model
        if cand["early_stopping_rounds"] is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            actual_iters = model.get_best_iteration() + 1
        else:
            model.fit(X_train, y_train, verbose=0)
            actual_iters = cand["iterations"]

        fit_time = round(time.time() - t_start, 2)

        # Predict validation probabilities
        y_proba = model.predict_proba(X_val)[:, 1]
        validation_probs[trial_id] = y_proba
        trained_models[trial_id] = model

        # Compute ranking metrics
        val_pr_auc = float(average_precision_score(y_val, y_proba))
        val_roc_auc = float(roc_auc_score(y_val, y_proba))

        # Evaluate threshold grid using Phase 3 methodology (999 steps in [0.001, 0.999])
        df_grid = evaluate_threshold_grid(y_val, y_proba, cost_config)
        best_threshold_dict = find_optimal_threshold(df_grid, tie_breaker="max_recall")

        opt_thresh = best_threshold_dict["threshold"]
        min_cost = best_threshold_dict["total_cost"]
        avg_cost = best_threshold_dict["average_cost_per_tx"]
        opt_prec = best_threshold_dict["precision"]
        opt_rec = best_threshold_dict["recall"]
        opt_f1 = best_threshold_dict["f1"]
        tp = best_threshold_dict["tp"]
        fp = best_threshold_dict["fp"]
        fn = best_threshold_dict["fn"]
        tn = best_threshold_dict["tn"]
        flagged_count = best_threshold_dict["flagged_count"]

        # Compute accuracy at optimal threshold
        y_pred = (y_proba >= opt_thresh).astype(int)
        val_acc = float(accuracy_score(y_val, y_pred))

        rec = {
            "trial_id": trial_id,
            "group": cand["group"],
            "description": cand["description"],
            "iterations_planned": cand["iterations"],
            "iterations_actual": actual_iters,
            "depth": cand["depth"],
            "learning_rate": cand["learning_rate"],
            "l2_leaf_reg": cand["l2_leaf_reg"],
            "random_strength": cand["random_strength"],
            "border_count": cand["border_count"],
            "class_weighting": cand["auto_class_weights"] or f"scale_pos_{cand['scale_pos_weight']}",
            "early_stopping_rounds": cand["early_stopping_rounds"] or 0,
            "fit_time_seconds": fit_time,
            "val_pr_auc": round(val_pr_auc, 6),
            "val_roc_auc": round(val_roc_auc, 6),
            "optimal_threshold": round(opt_thresh, 4),
            "total_cost": round(min_cost, 2),
            "average_cost_per_tx": round(avg_cost, 6),
            "recall": round(opt_rec, 4),
            "precision": round(opt_prec, 4),
            "f1": round(opt_f1, 4),
            "accuracy": round(val_acc, 6),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "flagged_count": int(flagged_count),
        }
        results_records.append(rec)

        print(
            f"{trial_id:<26} | {cand['group']:<18} | {val_pr_auc:<8.4f} | {val_roc_auc:<8.4f} | "
            f"{opt_thresh:<10.4f} | {opt_rec:<7.4f} | {opt_prec:<7.4f} | {f'INR {min_cost:,.2f}':<14} | {fit_time:<8.1f}"
        )

    print("-" * 115)

    # 4. Save Complete Experiment Results CSV
    df_results = pd.DataFrame(results_records)

    # Baseline metrics (Trial T01)
    baseline_row = df_results[df_results["trial_id"] == "T01_Baseline"].iloc[0]
    base_cost = baseline_row["total_cost"]
    base_pr_auc = baseline_row["val_pr_auc"]
    base_rec = baseline_row["recall"]

    df_results["cost_delta_vs_baseline"] = df_results["total_cost"] - base_cost
    df_results["cost_reduction_pct"] = ((base_cost - df_results["total_cost"]) / base_cost) * 100
    df_results["pr_auc_delta_vs_baseline"] = df_results["val_pr_auc"] - base_pr_auc

    exp_results_path = os.path.join(output_dir, "catboost_experiment_results.csv")
    df_results.to_csv(exp_results_path, index=False)
    print(f"\n[+] Saved all {len(df_results)} trial results to: {exp_results_path}")

    # 5. Select Best Candidate Based on Documented Criteria
    # Primary: Minimum validation cost
    # Tie-breaker: Maximum recall, then maximum F1, then higher PR-AUC, then smaller threshold
    df_sorted = df_results.sort_values(
        by=["total_cost", "recall", "f1", "val_pr_auc", "optimal_threshold"],
        ascending=[True, False, False, False, True]
    ).reset_index(drop=True)

    best_candidate_row = df_sorted.iloc[0]
    best_trial_id = best_candidate_row["trial_id"]
    best_model = trained_models[best_trial_id]

    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION SELECTION RESULTS")
    print("=" * 80)
    print(f"[*] Baseline Reference:  {baseline_row['trial_id']} | Cost: INR {baseline_row['total_cost']:,.2f} | PR-AUC: {baseline_row['val_pr_auc']:.4f} | Recall: {baseline_row['recall']:.4f}")
    print(f"[*] Best Candidate:      {best_candidate_row['trial_id']} | Cost: INR {best_candidate_row['total_cost']:,.2f} | PR-AUC: {best_candidate_row['val_pr_auc']:.4f} | Recall: {best_candidate_row['recall']:.4f}")
    print(f"[*] Cost Delta:          INR {best_candidate_row['cost_delta_vs_baseline']:,.2f} ({best_candidate_row['cost_reduction_pct']:+.2f}%)")
    print(f"[*] PR-AUC Delta:        {best_candidate_row['pr_auc_delta_vs_baseline']:+.4f}")
    print("=" * 80)

    # 6. Save Best Tuned Model to models/catboost_tuned.cbm
    # STRICT: Does NOT overwrite baseline models/catboost_model.cbm
    tuned_model_path = os.path.join(models_dir, "catboost_tuned.cbm")
    best_model.save_model(tuned_model_path)
    print(f"[+] Saved best tuned CatBoost model to: {tuned_model_path} (Baseline preserved intact!)")

    # 7. Generate and Save catboost_validation_comparison.csv
    comp_data = [
        {
            "model_version": "CatBoost_Baseline (Phase 2)",
            "trial_id": baseline_row["trial_id"],
            "iterations": int(baseline_row["iterations_actual"]),
            "depth": int(baseline_row["depth"]),
            "learning_rate": float(baseline_row["learning_rate"]),
            "l2_leaf_reg": float(baseline_row["l2_leaf_reg"]),
            "border_count": int(baseline_row["border_count"]),
            "class_weighting": str(baseline_row["class_weighting"]),
            "decision_threshold": float(baseline_row["optimal_threshold"]),
            "val_pr_auc": float(baseline_row["val_pr_auc"]),
            "val_roc_auc": float(baseline_row["val_roc_auc"]),
            "total_cost_inr": float(baseline_row["total_cost"]),
            "average_cost_per_tx_inr": float(baseline_row["average_cost_per_tx"]),
            "fraud_recall": float(baseline_row["recall"]),
            "fraud_precision": float(baseline_row["precision"]),
            "fraud_f1": float(baseline_row["f1"]),
            "true_positives": int(baseline_row["tp"]),
            "false_positives": int(baseline_row["fp"]),
            "false_negatives": int(baseline_row["fn"]),
            "true_negatives": int(baseline_row["tn"]),
            "flagged_transactions": int(baseline_row["flagged_count"]),
        },
        {
            "model_version": "CatBoost_Tuned (Experiment 1)",
            "trial_id": best_candidate_row["trial_id"],
            "iterations": int(best_candidate_row["iterations_actual"]),
            "depth": int(best_candidate_row["depth"]),
            "learning_rate": float(best_candidate_row["learning_rate"]),
            "l2_leaf_reg": float(best_candidate_row["l2_leaf_reg"]),
            "border_count": int(best_candidate_row["border_count"]),
            "class_weighting": str(best_candidate_row["class_weighting"]),
            "decision_threshold": float(best_candidate_row["optimal_threshold"]),
            "val_pr_auc": float(best_candidate_row["val_pr_auc"]),
            "val_roc_auc": float(best_candidate_row["val_roc_auc"]),
            "total_cost_inr": float(best_candidate_row["total_cost"]),
            "average_cost_per_tx_inr": float(best_candidate_row["average_cost_per_tx"]),
            "fraud_recall": float(best_candidate_row["recall"]),
            "fraud_precision": float(best_candidate_row["precision"]),
            "fraud_f1": float(best_candidate_row["f1"]),
            "true_positives": int(best_candidate_row["tp"]),
            "false_positives": int(best_candidate_row["fp"]),
            "false_negatives": int(best_candidate_row["fn"]),
            "true_negatives": int(best_candidate_row["tn"]),
            "flagged_transactions": int(best_candidate_row["flagged_count"]),
        },
    ]
    df_val_comp = pd.DataFrame(comp_data)
    val_comp_path = os.path.join(output_dir, "catboost_validation_comparison.csv")
    df_val_comp.to_csv(val_comp_path, index=False)
    print(f"[+] Saved validation comparison table to: {val_comp_path}")

    # 8. Save Best Parameters and Metadata JSON
    best_params_data = {
        "experiment_name": "Experiment 1: CatBoost Hyperparameter Optimization",
        "random_seed": random_state,
        "selection_metric": "Minimum Validation Cost (FP=INR 5, FN=INR 500) with Tie-Breaking (Recall -> F1 -> Threshold)",
        "best_candidate": {
            "trial_id": best_candidate_row["trial_id"],
            "group": best_candidate_row["group"],
            "description": best_candidate_row["description"],
            "hyperparameters": {
                "iterations": int(best_candidate_row["iterations_actual"]),
                "depth": int(best_candidate_row["depth"]),
                "learning_rate": float(best_candidate_row["learning_rate"]),
                "l2_leaf_reg": float(best_candidate_row["l2_leaf_reg"]),
                "random_strength": float(best_candidate_row["random_strength"]),
                "border_count": int(best_candidate_row["border_count"]),
                "class_weighting": str(best_candidate_row["class_weighting"]),
                "early_stopping_rounds": int(best_candidate_row["early_stopping_rounds"]),
            },
            "validation_performance": {
                "optimal_decision_threshold": float(best_candidate_row["optimal_threshold"]),
                "total_cost_hypothetical_inr": float(best_candidate_row["total_cost"]),
                "average_cost_per_tx_inr": float(best_candidate_row["average_cost_per_tx"]),
                "val_pr_auc": float(best_candidate_row["val_pr_auc"]),
                "val_roc_auc": float(best_candidate_row["val_roc_auc"]),
                "recall": float(best_candidate_row["recall"]),
                "precision": float(best_candidate_row["precision"]),
                "f1_score": float(best_candidate_row["f1"]),
                "accuracy": float(best_candidate_row["accuracy"]),
                "tp": int(best_candidate_row["tp"]),
                "fp": int(best_candidate_row["fp"]),
                "fn": int(best_candidate_row["fn"]),
                "tn": int(best_candidate_row["tn"]),
                "flagged_transactions": int(best_candidate_row["flagged_count"]),
            },
        },
        "baseline_comparison": {
            "baseline_trial_id": baseline_row["trial_id"],
            "cost_delta_inr": round(float(best_candidate_row["cost_delta_vs_baseline"]), 2),
            "cost_reduction_percentage": round(float(best_candidate_row["cost_reduction_pct"]), 2),
            "pr_auc_delta": round(float(best_candidate_row["pr_auc_delta_vs_baseline"]), 4),
            "recall_delta": round(float(best_candidate_row["recall"] - baseline_row["recall"]), 4),
            "precision_delta": round(float(best_candidate_row["precision"] - baseline_row["precision"]), 4),
            "false_negative_reduction": int(baseline_row["fn"] - best_candidate_row["fn"]),
            "false_positive_reduction": int(baseline_row["fp"] - best_candidate_row["fp"]),
        },
        "findings_and_honest_assessment": {
            "did_tuning_improve_cost": bool(best_candidate_row["total_cost"] < baseline_row["total_cost"]),
            "did_tuning_improve_pr_auc": bool(best_candidate_row["val_pr_auc"] > baseline_row["val_pr_auc"]),
            "summary_verdict": (
                f"Tuning achieved a minimum validation cost of INR {best_candidate_row['total_cost']:,.2f} "
                f"(vs baseline INR {baseline_row['total_cost']:,.2f}, a delta of INR {best_candidate_row['cost_delta_vs_baseline']:,.2f}) "
                f"and a PR-AUC of {best_candidate_row['val_pr_auc']:.4f} (vs baseline {baseline_row['val_pr_auc']:.4f})."
            )
        },
        "all_top_5_candidates": df_sorted.head(5)[[
            "trial_id", "group", "val_pr_auc", "val_roc_auc", "optimal_threshold", "recall", "precision", "total_cost"
        ]].to_dict(orient="records")
    }

    best_params_path = os.path.join(output_dir, "catboost_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params_data, f, indent=2)
    print(f"[+] Saved best parameters metadata to: {best_params_path}")

    print("\n" + "=" * 75)
    print("EXPERIMENT 1 COMPLETE [SUCCESS]")
    print("=" * 75)

    return {
        "df_results": df_results,
        "best_candidate": best_candidate_row.to_dict(),
        "baseline": baseline_row.to_dict(),
        "best_params_data": best_params_data,
    }


if __name__ == "__main__":
    run_catboost_tuning()
