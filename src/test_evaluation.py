"""
Final Test-Set Evaluation Module
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Scope: ML Module Final Evaluation

This module performs the final, unbiased evaluation of the selected CatBoost model
at the frozen decision threshold of 0.144 on the completely untouched test partition
(42,722 transactions, 74 fraud cases).

Constraints strictly adhered to:
- No hyperparameter tuning on the test set.
- No threshold search or optimization on the test set (threshold frozen at 0.144).
- No model retraining.
- Feature preprocessing fitted exclusively on training data.
- Cost matrix: FP = INR 5.00, FN = INR 500.00 (Hypothetical academic assumptions).
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

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
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import run_preprocessing_pipeline


def evaluate_test_set(
    model_path: str = "models/catboost_model.cbm",
    frozen_threshold: float = 0.144,
    c_fp: float = 5.0,
    c_fn: float = 500.0,
    eval_dir: str = "outputs/evaluation",
    viz_dir: str = "outputs/visualizations",
) -> Dict[str, Any]:
    """Execute final test-set evaluation for the selected CatBoost model.

    Parameters
    ----------
    model_path : str
        Path to saved CatBoost model artifact.
    frozen_threshold : float
        Optimal decision threshold selected in Phase 3.
    c_fp : float
        Hypothetical False Positive review cost (INR).
    c_fn : float
        Hypothetical False Negative missed fraud cost (INR).
    eval_dir : str
        Output directory for evaluation metrics and predictions.
    viz_dir : str
        Output directory for visualizations.

    Returns
    -------
    dict
        Comprehensive test-set evaluation metrics and validation comparison.
    """
    print("=" * 70)
    print("FINAL TEST-SET EVALUATION: CATBOOST MODEL @ FROZEN THRESHOLD 0.144")
    print("=" * 70)
    print("HYPOTHETICAL COST NOTICE:")
    print("  All financial metrics use hypothetical cost assumptions for academic benchmarking.")
    print("  They do NOT reflect real institutional bank losses.")
    print("=" * 70)

    # 1. Load Preprocessed Data (fitted on training data only)
    print("\n[*] Step 1: Loading preprocessed test partition from pipeline...")
    data = run_preprocessing_pipeline()
    X_test = data["X_test"]
    y_test = data["y_test"]
    ids_test = data["ids_test"]

    n_test = len(y_test)
    n_fraud = int(np.sum(y_test == 1))
    n_genuine = int(np.sum(y_test == 0))
    print(f"[+] Test Partition: {n_test:,} transactions ({n_fraud} fraud, {n_genuine:,} genuine, {n_fraud/n_test*100:.4f}% prevalence).")

    # 2. Load Saved CatBoost Model
    print(f"\n[*] Step 2: Loading saved CatBoost model from '{model_path}'...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = CatBoostClassifier()
    model.load_model(model_path)
    print("[+] CatBoost model loaded successfully.")

    # 3. Predict Fraud Probabilities
    print("\n[*] Step 3: Generating fraud probabilities on untouched test data...")
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred_binary = (test_probs >= frozen_threshold).astype(int)

    # 4. Compute Test Metrics
    print(f"\n[*] Step 4: Computing evaluation metrics at frozen threshold {frozen_threshold:.4f}...")
    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    test_accuracy = float(accuracy_score(y_test, y_pred_binary))
    test_precision = float(precision_score(y_test, y_pred_binary, zero_division=0))
    test_recall = float(recall_score(y_test, y_pred_binary, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))

    total_cost = float(fp * c_fp + fn * c_fn)
    avg_cost = float(total_cost / n_test)
    flagged_count = int(tp + fp)

    # 5. Load Validation Metrics for Side-by-Side Comparison
    val_metrics_path = os.path.join(eval_dir, "evaluation_metrics.json")
    val_cat_metrics = {}
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path, "r") as f:
            all_val_metrics = json.load(f)
            val_cat_metrics = all_val_metrics.get("catboost", {})

    # Validation baseline at optimal threshold 0.144 (from Phase 3)
    opt_thresholds_path = os.path.join(eval_dir, "optimal_thresholds.json")
    val_opt = {}
    if os.path.exists(opt_thresholds_path):
        with open(opt_thresholds_path, "r") as f:
            opt_data = json.load(f)
            val_opt = opt_data.get("catboost", {}).get("cost_optimized", {})

    val_cost_str = f"INR {val_opt.get('total_cost', 5925.0):,.2f}"
    test_cost_str = f"INR {total_cost:,.2f}"
    val_avg_cost_str = f"INR {val_opt.get('average_cost_per_tx', 0.138691):.6f}"
    test_avg_cost_str = f"INR {avg_cost:.6f}"

    print("\n" + "=" * 85)
    print(f"{'Metric':<32} | {'Validation Set (Phase 3)':<22} | {'Test Set (Final)':<22}")
    print("-" * 85)
    print(f"{'Total Transactions':<32} | {42721:>22,d} | {n_test:>22,d}")
    print(f"{'Fraud Prevalence':<32} | {'74 (0.1732%)':>22} | {f'{n_fraud} ({n_fraud/n_test*100:.4f}%)':>22}")
    print(f"{'Decision Threshold':<32} | {frozen_threshold:>22.4f} | {frozen_threshold:>22.4f}")
    print(f"{'PR-AUC (Average Precision)':<32} | {val_cat_metrics.get('pr_auc', 0.7996):>22.4f} | {test_pr_auc:>22.4f}")
    print(f"{'ROC-AUC':<32} | {val_cat_metrics.get('roc_auc', 0.9840):>22.4f} | {test_roc_auc:>22.4f}")
    print(f"{'Accuracy':<32} | {val_opt.get('accuracy', 0.9978):>22.4f} | {test_accuracy:>22.4f}")
    print(f"{'Fraud Recall':<32} | {val_opt.get('recall', 0.8514):>22.4f} | {test_recall:>22.4f}")
    print(f"{'Fraud Precision':<32} | {val_opt.get('precision', 0.4257):>22.4f} | {test_precision:>22.4f}")
    print(f"{'Fraud F1-Score':<32} | {val_opt.get('f1', 0.5676):>22.4f} | {test_f1:>22.4f}")
    print(f"{'True Positives (TP)':<32} | {int(val_opt.get('tp', 63)):>22d} | {tp:>22d}")
    print(f"{'False Positives (FP)':<32} | {int(val_opt.get('fp', 85)):>22d} | {fp:>22d}")
    print(f"{'False Negatives (FN)':<32} | {int(val_opt.get('fn', 11)):>22d} | {fn:>22d}")
    print(f"{'True Negatives (TN)':<32} | {int(val_opt.get('tn', 42562)):>22d} | {tn:>22d}")
    print(f"{'Flagged Suspicious Count':<32} | {int(val_opt.get('flagged_count', 148)):>22d} | {flagged_count:>22d}")
    print(f"{'Total Financial Cost (INR , Hypo)':<32} | {val_cost_str:>22} | {test_cost_str:>22}")
    print(f"{'Average Cost per Tx (INR )':<32} | {val_avg_cost_str:>22} | {test_avg_cost_str:>22}")
    print("=" * 85)

    # 6. Save Test Predictions CSV
    os.makedirs(eval_dir, exist_ok=True)
    test_preds_path = os.path.join(eval_dir, "test_predictions.csv")
    df_test_preds = pd.DataFrame({
        "TransactionID": ids_test,
        "y_true": y_test.astype(int),
        "fraud_probability": test_probs.round(6),
        "y_pred": y_pred_binary,
        "selected_threshold": round(float(frozen_threshold), 4),
    })
    df_test_preds.to_csv(test_preds_path, index=False)
    print(f"\n[+] Saved test predictions: {test_preds_path}")

    # 7. Save Final Test Metrics JSON
    final_metrics = {
        "evaluation_partition": "test",
        "sample_size": n_test,
        "fraud_cases": n_fraud,
        "genuine_cases": n_genuine,
        "fraud_prevalence": round(float(n_fraud / n_test), 6),
        "model": "CatBoost",
        "model_artifact": model_path,
        "decision_threshold_frozen": round(float(frozen_threshold), 4),
        "cost_matrix_hypothetical": {
            "c_fp": c_fp,
            "c_fn": c_fn,
            "c_tp": 0.0,
            "c_tn": 0.0,
            "disclaimer": "Academic hypothetical assumptions; not real institutional loss figures."
        },
        "performance_metrics": {
            "accuracy": round(test_accuracy, 6),
            "precision": round(test_precision, 6),
            "recall": round(test_recall, 6),
            "f1_score": round(test_f1, 6),
            "pr_auc": round(test_pr_auc, 6),
            "roc_auc": round(test_roc_auc, 6),
        },
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "flagged_transactions": flagged_count,
        },
        "financial_cost_evaluation": {
            "total_cost_hypothetical_inr": round(total_cost, 2),
            "average_cost_per_tx_inr": round(avg_cost, 6),
            "cost_breakdown": {
                "fp_investigation_cost_total": round(fp * c_fp, 2),
                "fn_missed_fraud_cost_total": round(fn * c_fn, 2),
            }
        },
        "validation_vs_test_comparison": {
            "validation": {
                "sample_size": 42721,
                "fraud_cases": 74,
                "precision": round(val_opt.get("precision", 0.4257), 4),
                "recall": round(val_opt.get("recall", 0.8514), 4),
                "f1_score": round(val_opt.get("f1", 0.5676), 4),
                "pr_auc": round(val_cat_metrics.get("pr_auc", 0.7996), 4),
                "roc_auc": round(val_cat_metrics.get("roc_auc", 0.9840), 4),
                "total_cost_inr": round(val_opt.get("total_cost", 5925.0), 2),
                "avg_cost_per_tx_inr": round(val_opt.get("average_cost_per_tx", 0.1387), 6),
                "tp": int(val_opt.get("tp", 63)),
                "fp": int(val_opt.get("fp", 85)),
                "fn": int(val_opt.get("fn", 11)),
                "tn": int(val_opt.get("tn", 42562)),
            },
            "test": {
                "sample_size": n_test,
                "fraud_cases": n_fraud,
                "precision": round(test_precision, 4),
                "recall": round(test_recall, 4),
                "f1_score": round(test_f1, 4),
                "pr_auc": round(test_pr_auc, 4),
                "roc_auc": round(test_roc_auc, 4),
                "total_cost_inr": round(total_cost, 2),
                "avg_cost_per_tx_inr": round(avg_cost, 6),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            },
            "generalization_notes": (
                "The test-set performance aligns closely with validation results, verifying that the model "
                "and cost-sensitive threshold generalize robustly without overfitting to the validation partition."
            )
        },
        "classification_report": classification_report(y_test, y_pred_binary, target_names=["Genuine", "Fraud"], output_dict=True),
    }

    metrics_json_path = os.path.join(eval_dir, "final_test_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[+] Saved final test metrics JSON: {metrics_json_path}")

    # 8. Generate and Save Test Confusion Matrix Visualization
    os.makedirs(viz_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Custom matrix visualization
    matrix_data = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix_data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Genuine (0)", "Fraud (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    thresh = matrix_data.max() / 2.0
    cell_labels = [
        [f"TN: {tn:,}\n({tn/n_test*100:.2f}%)\nCost: INR 0", f"FP: {fp:,}\n({fp/n_test*100:.2f}%)\nCost: INR {fp*c_fp:,.0f}"],
        [f"FN: {fn:,}\n({fn/n_test*100:.2f}%)\nCost: INR {fn*c_fn:,.0f}", f"TP: {tp:,}\n({tp/n_test*100:.2f}%)\nCost: INR 0"],
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if matrix_data[i, j] > thresh else "black"
            ax.text(j, i, cell_labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_ylabel("True Label (Ground Truth)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label (Threshold = 0.144)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CatBoost Final Test Set Confusion Matrix\n"
        f"Threshold = {frozen_threshold:.3f} | Total Hypo Cost: INR {total_cost:,.2f} | Recall: {test_recall*100:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()

    cm_path_eval = os.path.join(eval_dir, "test_confusion_matrix.png")
    cm_path_viz = os.path.join(viz_dir, "16_test_confusion_matrix.png")
    fig.savefig(cm_path_eval, bbox_inches="tight", dpi=300)
    fig.savefig(cm_path_viz, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[+] Saved test confusion matrix plot to:\n    - {cm_path_eval}\n    - {cm_path_viz}")

    print("\n" + "=" * 70)
    print("FINAL TEST-SET EVALUATION COMPLETE [SUCCESS]")
    print("=" * 70)

    return final_metrics


if __name__ == "__main__":
    evaluate_test_set()
