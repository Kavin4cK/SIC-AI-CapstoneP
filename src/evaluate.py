"""
Model Evaluation Module
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Phase 2: Machine Learning Model Development

This script:
1. Loads the trained LightGBM and CatBoost models.
2. Evaluates both models independently on the validation partition (X_val, y_val, ids_val).
3. Calculates comprehensive classification metrics:
   - Accuracy, Precision, Recall, F1-Score (binary & macro/weighted)
   - PR-AUC (Average Precision) & ROC-AUC
   - Confusion Matrices & Classification Reports
4. Emphasizes fraud-class precision and recall.
5. Saves tabular comparison results:
   - outputs/evaluation/model_comparison.csv
   - outputs/evaluation/classification_reports.txt
   - outputs/evaluation/evaluation_metrics.json
6. Generates evaluation visualizations:
   - outputs/visualizations/06_pr_curves_comparison.png
   - outputs/visualizations/07_roc_curves_comparison.png
   - outputs/visualizations/08_confusion_matrices.png
   - outputs/visualizations/09_feature_importances.png
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to sys.path so 'src' is discoverable regardless of working directory
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variables for headless matplotlib execution
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
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
    precision_recall_curve,
    roc_curve,
)

# Import Phase 1 reusable preprocessing pipeline
from src.preprocessing import run_preprocessing_pipeline


def load_trained_models(
    models_dir: str = "models"
) -> Tuple[lgb.Booster, CatBoostClassifier]:
    """Load persisted models from disk.

    Parameters
    ----------
    models_dir : str
        Directory containing model artifacts.

    Returns
    -------
    Tuple[lgb.Booster, CatBoostClassifier]
        Loaded LightGBM booster and CatBoostClassifier.
    """
    lgb_path = os.path.join(models_dir, "lightgbm_model.txt")
    cat_path = os.path.join(models_dir, "catboost_model.cbm")

    if not os.path.exists(lgb_path):
        raise FileNotFoundError(f"LightGBM model not found at: {lgb_path}. Run src/train.py first.")
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"CatBoost model not found at: {cat_path}. Run src/train.py first.")

    print(f"[*] Loading LightGBM model from: {lgb_path}...")
    lgb_booster = lgb.Booster(model_file=lgb_path)

    print(f"[*] Loading CatBoost model from: {cat_path}...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)

    print("[+] Both models loaded successfully.")
    return lgb_booster, cat_model


def compute_model_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Calculate all evaluation metrics for a single model on validation data.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth binary labels (0 = Genuine, 1 = Fraud).
    y_pred : np.ndarray
        Predicted binary labels.
    y_proba : np.ndarray
        Predicted fraud probabilities in [0, 1].
    model_name : str
        Display name of the model.

    Returns
    -------
    dict
        Dictionary containing scalar metrics, confusion matrix, and report.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_fraud = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_fraud = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fraud = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    clf_report_dict = classification_report(y_true, y_pred, target_names=["Genuine", "Fraud"], output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_true, y_pred, target_names=["Genuine", "Fraud"], digits=4, zero_division=0)

    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_fraud": prec_fraud,
        "recall_fraud": rec_fraud,
        "f1_fraud": f1_fraud,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": clf_report_dict,
        "classification_report_text": clf_report_text,
    }


def export_metrics_and_comparison(
    lgb_metrics: Dict[str, Any],
    cat_metrics: Dict[str, Any],
    output_dir: str = "outputs/evaluation",
) -> pd.DataFrame:
    """Save evaluation metrics comparison in CSV, JSON, and text formats.

    Parameters
    ----------
    lgb_metrics, cat_metrics : dict
        Calculated metrics for both models.
    output_dir : str
        Directory to save results.

    Returns
    -------
    pd.DataFrame
        Model comparison summary DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = [
        {
            "Model": lgb_metrics["model_name"],
            "PR-AUC (Avg Precision)": round(lgb_metrics["pr_auc"], 4),
            "ROC-AUC": round(lgb_metrics["roc_auc"], 4),
            "Fraud Precision": round(lgb_metrics["precision_fraud"], 4),
            "Fraud Recall": round(lgb_metrics["recall_fraud"], 4),
            "Fraud F1-Score": round(lgb_metrics["f1_fraud"], 4),
            "Accuracy": round(lgb_metrics["accuracy"], 6),
            "True Positives (TP)": lgb_metrics["true_positives"],
            "False Positives (FP)": lgb_metrics["false_positives"],
            "False Negatives (FN)": lgb_metrics["false_negatives"],
            "True Negatives (TN)": lgb_metrics["true_negatives"],
            "Macro F1": round(lgb_metrics["f1_macro"], 4),
            "Weighted F1": round(lgb_metrics["f1_weighted"], 4),
        },
        {
            "Model": cat_metrics["model_name"],
            "PR-AUC (Avg Precision)": round(cat_metrics["pr_auc"], 4),
            "ROC-AUC": round(cat_metrics["roc_auc"], 4),
            "Fraud Precision": round(cat_metrics["precision_fraud"], 4),
            "Fraud Recall": round(cat_metrics["recall_fraud"], 4),
            "Fraud F1-Score": round(cat_metrics["f1_fraud"], 4),
            "Accuracy": round(cat_metrics["accuracy"], 6),
            "True Positives (TP)": cat_metrics["true_positives"],
            "False Positives (FP)": cat_metrics["false_positives"],
            "False Negatives (FN)": cat_metrics["false_negatives"],
            "True Negatives (TN)": cat_metrics["true_negatives"],
            "Macro F1": round(cat_metrics["f1_macro"], 4),
            "Weighted F1": round(cat_metrics["f1_weighted"], 4),
        },
    ]

    df_comp = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[+] Saved model comparison table: {csv_path}")

    # Export full JSON
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"LightGBM": lgb_metrics, "CatBoost": cat_metrics}, f, indent=2)
    print(f"[+] Saved complete metrics JSON: {json_path}")

    # Export Classification Reports text
    txt_path = os.path.join(output_dir, "classification_reports.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 2 MODEL EVALUATION: CLASSIFICATION REPORTS (VALIDATION SET)\n")
        f.write("=" * 70 + "\n\n")
        f.write("--- 1. LIGHTGBM CLASSIFIER ---\n")
        f.write(lgb_metrics["classification_report_text"] + "\n\n")
        f.write("--- 2. CATBOOST CLASSIFIER ---\n")
        f.write(cat_metrics["classification_report_text"] + "\n")
    print(f"[+] Saved classification reports text: {txt_path}")

    return df_comp


def generate_evaluation_visualizations(
    y_val: pd.Series,
    lgb_proba: np.ndarray,
    cat_proba: np.ndarray,
    lgb_pred: np.ndarray,
    cat_pred: np.ndarray,
    lgb_booster: lgb.Booster,
    cat_model: CatBoostClassifier,
    feature_names: list,
    output_dir: str = "outputs/visualizations",
) -> None:
    """Generate high-quality evaluation visualization plots.

    Parameters
    ----------
    y_val : pd.Series
        Validation target labels.
    lgb_proba, cat_proba : np.ndarray
        Predicted fraud probabilities.
    lgb_pred, cat_pred : np.ndarray
        Predicted binary labels.
    lgb_booster, cat_model : objects
        Trained model objects for feature importances.
    feature_names : list
        List of feature column names.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("GENERATING MODEL EVALUATION VISUALIZATIONS")
    print("=" * 70)

    # Plot styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Visualization 06: Precision-Recall Curves Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))

    p_lgb, r_lgb, _ = precision_recall_curve(y_val, lgb_proba)
    p_cat, r_cat, _ = precision_recall_curve(y_val, cat_proba)
    ap_lgb = average_precision_score(y_val, lgb_proba)
    ap_cat = average_precision_score(y_val, cat_proba)

    baseline_prevalence = y_val.mean()

    ax.plot(r_lgb, p_lgb, color="#2b5c8f", linewidth=2.2, label=f"LightGBM (PR-AUC = {ap_lgb:.4f})")
    ax.plot(r_cat, p_cat, color="#d9534f", linewidth=2.2, linestyle="--", label=f"CatBoost (PR-AUC = {ap_cat:.4f})")
    ax.axhline(baseline_prevalence, color="gray", linestyle=":", linewidth=1.5, label=f"No-Skill Baseline ({baseline_prevalence*100:.3f}%)")

    ax.set_title("Precision-Recall Curves Comparison (Validation Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Fraud Class)", fontsize=11)
    ax.set_ylabel("Precision (Fraud Class)", fontsize=11)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    v6_path = os.path.join(output_dir, "06_pr_curves_comparison.png")
    plt.savefig(v6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v6_path}")

    # -------------------------------------------------------------------------
    # Visualization 07: ROC Curves Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))

    fpr_lgb, tpr_lgb, _ = roc_curve(y_val, lgb_proba)
    fpr_cat, tpr_cat, _ = roc_curve(y_val, cat_proba)
    auc_lgb = roc_auc_score(y_val, lgb_proba)
    auc_cat = roc_auc_score(y_val, cat_proba)

    ax.plot(fpr_lgb, tpr_lgb, color="#2b5c8f", linewidth=2.2, label=f"LightGBM (ROC-AUC = {auc_lgb:.4f})")
    ax.plot(fpr_cat, tpr_cat, color="#d9534f", linewidth=2.2, linestyle="--", label=f"CatBoost (ROC-AUC = {auc_cat:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Chance (AUC = 0.5000)")

    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    v7_path = os.path.join(output_dir, "07_roc_curves_comparison.png")
    plt.savefig(v7_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v7_path}")

    # -------------------------------------------------------------------------
    # Visualization 08: Confusion Matrices Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cm_lgb = confusion_matrix(y_val, lgb_pred)
    cm_cat = confusion_matrix(y_val, cat_pred)

    def draw_cm(ax, cm, title, cmap):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Genuine (0)", "Fraud (1)"], fontsize=10)
        ax.set_yticklabels(["Genuine (0)", "Fraud (1)"], fontsize=10)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

        # Annotate counts and percentages
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm[i, j] / cm[i].sum() * 100
                cell_text = f"{cm[i, j]:,}\n({pct:.1f}%)"
                ax.text(j, i, cell_text, ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight="bold")

    draw_cm(ax1, cm_lgb, f"LightGBM Confusion Matrix\n(TP: {cm_lgb[1,1]}, FP: {cm_lgb[0,1]}, FN: {cm_lgb[1,0]})", plt.cm.Blues)
    draw_cm(ax2, cm_cat, f"CatBoost Confusion Matrix\n(TP: {cm_cat[1,1]}, FP: {cm_cat[0,1]}, FN: {cm_cat[1,0]})", plt.cm.Reds)

    plt.suptitle("Validation Set Confusion Matrices: LightGBM vs. CatBoost", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v8_path = os.path.join(output_dir, "08_confusion_matrices.png")
    plt.savefig(v8_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v8_path}")

    # -------------------------------------------------------------------------
    # Visualization 09: Top Feature Importances Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # LightGBM feature importance (gain)
    lgb_imp = lgb_booster.feature_importance(importance_type="gain")
    lgb_df = pd.DataFrame({"Feature": feature_names, "Importance": lgb_imp}).sort_values("Importance", ascending=False).head(12)

    ax1.barh(lgb_df["Feature"][::-1], lgb_df["Importance"][::-1], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax1.set_title("LightGBM Top 12 Feature Importance (Gain)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Importance (Gain)", fontsize=11)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # CatBoost feature importance
    cat_imp = cat_model.get_feature_importance()
    cat_df = pd.DataFrame({"Feature": feature_names, "Importance": cat_imp}).sort_values("Importance", ascending=False).head(12)

    ax2.barh(cat_df["Feature"][::-1], cat_df["Importance"][::-1], color="#d9534f", edgecolor="black", alpha=0.85)
    ax2.set_title("CatBoost Top 12 Feature Importance (Prediction Values)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Importance Score", fontsize=11)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Model Feature Importances: Dominant Discriminative Predictors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v9_path = os.path.join(output_dir, "09_feature_importances.png")
    plt.savefig(v9_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v9_path}")


def run_evaluation(
    data_path: str = "data/creditcard.csv",
    models_dir: str = "models",
    output_dir: str = "outputs/evaluation",
    viz_dir: str = "outputs/visualizations",
) -> Dict[str, Any]:
    """Execute end-to-end evaluation workflow.

    Parameters
    ----------
    data_path : str
        Path to creditcard.csv.
    models_dir : str
        Directory where models are saved.
    output_dir : str
        Directory to save tabular outputs.
    viz_dir : str
        Directory to save plots.

    Returns
    -------
    dict
        Evaluation results and summary table.
    """
    print("=" * 70)
    print("PHASE 2: MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    # 1. Load preprocessed splits
    pipeline_data = run_preprocessing_pipeline(data_path=data_path)
    X_val = pipeline_data["X_val"]
    y_val = pipeline_data["y_val"]
    ids_val = pipeline_data["ids_val"]
    feature_names = list(X_val.columns)

    print(f"\n[*] Evaluating on Validation Partition:")
    print(f"    - Total Validation Samples: {len(X_val):,}")
    print(f"    - Genuine Samples:         {(y_val == 0).sum():,} ({(y_val == 0).mean()*100:.3f}%)")
    print(f"    - Fraudulent Samples:      {(y_val == 1).sum():,} ({(y_val == 1).mean()*100:.3f}%)")

    # 2. Load trained models
    lgb_booster, cat_model = load_trained_models(models_dir=models_dir)

    # 3. Generate validation predictions and probabilities
    print("\n[*] Generating predictions on validation data...")
    lgb_proba = lgb_booster.predict(X_val)
    lgb_pred = (lgb_proba >= 0.5).astype(int)

    cat_proba = cat_model.predict_proba(X_val)[:, 1]
    cat_pred = (cat_proba >= 0.5).astype(int)

    # 4. Compute metrics
    lgb_metrics = compute_model_metrics(y_val, lgb_pred, lgb_proba, model_name="LightGBM")
    cat_metrics = compute_model_metrics(y_val, cat_pred, cat_proba, model_name="CatBoost")

    # 5. Display comparative results
    print("\n" + "=" * 70)
    print("VALIDATION METRICS COMPARISON (DEFAULT THRESHOLD = 0.5)")
    print("=" * 70)
    print(f"{'Metric':<25} | {'LightGBM':<15} | {'CatBoost':<15}")
    print("-" * 60)
    print(f"{'PR-AUC (Avg Precision)':<25} | {lgb_metrics['pr_auc']:>15.4f} | {cat_metrics['pr_auc']:>15.4f}")
    print(f"{'ROC-AUC':<25} | {lgb_metrics['roc_auc']:>15.4f} | {cat_metrics['roc_auc']:>15.4f}")
    print(f"{'Fraud Precision':<25} | {lgb_metrics['precision_fraud']:>15.4f} | {cat_metrics['precision_fraud']:>15.4f}")
    print(f"{'Fraud Recall':<25} | {lgb_metrics['recall_fraud']:>15.4f} | {cat_metrics['recall_fraud']:>15.4f}")
    print(f"{'Fraud F1-Score':<25} | {lgb_metrics['f1_fraud']:>15.4f} | {cat_metrics['f1_fraud']:>15.4f}")
    print(f"{'Overall Accuracy':<25} | {lgb_metrics['accuracy']:>15.6f} | {cat_metrics['accuracy']:>15.6f}")
    print(f"{'True Positives (TP)':<25} | {lgb_metrics['true_positives']:>15d} | {cat_metrics['true_positives']:>15d}")
    print(f"{'False Positives (FP)':<25} | {lgb_metrics['false_positives']:>15d} | {cat_metrics['false_positives']:>15d}")
    print(f"{'False Negatives (FN)':<25} | {lgb_metrics['false_negatives']:>15d} | {cat_metrics['false_negatives']:>15d}")
    print(f"{'True Negatives (TN)':<25} | {lgb_metrics['true_negatives']:>15d} | {cat_metrics['true_negatives']:>15d}")

    # 6. Save comparison CSV, JSON, and text reports
    df_comp = export_metrics_and_comparison(lgb_metrics, cat_metrics, output_dir=output_dir)

    # 7. Generate evaluation visualization figures
    generate_evaluation_visualizations(
        y_val=y_val,
        lgb_proba=lgb_proba,
        cat_proba=cat_proba,
        lgb_pred=lgb_pred,
        cat_pred=cat_pred,
        lgb_booster=lgb_booster,
        cat_model=cat_model,
        feature_names=feature_names,
        output_dir=viz_dir,
    )

    print("\n" + "=" * 70)
    print("PHASE 2 MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "lgb_metrics": lgb_metrics,
        "cat_metrics": cat_metrics,
        "comparison_table": df_comp,
    }


if __name__ == "__main__":
    run_evaluation()
