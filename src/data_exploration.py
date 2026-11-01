"""
Dataset Exploration Module
Project: Cost-Sensitive Fraud Detection with an Operator Feedback Loop
Phase 1: Data Exploration and Preprocessing

This script performs comprehensive Exploratory Data Analysis (EDA) on the
credit card fraud dataset, analyzes class imbalance, Amount and Time distributions,
checks missing values and duplicates, and exports high-quality visualizations
to the outputs/visualizations/ directory.
"""

import os
import sys
import tempfile
from typing import Dict, Any, Tuple

# Set matplotlib cache directory to temp to prevent permission issues on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the credit card transaction dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the creditcard.csv dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    print(f"[*] Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df):,} rows and {len(df.columns)} columns successfully.")
    return df


def inspect_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset dimensions, columns, and data types.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Dictionary containing shape, column names, and data types.
    """
    shape = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.value_counts().to_dict()

    print("\n" + "=" * 70)
    print("1. DATASET STRUCTURE & DIMENSIONS")
    print("=" * 70)
    print(f"Total Rows (Transactions): {shape[0]:,}")
    print(f"Total Columns (Features):  {shape[1]:,}")
    print("\nData Types Summary:")
    for dtype, count in dtypes.items():
        print(f"  - {dtype}: {count} columns")

    print("\nColumn Details:")
    for i, col in enumerate(columns, 1):
        print(f"  {i:02d}. {col:<10} | dtype: {str(df[col].dtype):<10} | memory: {df[col].memory_usage(deep=True) / 1024**2:.2f} MB")

    total_mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nTotal In-Memory Size: {total_mem:.2f} MB")

    return {"shape": shape, "columns": columns, "dtypes": dtypes, "memory_mb": total_mem}


def check_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing values and duplicate records.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Dictionary containing missing value summary and duplicate counts.
    """
    print("\n" + "=" * 70)
    print("2. DATA INTEGRITY: MISSING VALUES & DUPLICATES")
    print("=" * 70)

    # Missing values
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"Missing Values Check:")
    print(f"  - Total missing cells: {total_missing}")
    if total_missing > 0:
        missing_cols = missing_series[missing_series > 0]
        for col, cnt in missing_cols.items():
            print(f"    * {col}: {cnt} missing ({cnt / len(df) * 100:.2f}%)")
    else:
        print("  - Complete dataset: zero missing values detected across all columns.")

    # Duplicate records
    duplicated_mask = df.duplicated()
    dup_count = int(duplicated_mask.sum())
    dup_by_class = df[duplicated_mask]["Class"].value_counts().to_dict() if dup_count > 0 else {}

    print(f"\nDuplicate Records Check:")
    print(f"  - Exact duplicate rows count: {dup_count:,} ({dup_count / len(df) * 100:.3f}% of dataset)")
    if dup_count > 0:
        genuine_dups = dup_by_class.get(0, 0)
        fraud_dups = dup_by_class.get(1, 0)
        print(f"    * Genuine duplicates (Class 0):   {genuine_dups:,}")
        print(f"    * Fraudulent duplicates (Class 1): {fraud_dups:,}")
        print("\n  [Decision & Context on Duplicates]:")
        print("    Because features V1-V28 are PCA transformed and identifiers (e.g. cardholder,")
        print("    merchant, location, session) are absent, the exact real-world meaning of duplicate")
        print("    rows cannot be established from this anonymized dataset. They could represent distinct")
        print("    transactions with identical PCA projections, automated retry submissions, or logging")
        print("    duplicates. Retaining duplicates by default (`drop_duplicates=False`) prevents arbitrary")
        print(f"    data loss and preserves all 19 duplicate fraud records ({fraud_dups / 492 * 100:.2f}% of total fraud).")

    return {
        "total_missing": total_missing,
        "duplicate_count": dup_count,
        "duplicate_by_class": dup_by_class,
    }


def analyze_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and display genuine and fraudulent transaction statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame containing 'Class'.

    Returns
    -------
    dict
        Dictionary containing counts and percentages for each class.
    """
    print("\n" + "=" * 70)
    print("3. CLASS DISTRIBUTION (TARGET VARIABLE: 'Class')")
    print("=" * 70)

    counts = df["Class"].value_counts().to_dict()
    proportions = df["Class"].value_counts(normalize=True).to_dict()

    genuine_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    genuine_pct = proportions.get(0, 0.0) * 100
    fraud_pct = proportions.get(1, 0.0) * 100
    imbalance_ratio = genuine_count / fraud_count if fraud_count > 0 else 0

    print(f"Genuine Transactions (Class 0):    {genuine_count:>7,}  ({genuine_pct:>6.4f}%)")
    print(f"Fraudulent Transactions (Class 1): {fraud_count:>7,}  ({fraud_pct:>6.4f}%)")
    print(f"Imbalance Ratio:                   {imbalance_ratio:,.2f} : 1")
    print("\n  [Implication for ML]:")
    print("    Standard accuracy is misleading (a naive classifier predicting Class 0 achieves 99.83%).")
    print("    Stratified splitting is strictly required to preserve the 0.1727% minority representation.")
    print("    Cost-sensitive evaluation metrics (PR-AUC, Recall, Cost Matrices) are essential.")

    return {
        "genuine_count": genuine_count,
        "fraud_count": fraud_count,
        "genuine_pct": genuine_pct,
        "fraud_pct": fraud_pct,
        "imbalance_ratio": imbalance_ratio,
    }


def analyze_amount(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the Amount feature across genuine and fraudulent subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Descriptive statistics for transaction Amount.
    """
    print("\n" + "=" * 70)
    print("4. TRANSACTION AMOUNT ANALYSIS")
    print("=" * 70)

    overall = df["Amount"].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    genuine = df[df["Class"] == 0]["Amount"].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    fraud = df[df["Class"] == 1]["Amount"].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

    overall_iqr = float(df["Amount"].quantile(0.75) - df["Amount"].quantile(0.25))
    genuine_iqr = float(df[df["Class"] == 0]["Amount"].quantile(0.75) - df[df["Class"] == 0]["Amount"].quantile(0.25))
    fraud_iqr = float(df[df["Class"] == 1]["Amount"].quantile(0.75) - df[df["Class"] == 1]["Amount"].quantile(0.25))

    overall_skew = float(df["Amount"].skew())
    genuine_skew = float(df[df["Class"] == 0]["Amount"].skew())
    fraud_skew = float(df[df["Class"] == 1]["Amount"].skew())

    print(f"{'Statistic':<12} | {'Overall ($)':<15} | {'Genuine ($)':<15} | {'Fraudulent ($)':<15}")
    print("-" * 65)
    for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        val_all = overall.get(stat, 0.0)
        val_gen = genuine.get(stat, 0.0)
        val_frd = fraud.get(stat, 0.0)
        if stat == "count":
            print(f"{stat:<12} | {val_all:>15,.0f} | {val_gen:>15,.0f} | {val_frd:>15,.0f}")
        else:
            print(f"{stat:<12} | {val_all:>15,.2f} | {val_gen:>15,.2f} | {val_frd:>15,.2f}")

    print(f"{'IQR':<12} | {overall_iqr:>15.2f} | {genuine_iqr:>15.2f} | {fraud_iqr:>15.2f}")
    print(f"{'skewness':<12} | {overall_skew:>15.2f} | {genuine_skew:>15.2f} | {fraud_skew:>15.2f}")

    print("\n  [Note on Amount IQR Values]:")
    print(f"    - Full-dataset descriptive Amount IQR: ${overall_iqr:.2f} (Q1 = ${overall['25%']:.2f}, Q3 = ${overall['75%']:.2f}).")
    print("    - When fitting RobustScaler exclusively on the 70% training split, the training-only IQR")
    print("      is approximately $71.83 (Q1 = $5.67, Q3 = $77.50). This slight variation is expected")
    print("      from sample partitioning and illustrates proper isolation of training statistics.")

    print("\n  [Key Insights for Preprocessing]:")
    print("    - Heavy right skew (skewness = 16.98), with median $22.00 vs max $25,691.16.")
    print("    - Fraudulent transactions have a LOWER median ($9.25) than genuine ($22.00).")
    print("    - Standard scaling (z-score) would be heavily distorted by extreme outliers.")
    print("    - Decision: Use `RobustScaler` (median centering and IQR scaling) to scale Amount.")

    return {
        "overall": overall,
        "genuine": genuine,
        "fraud": fraud,
        "iqr": {"overall": overall_iqr, "genuine": genuine_iqr, "fraud": fraud_iqr},
        "skewness": {"overall": overall_skew, "genuine": genuine_skew, "fraud": fraud_skew},
    }


def analyze_time(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the Time feature (elapsed seconds from dataset start).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Descriptive statistics and temporal metrics.
    """
    print("\n" + "=" * 70)
    print("5. TIME FEATURE ANALYSIS")
    print("=" * 70)

    time_min = float(df["Time"].min())
    time_max = float(df["Time"].max())
    time_mean = float(df["Time"].mean())
    time_std = float(df["Time"].std())
    time_median = float(df["Time"].median())
    time_iqr = float(df["Time"].quantile(0.75) - df["Time"].quantile(0.25))

    duration_sec = time_max - time_min
    duration_hrs = duration_sec / 3600.0

    print(f"Feature Definition: Elapsed seconds from the first recorded transaction.")
    print(f"  Note: 'Time' represents relative elapsed time, NOT an actual clock timestamp or date.")
    print(f"Elapsed Time Span:  {time_min:.1f} to {time_max:.1f} seconds")
    print(f"Total Duration:     {duration_sec:,.0f} seconds = {duration_hrs:.2f} hours (~{duration_hrs / 24.0:.1f} days)")
    print(f"Descriptive Stats:  Mean = {time_mean:,.1f}s | Median = {time_median:,.1f}s | Std = {time_std:,.1f}s | IQR = {time_iqr:,.1f}s")

    print("\n  [Key Insights for Preprocessing]:")
    print("    - 'Time' is a continuous elapsed-time metric recording seconds from start.")
    print("    - Scaler must be fitted exclusively on training data to prevent leakage.")
    print("    - Decision: Retain 'Time' as a scaled elapsed-time feature using RobustScaler")
    print("      fitted strictly on the training partition.")

    return {
        "time_min": time_min,
        "time_max": time_max,
        "time_mean": time_mean,
        "time_std": time_std,
        "time_median": time_median,
        "time_iqr": time_iqr,
        "duration_seconds": duration_sec,
        "duration_hours": duration_hrs,
    }


def generate_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """Generate and save all required exploratory visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    output_dir : str
        Directory where visualization PNGs will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("6. GENERATING EDA VISUALIZATIONS")
    print("=" * 70)

    # Plot styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Visualization 1: Genuine vs Fraudulent Transaction Distribution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    counts = df["Class"].value_counts().sort_index()
    labels = ["Genuine (Class 0)", "Fraudulent (Class 1)"]
    colors = ["#2b5c8f", "#d9534f"]

    # Linear scale
    bars1 = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_title("Class Distribution (Linear Scale)\nSevere Imbalance Visible", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Transactions", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/len(df)*100:.3f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Log10 scale
    bars2 = ax2.bar(labels, counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_title("Class Distribution (Logarithmic Scale)\nMinority Class Exposed", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Transactions (Log10 Scale)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:,}\n({h/len(df)*100:.3f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Credit Card Transaction Class Imbalance (Total: 284,807 Transactions)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v1_path = os.path.join(output_dir, "01_class_distribution.png")
    plt.savefig(v1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v1_path}")

    # -------------------------------------------------------------------------
    # Visualization 2: Transaction Amount Distribution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Amounts up to 99th percentile ($500)
    p99 = df["Amount"].quantile(0.99)
    ax1.hist(df[df["Amount"] <= 500]["Amount"], bins=50, color="#2b5c8f", edgecolor="black", alpha=0.75)
    ax1.set_title("Transaction Amount Distribution\nTruncated at $500 (~99th percentile)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Transaction Amount ($)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.axvline(df["Amount"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: ${df['Amount'].median():.2f}")
    ax1.axvline(df["Amount"].mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: ${df['Amount'].mean():.2f}")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Log10-transformed amounts: log10(Amount + 1)
    log_amounts = np.log10(df["Amount"] + 1)
    ax2.hist(log_amounts, bins=50, color="#17a2b8", edgecolor="black", alpha=0.75)
    ax2.set_title("Log-Transformed Amount Distribution: $\\log_{10}(\\mathrm{Amount} + 1)$\nFull Range ($0 to $25,691)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$\\log_{10}(\\mathrm{Amount} + 1)$", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.axvline(np.log10(df["Amount"].median() + 1), color="red", linestyle="--", linewidth=1.5, label=f"Median: ${df['Amount'].median():.2f}")
    ax2.axvline(np.log10(df["Amount"].mean() + 1), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: ${df['Amount'].mean():.2f}")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Distribution of Credit Card Transaction Amounts", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v2_path = os.path.join(output_dir, "02_amount_distribution.png")
    plt.savefig(v2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v2_path}")

    # -------------------------------------------------------------------------
    # Visualization 3: Amount Comparison: Genuine vs Fraudulent
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    gen_amt = df[df["Class"] == 0]["Amount"]
    frd_amt = df[df["Class"] == 1]["Amount"]

    # Boxplots on log scale
    box_data = [np.log10(gen_amt + 1), np.log10(frd_amt + 1)]
    bp = ax1.boxplot(box_data, tick_labels=["Genuine (Class 0)", "Fraudulent (Class 1)"],
                     patch_artist=True, widths=0.45,
                     boxprops=dict(facecolor="#e9ecef", edgecolor="black"),
                     medianprops=dict(color="crimson", linewidth=2.5),
                     flierprops=dict(marker="o", markersize=3, alpha=0.3, color="gray"))
    bp["boxes"][0].set_facecolor("#cfe2ff")
    bp["boxes"][1].set_facecolor("#f8d7da")
    ax1.set_title("Log-Transformed Amount by Class (Boxplot)\n$\\log_{10}(\\mathrm{Amount} + 1)$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("$\\log_{10}(\\mathrm{Amount} + 1)$", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotated medians & IQRs (full dataset)
    ax1.annotate(f"Median: ${gen_amt.median():.2f}\nIQR: ${gen_amt.quantile(0.75)-gen_amt.quantile(0.25):.2f}",
                 xy=(1, np.log10(gen_amt.median() + 1)), xytext=(1.25, 1.0),
                 arrowprops=dict(arrowstyle="->", color="blue"), fontsize=9)
    ax1.annotate(f"Median: ${frd_amt.median():.2f}\nIQR: ${frd_amt.quantile(0.75)-frd_amt.quantile(0.25):.2f}",
                 xy=(2, np.log10(frd_amt.median() + 1)), xytext=(2.25, 0.5),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=9)

    # Density / Histogram comparison on log scale
    bins = np.linspace(0, np.log10(df["Amount"].max() + 1), 40)
    ax2.hist(np.log10(gen_amt + 1), bins=bins, density=True, alpha=0.5, color="#2b5c8f", label="Genuine (density)")
    ax2.hist(np.log10(frd_amt + 1), bins=bins, density=True, alpha=0.5, color="#d9534f", label="Fraudulent (density)")
    ax2.set_title("Normalized Density Comparison\n$\\log_{10}(\\mathrm{Amount} + 1)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$\\log_{10}(\\mathrm{Amount} + 1)$", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Transaction Amount Comparison: Genuine vs. Fraudulent Transactions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v3_path = os.path.join(output_dir, "03_amount_by_class.png")
    plt.savefig(v3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v3_path}")

    # -------------------------------------------------------------------------
    # Visualization 4: Transaction Distribution Over Elapsed Time (~48 Hours)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Convert elapsed seconds to elapsed hours from start (0 to ~48 hours)
    elapsed_hours = df["Time"] / 3600.0
    bins_48 = np.arange(0, 49, 1)

    gen_counts_48, _ = np.histogram(elapsed_hours[df["Class"] == 0], bins=bins_48)
    frd_counts_48, _ = np.histogram(elapsed_hours[df["Class"] == 1], bins=bins_48)
    bin_centers = (bins_48[:-1] + bins_48[1:]) / 2

    color_gen = "#2b5c8f"
    color_frd = "#d9534f"

    # Subplot 1: Genuine Volume vs Fraud Count across elapsed time
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(bin_centers, gen_counts_48, color=color_gen, linewidth=2, label="Genuine Transactions (Left Axis)")
    l2 = ax1_twin.plot(bin_centers, frd_counts_48, color=color_frd, linewidth=2, linestyle="--", marker="o", markersize=4, label="Fraudulent Transactions (Right Axis)")

    ax1.set_title("Transaction Frequency Over Elapsed Time (1-Hour Elapsed Bins)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Genuine Transaction Count", color=color_gen, fontsize=11)
    ax1_twin.set_ylabel("Fraudulent Transaction Count", color=color_frd, fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    # Subplot 2: Cumulative Distribution of Transactions over Elapsed Time
    # Shows the progression of transactions across the 48-hour recording window
    sorted_gen_time = np.sort(elapsed_hours[df["Class"] == 0])
    sorted_frd_time = np.sort(elapsed_hours[df["Class"] == 1])
    cdf_gen = np.arange(1, len(sorted_gen_time) + 1) / len(sorted_gen_time)
    cdf_frd = np.arange(1, len(sorted_frd_time) + 1) / len(sorted_frd_time)

    ax2.plot(sorted_gen_time, cdf_gen, color=color_gen, linewidth=2, label="Genuine Cumulative Distribution")
    ax2.plot(sorted_frd_time, cdf_frd, color=color_frd, linewidth=2, linestyle="--", label="Fraudulent Cumulative Distribution")
    ax2.set_title("Cumulative Progression of Transactions Over Elapsed Time", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Elapsed Time (Hours from dataset start: 0 to ~48 hours)", fontsize=11)
    ax2.set_ylabel("Cumulative Fraction", fontsize=11)
    ax2.set_xlim(0, 48)
    ax2.set_xticks(range(0, 49, 4))
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Analysis of Elapsed Time: Progression of Genuine and Fraudulent Transactions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    v4_path = os.path.join(output_dir, "04_transactions_over_time.png")
    plt.savefig(v4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v4_path}")

    # -------------------------------------------------------------------------
    # Visualization 5: Feature Correlations with Class
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    correlations = df.corr()["Class"].drop("Class").sort_values()

    # Color code bars: Red for positive, Blue for negative
    bar_colors = ["#d9534f" if c > 0 else "#2b5c8f" for c in correlations.values]
    ax.barh(correlations.index, correlations.values, color=bar_colors, edgecolor="black", alpha=0.8, height=0.6)
    ax.set_title("Pearson Correlation of Features with Fraud Target ('Class')", fontsize=13, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient with Class (Target)", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    v5_path = os.path.join(output_dir, "05_feature_correlations.png")
    plt.savefig(v5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {v5_path}")

    print("\n[+] All visualizations successfully generated and saved in:")
    print(f"    {os.path.abspath(output_dir)}")


def run_exploration(data_path: str = "data/creditcard.csv", output_dir: str = "outputs/visualizations") -> Dict[str, Any]:
    """Execute end-to-end dataset exploration workflow.

    Parameters
    ----------
    data_path : str
        Relative or absolute path to data CSV.
    output_dir : str
        Directory to save visualization outputs.

    Returns
    -------
    dict
        Consolidated summary of dataset statistics.
    """
    print("=" * 70)
    print("PHASE 1: CREDIT CARD FRAUD DATASET EXPLORATION")
    print("=" * 70)

    df = load_dataset(data_path)
    structure_info = inspect_structure(df)
    integrity_info = check_missing_and_duplicates(df)
    class_info = analyze_class_distribution(df)
    amount_info = analyze_amount(df)
    time_info = analyze_time(df)
    generate_visualizations(df, output_dir)

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE - READY FOR PREPROCESSING PHASE")
    print("=" * 70)

    return {
        "structure": structure_info,
        "integrity": integrity_info,
        "class_distribution": class_info,
        "amount_stats": amount_info,
        "time_stats": time_info,
    }


if __name__ == "__main__":
    # Support running from workspace root
    data_filepath = os.path.join("data", "creditcard.csv")
    output_directory = os.path.join("outputs", "visualizations")
    run_exploration(data_path=data_filepath, output_dir=output_directory)
