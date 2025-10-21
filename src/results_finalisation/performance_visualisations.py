# src/results_finalisation/performance_visualisations.py
"""
performance_visualisations.py

Title: Final performance visualisations and comparison plots

Summary:
- Generate final performance visualisations and a unified comparison table for:
    - LightGBM retrained (Phase 5)
    - TCN refined (Phase 4.5)
- This script intentionally avoids guessing column names or performing log<->raw transforms.
- It uses the exact columns and the metric keys specified (raw calibrated metrics, tuned metrics, raw predictions).
- KDE is implemented with a small, dependency-free function (no scipy required).

Logic:
1. Load prediction CSVs and metrics JSONs for LightGBM and TCN_refined.
2. Use only the explicit columns that we agreed on:
    - LightGBM predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg
    - TCN predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg_raw
    - LightGBM metrics JSON: keys for "max_risk", "median_risk", "pct_time_high"
    - TCN metrics JSON: use "max_risk", "median_risk_tuned" and "pct_time_high_raw_cal"
3. Compute calibration diagnostics (Brier score, ECE) from CSV probabilities.
4. Compose comparison table (CSV) using JSON values and computed calibration diagnostics.
5. Produce final plots and save them to disk:
    - ROC, Precision-Recall
    - Calibration curve, Calibration histogram
    - Regression: True vs Predicted scatter; Residual histogram + KDE; Error vs Truth
    - Metric comparison grouped bar charts (classification: AUC/F1/Brier/ECE; regression: RMSE/R2)
6. Save outputs under: src/results_finalisation/comparison_plots/
"""

# -----------------------
# Imports
# -----------------------
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import scikit-learn metrics used for evaluation and plotting
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
    mean_squared_error, r2_score,
)
from sklearn.calibration import calibration_curve

# -----------------------
# Configuration / Paths
# -----------------------
ROOT = Path(__file__).resolve().parent.parent

# LightGBM input files
LIGHTGBM_PRED_CSV = ROOT / "prediction_evaluations" / "lightgbm_results" / "lightgbm_predictions.csv"
LIGHTGBM_MET_JSON = ROOT / "prediction_evaluations" / "lightgbm_results" / "lightgbm_metrics.json"

# TCN input files
TCN_PRED_CSV = ROOT / "prediction_evaluations" / "tcn_results_refined" / "tcn_predictions_refined.csv"
TCN_MET_JSON = ROOT / "prediction_evaluations" / "tcn_results_refined" / "tcn_metrics_refined.json"

# Output folder for all plots and comparison tables
OUT_DIR = ROOT / "results_finalisation" / "comparison_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output table CSV summarising metrics for all tasks/models
COMPARISON_CSV = OUT_DIR / "comparison_table.csv"

# -----------------------
# Fixed column mapping (explicit, no guessing)
# -----------------------
# LightGBM columns (expected)
LGB_COLS = {
    "y_true_max": "y_true_max",
    "prob_max": "prob_max",
    "y_true_median": "y_true_median",
    "prob_median": "prob_median",
    "y_true_reg": "y_true_reg",
    "y_pred_reg": "y_pred_reg"
}

# TCN columns we will use (explicit)
TCN_COLS = {
    "y_true_max": "y_true_max",
    "prob_max": "prob_max",
    "y_true_median": "y_true_median",
    "prob_median": "prob_median",
    # For regression we will use the raw predictions column:
    "y_true_reg": "y_true_reg",
    "y_pred_reg_raw": "y_pred_reg_raw"
}

# -----------------------
# Utility functions
# -----------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    - equal-width bins between 0..1
    - returns scalar ECE
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1  # 0..n_bins-1
    total = len(y_prob)
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if mask.sum() == 0:
            continue
        bin_prob_mean = y_prob[mask].mean()
        bin_true_mean = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(bin_prob_mean - bin_true_mean)
    return float(ece)

def kde_1d(values: np.ndarray, grid: np.ndarray, bandwidth: float = None) -> np.ndarray:
    """
    Simple Gaussian KDE implementation (dependency-free).
    - values: data points (1d)
    - grid: points where to evaluate the density
    - bandwidth: if None, use Silverman's rule of thumb
    Returns density evaluated at grid.
    """
    vals = np.asarray(values).ravel()
    n = len(vals)
    if n == 0:
        return np.zeros_like(grid)
    if bandwidth is None:
        std = np.std(vals, ddof=1) if n > 1 else 1.0
        bandwidth = 1.06 * std * n ** (-1 / 5) if std > 0 else 0.1
    # Evaluate kernel density
    diffs = grid[:, None] - vals[None, :]
    kernel = np.exp(-0.5 * (diffs / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    density = kernel.sum(axis=1) / n
    return density

# -----------------------
# Plot helper wrappers (for consistent styling)
# -----------------------
def plot_roc(y_true, y_prob, ax, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

def plot_pr(y_true, y_prob, ax, label):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

def plot_calibration_curve(y_true, y_prob, ax, label, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")

# -----------------------
# Load predictions and metrics
# -----------------------
"""
- Load predictions and metrics from CSVs/JSONs
- LightGBM and TCN_refined prediction CSVs contain:
    - For classification: columns "y_true_max", "prob_max", "y_true_median", "prob_median" where "y_true_*" are ground truth binary labels and "prob_*" are model probabilities.
    - For regression: columns like "y_true_reg", "y_pred_reg", etc. for pct_time_high task.
- These columns are used for all metric calculations and plots below.
- Alignment: Both CSVs must have the same patient order for valid comparisons.
"""
print("Loading prediction CSVs and metrics JSONs...")

# Load LightGBM predictions and metrics
df_lgb = pd.read_csv(LIGHTGBM_PRED_CSV)
with open(LIGHTGBM_MET_JSON, "r") as fh:
    lgb_metrics = json.load(fh)

# Load TCN predictions and metrics
df_tcn = pd.read_csv(TCN_PRED_CSV)
with open(TCN_MET_JSON, "r") as fh:
    tcn_metrics = json.load(fh)

# Basic alignment check (we only check the primary truth column; order already confirmed)
if len(df_lgb) != len(df_tcn):
    raise RuntimeError(f"Prediction CSV length mismatch: LightGBM={len(df_lgb)} vs TCN={len(df_tcn)}")

# Quick equality check for y_true_max (sanity); if mismatch, we still proceed but warn.
if not np.allclose(df_lgb[LGB_COLS["y_true_max"]].astype(float).values,
                   df_tcn[TCN_COLS["y_true_max"]].astype(float).values):
    print("WARNING: y_true_max differs between LightGBM and TCN files. Verify patient ordering!")

# -----------------------
# Compute additional metrics (Brier, ECE) and create comparison table rows
# -----------------------
"""
- For each model, aggregate all key metrics for each task.
- For classification: uses columns
    - y_true_max/prob_max (max risk), y_true_median/prob_median (median risk)
    - Ground truth: "y_true_*", predictions: "prob_*"
- For regression: uses columns
    - y_true_reg/y_pred_reg, y_true_reg_raw/y_pred_reg_raw, etc.
    - Ground truth: "y_true_reg*", predictions: "y_pred_reg*"
- All metrics and plots are calculated using actual correct values and labels from the prediction CSVs.
"""
rows = []

def collect_class_metrics(name: str, df: pd.DataFrame, metrics_json: Dict[str, Any],
                          true_col: str, prob_col: str, json_key: str, tuned_median: bool=False):
    """
    Collect classification metrics row for model 'name' for a specific target.
    - true_col/prob_col: columns in df to use
    - json_key: key in metrics_json to source AUC/F1/etc (if present)
    - tuned_median: if True and json contains 'median_risk_tuned', prefer that
    """
    y_true = df[true_col].astype(float).values
    y_prob = df[prob_col].astype(float).values

    # Use JSON metrics when available (explicit keys)
    meta = {}
    if tuned_median and "median_risk_tuned" in metrics_json:
        meta = metrics_json.get("median_risk_tuned", {})
    else:
        meta = metrics_json.get(json_key, {})

    # Compute Brier and ECE from CSV probs
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=10))

    row = {
        "model": name,
        "target": json_key,
        "roc_auc": meta.get("roc_auc", np.nan),
        "f1": meta.get("f1", np.nan),
        "accuracy": meta.get("accuracy", np.nan),
        "precision": meta.get("precision", np.nan),
        "recall": meta.get("recall", np.nan),
        "brier": brier,
        "ece": ece,
    }
    rows.append(row)

def collect_regression_metrics(name: str, df: pd.DataFrame, metrics_json: Dict[str, Any],
                               y_true_col: str, y_pred_col: str, json_key_preferred: str = None):
    """
    Collect regression metrics row.
    - y_true_col, y_pred_col: columns in df with raw values to compare (no transforms).
    - json_key_preferred: e.g. "pct_time_high_raw_cal" for TCN calibrated raw metrics.
    """
    y_true = df[y_true_col].astype(float).values
    y_pred = df[y_pred_col].astype(float).values

    # Prefer the JSON key specified (TCN calibrated raw), else fallback to json key 'pct_time_high'
    meta_reg = {}
    if json_key_preferred and json_key_preferred in metrics_json:
        meta_reg = metrics_json[json_key_preferred]
    else:
        # Try 'pct_time_high' or fallback empty
        meta_reg = metrics_json.get("pct_time_high", {})

    rmse_json = meta_reg.get("rmse", np.nan)
    r2_json = meta_reg.get("r2", np.nan)

    # Recompute from CSV arrays to ensure consistency / act as ground-truth
    rmse_calc = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_calc = float(r2_score(y_true, y_pred))

    row = {
        "model": name,
        "target": "pct_time_high",
        "rmse": float(rmse_json if not np.isnan(rmse_json) else rmse_calc),
        "r2": float(r2_json if not np.isnan(r2_json) else r2_calc),
        "rmse_calc": rmse_calc,
        "r2_calc": r2_calc
    }
    rows.append(row)

# Collect LightGBM classification rows
collect_class_metrics(
    name="LightGBM",
    df=df_lgb,
    metrics_json=lgb_metrics,
    true_col=LGB_COLS["y_true_max"], prob_col=LGB_COLS["prob_max"],
    json_key="max_risk", tuned_median=False
)
collect_class_metrics(
    name="LightGBM",
    df=df_lgb,
    metrics_json=lgb_metrics,
    true_col=LGB_COLS["y_true_median"], prob_col=LGB_COLS["prob_median"],
    json_key="median_risk", tuned_median=False
)
# Collect LightGBM regression row (raw)
collect_regression_metrics(
    name="LightGBM",
    df=df_lgb,
    metrics_json=lgb_metrics,
    y_true_col=LGB_COLS["y_true_reg"], y_pred_col=LGB_COLS["y_pred_reg"],
    json_key_preferred="pct_time_high"  # LightGBM JSON key
)

# Collect TCN classification rows (use median tuned metrics for median)
collect_class_metrics(
    name="TCN_refined",
    df=df_tcn,
    metrics_json=tcn_metrics,
    true_col=TCN_COLS["y_true_max"], prob_col=TCN_COLS["prob_max"],
    json_key="max_risk", tuned_median=False
)
collect_class_metrics(
    name="TCN_refined",
    df=df_tcn,
    metrics_json=tcn_metrics,
    true_col=TCN_COLS["y_true_median"], prob_col=TCN_COLS["prob_median"],
    json_key="median_risk_tuned", tuned_median=True
)
# Collect TCN regression row (use calibrated raw metrics key)
collect_regression_metrics(
    name="TCN_refined",
    df=df_tcn,
    metrics_json=tcn_metrics,
    y_true_col=TCN_COLS["y_true_reg"], y_pred_col=TCN_COLS["y_pred_reg_raw"],
    json_key_preferred="pct_time_high_raw_cal"
)

# Save comparison table CSV
df_comp = pd.DataFrame(rows)
df_comp.to_csv(COMPARISON_CSV, index=False)
print(f"Saved comparison table → {COMPARISON_CSV}")

# -----------------------
# Plotting: classification plots (both models on same axes)
# -----------------------
# For each classification target create ROC, PR, Calibration curve, Calibration histogram.
classification_targets = [
    ("max_risk", LGB_COLS["y_true_max"], LGB_COLS["prob_max"], TCN_COLS["prob_max"]),
    ("median_risk", LGB_COLS["y_true_median"], LGB_COLS["prob_median"], TCN_COLS["prob_median"]),
]

for tgt_name, y_col_lgb, prob_col_lgb, prob_col_tcn in classification_targets:
    # Ensure required columns exist
    if y_col_lgb not in df_lgb.columns or prob_col_lgb not in df_lgb.columns:
        print(f"Skipping classification plots for {tgt_name}: required LightGBM columns missing.")
        continue
    if prob_col_tcn not in df_tcn.columns:
        print(f"Skipping classification plots for {tgt_name}: required TCN columns missing.")
        continue

    # Ground-truth (take from LightGBM CSV; order assumed identical)
    y_true = df_lgb[y_col_lgb].astype(float).values

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_true, df_lgb[prob_col_lgb].astype(float).values, ax=ax, label="LightGBM")
    plot_roc(y_true, df_tcn[prob_col_tcn].astype(float).values, ax=ax, label="TCN_refined")
    ax.set_title(f"ROC Curve — {tgt_name}")
    ax.legend()
    fpath = OUT_DIR / f"roc_{tgt_name}.png"
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {fpath}")

    # Precision-Recall
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_true, df_lgb[prob_col_lgb].astype(float).values, ax=ax, label="LightGBM")
    plot_pr(y_true, df_tcn[prob_col_tcn].astype(float).values, ax=ax, label="TCN_refined")
    ax.set_title(f"Precision-Recall — {tgt_name}")
    ax.legend()
    fpath = OUT_DIR / f"pr_{tgt_name}.png"
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {fpath}")

    # Calibration curve
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_calibration_curve(y_true, df_lgb[prob_col_lgb].astype(float).values, ax=ax, label="LightGBM")
    plot_calibration_curve(y_true, df_tcn[prob_col_tcn].astype(float).values, ax=ax, label="TCN_refined")
    ax.set_title(f"Calibration Curve — {tgt_name}")
    ax.legend()
    fpath = OUT_DIR / f"calibration_{tgt_name}.png"
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {fpath}")

    # Calibration histograms (side-by-side)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(df_lgb[prob_col_lgb].astype(float).values, bins=10, range=(0.0, 1.0))
    axs[0].set_title("LightGBM probability histogram")
    axs[0].set_xlabel("Predicted probability")
    axs[0].set_ylabel("Count (test patients)")
    axs[1].hist(df_tcn[prob_col_tcn].astype(float).values, bins=10, range=(0.0, 1.0))
    axs[1].set_title("TCN_refined probability histogram")
    axs[1].set_xlabel("Predicted probability")
    fpath = OUT_DIR / f"calibration_hist_{tgt_name}.png"
    fig.suptitle(f"Calibration histograms — {tgt_name}")
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {fpath}")

# -----------------------
# Plotting: regression plots (pct_time_high)
# -----------------------
# Use the explicit raw columns:
y_true_lgb = df_lgb[LGB_COLS["y_true_reg"]].astype(float).values
y_pred_lgb = df_lgb[LGB_COLS["y_pred_reg"]].astype(float).values
y_true_tcn = df_tcn[TCN_COLS["y_true_reg"]].astype(float).values
y_pred_tcn = df_tcn[TCN_COLS["y_pred_reg_raw"]].astype(float).values

# Sanity check ordering
if not np.allclose(y_true_lgb, y_true_tcn, atol=1e-8):
    print("WARNING: regression ground-truth mismatch between LightGBM and TCN. Visualizations will still be produced but verify ordering!")

# Scatter overlay: true vs predicted
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, y_pred_lgb, alpha=0.85, label="LightGBM")
ax.scatter(y_true_tcn, y_pred_tcn, alpha=0.85, label="TCN_refined", marker="x")
mn = min(y_true_lgb.min(), y_pred_lgb.min(), y_pred_tcn.min())
mx = max(y_true_lgb.max(), y_pred_lgb.max(), y_pred_tcn.max())
ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Predicted pct_time_high")
ax.set_title("Regression: True vs Predicted (pct_time_high)")
ax.legend()
fpath = OUT_DIR / "regression_scatter_pct_time_high.png"
fig.savefig(fpath, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved", fpath)

# Residuals hist + KDE (per model). KDE uses kde_1d implemented above.
res_lgb = y_pred_lgb - y_true_lgb
res_tcn = y_pred_tcn - y_true_tcn

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# LightGBM residuals
axs[0].hist(res_lgb, bins=12, density=True, alpha=0.6)
# KDE overlay
grid = np.linspace(res_lgb.min() - 1e-6, res_lgb.max() + 1e-6, 200)
kde_vals = kde_1d(res_lgb, grid)
axs[0].plot(grid, kde_vals, lw=1.4)
axs[0].set_title("LightGBM residuals (pred - true)")
axs[0].set_xlabel("Residual")
axs[0].set_ylabel("Density")

# TCN residuals
axs[1].hist(res_tcn, bins=12, density=True, alpha=0.6)
grid2 = np.linspace(res_tcn.min() - 1e-6, res_tcn.max() + 1e-6, 200)
kde_vals2 = kde_1d(res_tcn, grid2)
axs[1].plot(grid2, kde_vals2, lw=1.4)
axs[1].set_title("TCN_refined residuals (pred - true)")
axs[1].set_xlabel("Residual")
fpath = OUT_DIR / "regression_residuals_pct_time_high.png"
fig.suptitle("Residuals + KDE — pct_time_high")
fig.savefig(fpath, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved", fpath)

# Error vs truth (heteroscedasticity / bias check)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, res_lgb, alpha=0.8, label="LightGBM")
ax.scatter(y_true_tcn, res_tcn, alpha=0.8, label="TCN_refined", marker="x")
ax.axhline(0, linestyle="--", color="gray")
ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Residual (pred - true)")
ax.set_title("Error vs Truth (pct_time_high)")
ax.legend()
fpath = OUT_DIR / "regression_error_vs_truth.png"
fig.savefig(fpath, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved", fpath)


# -----------------------
# Metric comparison bar charts (grouped bars per target)
# -----------------------
# Build plotting dataframe from df_comp (the CSV table we saved)
plot_df = df_comp.copy()

# For classification: show roc_auc, f1, brier, ece
for tgt in ["max_risk", "median_risk"]:
    sub = plot_df[plot_df["target"] == tgt]
    if sub.empty:
        continue
    models = sub["model"].values
    roc = sub["roc_auc"].values
    f1 = sub["f1"].values
    brier = sub["brier"].values
    ece = sub["ece"].values

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 1.5 * width, roc, width, label="ROC AUC")
    ax.bar(x - 0.5 * width, f1, width, label="F1")
    ax.bar(x + 0.5 * width, brier, width, label="Brier")
    ax.bar(x + 1.5 * width, ece, width, label="ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Metric value")
    ax.set_title(f"Classification metrics comparison — {tgt}")
    ax.legend()
    fpath = OUT_DIR / f"metrics_comparison_{tgt}.png"
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved", fpath)

# For regression: show rmse and r2 (use the json-preferred numbers in df_comp)
sub = plot_df[plot_df["target"] == "pct_time_high"]
if not sub.empty:
    models = sub["model"].values
    rmse = sub["rmse"].values
    r2 = sub["r2"].values

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, rmse, width, label="RMSE")
    ax.bar(x + width/2, r2, width, label="R²")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Metric value")
    ax.set_title("Regression metrics comparison — pct_time_high")
    ax.legend()
    fpath = OUT_DIR / "metrics_comparison_pct_time_high.png"
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved", fpath)

print("All plots and CSV saved to:", OUT_DIR)
print("Done.")