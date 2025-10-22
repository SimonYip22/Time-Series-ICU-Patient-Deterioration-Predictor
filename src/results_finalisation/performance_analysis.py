# src/results_finalisation/performance_analysis.py
"""
performance_analysis.py

Title: Final performance visualisations and cross-model comparison for LightGBM and TCN.

Purpose:
- This script consolidates all performance results from Phase 4.5 (TCN refined) and Phase 5 (LightGBM retrained) into a single Phase 6 comparative analysis.
- It aligns both models on the same patient set, computes additional calibration diagnostics (Brier and ECE), and generates standardised visualisations for classification and regression tasks.

Summary of Workflow:
1. Load prediction CSVs and metrics JSONs for both models.
2. Use explicit column mappings and metric keys (no inference, no conversions).
    - LightGBM predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg
    - TCN predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg_raw
    - LightGBM metrics JSON: keys for "max_risk", "median_risk", "pct_time_high"
    - TCN metrics JSON: keys for "max_risk", "median_risk_tuned" and "pct_time_high_raw_cal"
3. Compute calibration metrics (Brier score, ECE) directly from probability predictions.
4. Merge metrics (from JSON + computed) into one comparison table.
5. Generate all plots (classification, regression, calibration, residuals, and metric bars).
6. Save everything into `src/results_finalisation/comparison_plots/`.

Outputs:
- comparison_table.csv (aggregated metrics)
- 14 plots total:
    - Classification: ROC, Precision-Recall, Calibration curves, Calibration histograms
    - Regression: Regression scatter, residual distributions, error-vs-truth
    - Metric comparison grouped bars for each target (3 separate charts: max_risk, median_risk, pct_time_high)
"""

# -----------------------
# Imports
# -----------------------
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Core scikit-learn metrics used for classification and regression evaluation
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
    mean_squared_error, r2_score,
)
from sklearn.calibration import calibration_curve

# -----------------------
# Paths and configuration
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

# Output comparison table CSV summarising metrics for all tasks and models
COMPARISON_CSV = OUT_DIR / "comparison_table.csv"

# -----------------------
# Explicit column mapping
# -----------------------
"""
- Explicitly define which columns to use from each model’s prediction CSV.
- This ensures full control and consistency between LightGBM and TCN.
- Avoids guessing or log<->raw transforms.
"""
# LightGBM expected columns in its predictions CSV
LGB_COLS = {
    "y_true_max": "y_true_max",
    "prob_max": "prob_max",
    "y_true_median": "y_true_median",
    "prob_median": "prob_median",
    "y_true_reg": "y_true_reg",
    "y_pred_reg": "y_pred_reg"
}

# TCN expected columns in its predictions CSV
TCN_COLS = {
    "y_true_max": "y_true_max",
    "prob_max": "prob_max",
    "y_true_median": "y_true_median",
    "prob_median": "prob_median",
    # Important: use calibrated *raw* predictions for regression comparison with LightGBM
    "y_true_reg": "y_true_reg",
    "y_pred_reg_raw": "y_pred_reg_raw"
}

# -----------------------
# Utility functions
# -----------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Purpose:
    - Quantifies the average discrepancy between predicted probability and empirical frequency.
    - It measures probabilistic calibration (how well predicted probabilities match observed outcomes).
    - Must be built manually as ECE not built into scikit-learn.
    - Implemented with equal-width bins over [0, 1].

    Concept:
    - Splits predictions into equal-width probability bins.
    - Within each bin, compares the average predicted probability (model confidence) vs the observed fraction of positives.
    - The weighted average of these gaps = ECE.

    Inputs:
    - y_true: array-like of binary labels (0/1)
    - y_prob: array-like of predicted probabilities (0..1)
    - n_bins: number of bins (default 10)

    Formula:
        ECE = Σ_i (|mean(p_i) - mean(y_i)| * (#samples_i / total))

    Output:
    - float scalar ECE (lower is better)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1  # map probs into 0..n_bins-1
    ece = 0.0
    total = len(y_prob)
    for i in range(n_bins):
        mask = binids == i
        if mask.sum() == 0:
            continue
        bin_prob_mean = y_prob[mask].mean()
        bin_true_mean = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(bin_prob_mean - bin_true_mean)
    return float(ece)

def kde_1d(values, grid, bandwidth=None):
    """
    Compute a simple 1D Gaussian KDE (Kernel Density Estimate) without external dependencies.
    ----------------------------------------------------
    Purpose:
    - Used for smooth residual distribution visualisation.
    - matplotlib’s histogram only shows bars, so we overlay our own KDE curve to visualise continuous residual distribution.

    Parameters:
    - values: 1-D array-like of data points (e.g. residuals)
    - grid: 1-D array of points to evaluate the KDE at
    - bandwidth: optional kernel bandwidth (float). If None, Silverman's rule of thumb is used.

    Bandwidth note:
    - Bandwidth controls smoothing. Small -> wiggly KDE; Large -> over-smooth.
    - Silverman's rule (1.06 * std * n^(-1/5)) is a standard, safe default.
    - You can pass a custom bandwidth if you want less/more smoothing.

    Returns:
    - density array same shape as grid
    """
    vals = np.asarray(values).ravel()
    if vals.size == 0:
        return np.zeros_like(grid)
    n = len(vals)
    if bandwidth is None:
        std = np.std(vals, ddof=1) if n > 1 else 1.0
        # Fall back to small positive value if std == 0
        bandwidth = 1.06 * std * n ** (-1 / 5) if std > 0 else 0.1
    # compute gaussian kernels and average
    diffs = grid[:, None] - vals[None, :]
    kernel = np.exp(-0.5 * (diffs / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    density = kernel.sum(axis=1) / n
    return density

# -----------------------
# Plot helper functions
# -----------------------
"""
- Reusable wrappers for ROC, PR, and calibration plots.
- These prevent code duplication and ensure consistent axes/formatting.
"""

# ROC
def plot_roc(y_true, y_prob, ax, label):
    """Plot ROC curve for a single model on provided axis."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

# Precision-Recall
def plot_pr(y_true, y_prob, ax, label):
    """Plot precision-recall curve for a single model on provided axis."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

# Calibration 
def plot_calibration_curve(y_true, y_prob, ax, label, n_bins=10):
    """Plot reliability (calibration) curve for a single model on provided axis."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")

# -----------------------
# Load predictions and metrics
# -----------------------
"""
- Read predictions CSVs and metrics JSONs for both models.
- The CSVs supply ground-truth labels and per-patient predictions (probabilities + regression predictions).
- The JSONs supply precomputed metrics (roc_auc, f1, rmse, r2, etc.) — these are used directly where available.
"""
print("Loading LightGBM and TCN prediction CSVs and metrics JSONs...")

# Load LightGBM predictions CSV (contains per-patient y_true_*/prob_* and y_pred_reg)
df_lgb = pd.read_csv(LIGHTGBM_PRED_CSV)
# Load LightGBM metrics JSON (precomputed values for AUC/F1/RMSE etc.)
with open(LIGHTGBM_MET_JSON, "r") as f:
    lgb_metrics = json.load(f)

# Load TCN predictions (contains per-patient y_true_*/prob_* and y_pred_reg_raw)
df_tcn = pd.read_csv(TCN_PRED_CSV)
# Load TCN metrics JSON (contains keys including median_risk_tuned and pct_time_high_raw_cal)
with open(TCN_MET_JSON, "r") as f:
    tcn_metrics = json.load(f)

# Print counts for user confirmation
print(f"Loaded {len(df_lgb)} LightGBM + {len(df_tcn)} TCN rows")

# Quick sanity checks to avoid silent misalignment mistakes:
# 1) Ensure same number of rows
if len(df_lgb) != len(df_tcn):
    raise RuntimeError(f"Prediction CSV length mismatch: LightGBM={len(df_lgb)} vs TCN={len(df_tcn)}")

# 2) Quick check equality of primary label column (y_true_max) to detect ordering issues early
if not np.allclose(df_lgb[LGB_COLS["y_true_max"]].astype(float).values,
                   df_tcn[TCN_COLS["y_true_max"]].astype(float).values):
    print("WARNING: y_true_max differs between LightGBM and TCN files. Verify patient ordering!")

# -----------------------
# Metric collection functions
# -----------------------
"""
- The following functions extract metrics from JSON (precomputed) and combine them with calibration metrics computed directly from CSVs.
    - Classification: use JSON for AUC/F1/Accuracy/Precision/Recall; compute Brier & ECE from CSV using raw probabilities.
    - Regression: use JSON metrics for TCN (pct_time_high_raw_cal) and LightGBM (pct_time_high)
- Functions:
    - collect_class_metrics: merges JSON metrics (AUC/F1/...) with computed Brier & ECE (from CSV probabilities).
    - collect_regression_metrics: returns regression metrics strictly from JSON (we do NOT recompute RMSE/R² here).
"""
def collect_class_metrics(model_name, df, metrics_json, true_col, prob_col, json_key):
    """
    Build a single dictionary row aggregating classification metrics for one model & target.

    Inputs:
      - model_name: string label for the model (e.g. "LightGBM")
      - df: DataFrame for that model's predictions
      - metrics_json: JSON-loaded dict of precomputed metrics for that model
      - true_col: column name (in df) with ground-truth binary labels
      - prob_col: column name (in df) with predicted probabilities
      - json_key: key in metrics_json to obtain roc_auc, f1, accuracy, etc.

    Returns:
      - dict with fields: model, target, roc_auc, f1, accuracy, precision, recall, brier, ece
    """

    # Extract arrays; convert to float to avoid dtype surprises
    y_true = df[true_col].astype(float).values
    y_prob = df[prob_col].astype(float).values

    # Use precomputed metrics when present (json_key may be missing => fallback {} returns NaNs)
    meta = metrics_json.get(json_key, {})

    # Compute calibration diagnostics from raw probabilities (these are not stored in JSONs)
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=10))

    # Normalise target name so median_risk_tuned plots together with median_risk
    display_key = json_key.replace("_tuned", "")

    # Merge JSON-provided values with computed values
    return {
        "model": model_name,
        "target": display_key,   # Normalised name for consistent plotting
        "roc_auc": meta.get("roc_auc", np.nan),
        "f1": meta.get("f1", np.nan),
        "accuracy": meta.get("accuracy", np.nan),
        "precision": meta.get("precision", np.nan),
        "recall": meta.get("recall", np.nan),
        "brier": brier,
        "ece": ece,
    }

def collect_regression_metrics(model_name, metrics_json, json_key):
    """
    Collect regression metrics strictly from the JSON (no recomputation here).

    Inputs:
      - model_name: model label
      - metrics_json: JSON-loaded dict for this model
      - json_key: key in metrics_json to read RMSE/R² (e.g. "pct_time_high", "pct_time_high_raw_cal")

    Returns:
      - dict with fields: model, target='pct_time_high', rmse, r2
    """
    meta = metrics_json.get(json_key, {})
    # Use NaN when not present (explicit)
    return {
        "model": model_name,
        "target": "pct_time_high",
        "rmse": meta.get("rmse", np.nan),
        "r2": meta.get("r2", np.nan),
    }

# -----------------------
# Build comparison table (CSV)
# -----------------------
"""
Construct the unified comparison metrics table for LightGBM and TCN_refined.

Notes:
- For classification targets we merge JSON (AUC/F1/...) with computed Brier & ECE (from CSV).
- For regression we pull precomputed metrics from JSON:
    - LightGBM uses "pct_time_high" key
    - TCN uses "pct_time_high_raw_cal" key (explicit calibrated raw metric as requested)
- The table is intentionally compact: one row per (model, target) with key metrics.
"""
print("Building comparison metrics table...")

rows = [
    # LightGBM classification: max and median (use JSON for AUC/F1; compute Brier/ECE from predictions CSV)
    collect_class_metrics("LightGBM", df_lgb, lgb_metrics, LGB_COLS["y_true_max"], LGB_COLS["prob_max"], "max_risk"),
    collect_class_metrics("LightGBM", df_lgb, lgb_metrics, LGB_COLS["y_true_median"], LGB_COLS["prob_median"], "median_risk"),
    # LightGBM regression: use JSON key "pct_time_high" (precomputed RMSE/R2)
    collect_regression_metrics("LightGBM", lgb_metrics, "pct_time_high"),

    # TCN classification: max (use JSON) and median (use tuned JSON key median_risk_tuned)
    collect_class_metrics("TCN_refined", df_tcn, tcn_metrics, TCN_COLS["y_true_max"], TCN_COLS["prob_max"], "max_risk"),
    collect_class_metrics("TCN_refined", df_tcn, tcn_metrics, TCN_COLS["y_true_median"], TCN_COLS["prob_median"], "median_risk_tuned"),
    # TCN regression: use calibrated raw JSON key explicitly (pct_time_high_raw_cal)
    collect_regression_metrics("TCN_refined", tcn_metrics, "pct_time_high_raw_cal"),
]

# Save comparison DataFrame to CSV for record
df_comp = pd.DataFrame(rows)
df_comp.to_csv(COMPARISON_CSV, index=False)
print(f"Saved comparison table → {COMPARISON_CSV}")

# -----------------------
# Classification plots
# -----------------------
"""
Generate classification visualisations:
- ROC curves (overlay both models)
- Precision-Recall curves (overlay both models)
- Calibration reliability diagrams (overlay)
- Probability histograms (side-by-side)

Important detail:
- Ground truth y_true is read from LightGBM predictions CSV (we checked alignment earlier).
- For plotting both models we pass the same y_true array and each model's prob column.
"""

print("Generating classification plots (ROC / PR / Calibration / Histograms)...")

# iterate targets: name is used for file naming; pcol is the column name in each CSV for model probabilities
for name, pcol in [("max_risk", "prob_max"), ("median_risk", "prob_median")]:
    # Build ground-truth column name for extraction from LightGBM CSV
    # e.g. name="max_risk" -> base = "max" -> y_true column = "y_true_max"
    base = name.split("_")[0]
    y_true_col = f"y_true_{base}"

    # Extract ground-truth from LightGBM CSV (dtype -> float)
    # NOTE: we use LightGBM's y_true because we've verified ordering; this is identical to TCN's y_true column if aligned.
    y_true = df_lgb[y_true_col].astype(float).values

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_true, df_lgb[pcol].astype(float).values, ax, "LightGBM")  # LightGBM ROC (AUC printed in legend)
    plot_roc(y_true, df_tcn[pcol].astype(float).values, ax, "TCN_refined")  # TCN ROC
    ax.set_title(f"ROC — {name}")
    ax.legend()
    fig.savefig(OUT_DIR / f"roc_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC plot → {OUT_DIR / f'roc_{name}.png'}")

    # ---- Precision-Recall ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_true, df_lgb[pcol].astype(float).values, ax, "LightGBM")   # LightGBM PR (AP in legend)
    plot_pr(y_true, df_tcn[pcol].astype(float).values, ax, "TCN_refined")  # TCN PR
    ax.set_title(f"Precision-Recall — {name}")
    ax.legend()
    fig.savefig(OUT_DIR / f"pr_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR plot → {OUT_DIR / f'pr_{name}.png'}")

    # ---- Calibration curve (reliability diagram) ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_calibration_curve(y_true, df_lgb[pcol].astype(float).values, ax, "LightGBM")
    plot_calibration_curve(y_true, df_tcn[pcol].astype(float).values, ax, "TCN_refined")
    ax.set_title(f"Calibration Curve — {name}")
    ax.legend()
    fig.savefig(OUT_DIR / f"calibration_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot → {OUT_DIR / f'calibration_{name}.png'}")

    # ---- Calibration histograms (side-by-side) ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Left: LightGBM probability distribution
    axs[0].hist(df_lgb[pcol].astype(float).values, bins=10, range=(0.0, 1.0))
    axs[0].set_title("LightGBM probability histogram")
    axs[0].set_xlabel("Predicted probability")
    axs[0].set_ylabel("Count (test patients)")
    # Right: TCN probability distribution
    axs[1].hist(df_tcn[pcol].astype(float).values, bins=10, range=(0.0, 1.0))
    axs[1].set_title("TCN_refined probability histogram")
    axs[1].set_xlabel("Predicted probability")
    fig.suptitle(f"Calibration Histograms — {name}")
    fig.savefig(OUT_DIR / f"calibration_hist_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration hist → {OUT_DIR / f'calibration_hist_{name}.png'}")

print("Classification plots saved.")

# -----------------------
# Regression plots
# -----------------------
"""
Generate regression visualisations:
- True vs Predicted scatter (both models overlayed)
- Residual histograms with KDE (both models side-by-side)
- Error vs Truth scatter (residual vs true)

Important:
- Use explicit raw prediction columns:
    - LightGBM: y_true_reg, y_pred_reg
    - TCN: y_true_reg, y_pred_reg_raw (calibrated raw predictions)
"""
print("Generating regression plots (scatter / residuals / error-vs-truth)...")

# Extract explicit columns for regression (as floats)
y_true_lgb = df_lgb[LGB_COLS["y_true_reg"]].astype(float).values
y_pred_lgb = df_lgb[LGB_COLS["y_pred_reg"]].astype(float).values

y_true_tcn = df_tcn[TCN_COLS["y_true_reg"]].astype(float).values
y_pred_tcn = df_tcn[TCN_COLS["y_pred_reg_raw"]].astype(float).values

# Sanity: check that regression ground-truth arrays match between models (ordering)
if not np.allclose(y_true_lgb, y_true_tcn, atol=1e-8):
    # If mismatch, warn — plots will still be produced but need inspection (likely patient ordering problem)
    print("WARNING: regression ground-truth mismatch between LightGBM and TCN. Verify ordering!")

# ---- Scatter: true vs predicted (overlay both models) ----
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, y_pred_lgb, alpha=0.85, label="LightGBM")
ax.scatter(y_true_tcn, y_pred_tcn, alpha=0.85, label="TCN_refined", marker="x")
# Plot identity line for reference (perfect prediction)
mn = min(np.nanmin(y_true_lgb), np.nanmin(y_pred_lgb), np.nanmin(y_pred_tcn))
mx = max(np.nanmax(y_true_lgb), np.nanmax(y_pred_lgb), np.nanmax(y_pred_tcn))
ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Predicted pct_time_high")
ax.set_title("Regression Scatter — pct_time_high")
ax.legend()
fig.savefig(OUT_DIR / "regression_scatter_pct_time_high.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved regression scatter → {OUT_DIR / 'regression_scatter_pct_time_high.png'}")

# ---- Residual histograms + KDE (side-by-side) ----
# Residuals = predicted - true
res_lgb = y_pred_lgb - y_true_lgb
res_tcn = y_pred_tcn - y_true_tcn

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# LightGBM residuals
grid_l = np.linspace(np.nanmin(res_lgb), np.nanmax(res_lgb), 200)
axs[0].hist(res_lgb, bins=12, density=True, alpha=0.6)
# KDE overlay (bandwidth defaults to Silverman's rule unless specified)
axs[0].plot(grid_l, kde_1d(res_lgb, grid_l), lw=1.4)
axs[0].set_title("LightGBM residuals (pred - true)")
axs[0].set_xlabel("Residual")
axs[0].set_ylabel("Density")

# TCN residuals
grid_t = np.linspace(np.nanmin(res_tcn), np.nanmax(res_tcn), 200)
axs[1].hist(res_tcn, bins=12, density=True, alpha=0.6)
axs[1].plot(grid_t, kde_1d(res_tcn, grid_t), lw=1.4)
axs[1].set_title("TCN_refined residuals (pred - true)")
axs[1].set_xlabel("Residual")

fig.suptitle("Residual Distributions — pct_time_high")
fig.savefig(OUT_DIR / "regression_residuals_pct_time_high.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved residual distributions → {OUT_DIR / 'regression_residuals_pct_time_high.png'}")

# ---- Error vs Truth (residual vs true) ----
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, res_lgb, alpha=0.8, label="LightGBM")
ax.scatter(y_true_tcn, res_tcn, alpha=0.8, label="TCN_refined", marker="x")
ax.axhline(0, linestyle="--", color="gray")  # reference zero error line
ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Residual (pred - true)")
ax.set_title("Error vs Truth — pct_time_high")
ax.legend()
fig.savefig(OUT_DIR / "regression_error_vs_truth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved error-vs-truth → {OUT_DIR / 'regression_error_vs_truth.png'}")

print("Regression plots saved.")

# -----------------------
# Metric Comparison Bar Charts
# -----------------------
"""
Build one grouped bar chart per target:
- max_risk: ROC AUC, F1, Brier, ECE (LightGBM vs TCN)
- median_risk: same as above, but TCN uses median_risk_tuned from JSON
- pct_time_high: RMSE and R² (both models; TCN uses pct_time_high_raw_cal)
"""

print("Building metric comparison bar charts...")

# Classification
for tgt in ["max_risk","median_risk"]:
    sub = df_comp[df_comp["target"] == tgt]
    if sub.empty: continue
    x = np.arange(len(sub)); width=0.2
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x-1.5*width, sub["roc_auc"], width, label="ROC AUC")
    ax.bar(x-0.5*width, sub["f1"], width, label="F1")
    ax.bar(x+0.5*width, sub["brier"], width, label="Brier")
    ax.bar(x+1.5*width, sub["ece"], width, label="ECE")
    ax.set_xticks(x); ax.set_xticklabels(sub["model"])
    ax.set_title(f"Classification Metric Comparison — {tgt}")
    ax.legend(); fig.savefig(OUT_DIR/f"metrics_comparison_{tgt}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# Regression
sub = df_comp[df_comp["target"] == "pct_time_high"]
if not sub.empty:
    x = np.arange(len(sub)); width=0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x-width/2, sub["rmse"], width, label="RMSE")
    ax.bar(x+width/2, sub["r2"], width, label="R²")
    ax.set_xticks(x); ax.set_xticklabels(sub["model"])
    ax.set_title("Regression Metric Comparison — pct_time_high")
    ax.legend(); fig.savefig(OUT_DIR/"metrics_comparison_pct_time_high.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("All plots and comparison tables successfully saved to:")
print(f"   {OUT_DIR}")