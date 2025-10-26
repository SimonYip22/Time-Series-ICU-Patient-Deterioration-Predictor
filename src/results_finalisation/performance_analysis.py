# src/results_finalisation/performance_analysis.py
"""
performance_analysis.py

Title: Final performance visualisations and cross-model comparison for LightGBM and TCN.

Purpose:
- This script consolidates all performance results from Phase 4.5 (TCN refined) and Phase 5 (LightGBM retrained) into a single Phase 6 comparative analysis.
- It aligns both models on the same patient set, computes additional calibration diagnostics (Brier and ECE), and generates standardised visualisations for classification and regression tasks.
- In addition to generating visual plots, this script **saves all underlying numeric data** used for those plots (e.g. ROC, PR, calibration, histograms, regression scatter, and residuals) as CSV files.
- Saving the numeric data underlying each plot enables later quantitative analysis, reproducibility, and secondary processing (e.g. statistical comparison, re-plotting, or integration with other tools).
- This avoids reliance on visual approximations from plots alone, and ensures all numeric results are available for future reference and audit.


Summary of Workflow:
1. Load prediction CSVs and metrics JSONs for both models.
2. Use explicit column mappings and metric keys (no inference, no conversions).
    - LightGBM predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg
    - TCN predictions CSV columns: y_true_max, prob_max, y_true_median, prob_median, y_true_reg, y_pred_reg_raw
    - LightGBM metrics JSON: keys for "max_risk", "median_risk", "pct_time_high"
    - TCN metrics JSON: keys for "max_risk", "median_risk_tuned" and "pct_time_high_raw_cal"
3. Compute calibration metrics (Brier score, ECE) directly from probability predictions.
4. Merge metrics (from JSON + computed) into one comparison table.
5. Generate numeric plot metrics including summary statistics (mean, std, min, max, skew, kurtosis).
6. Generate all plots (classification, regression, calibration, residuals, and metric bars).
7. Save everything into `src/results_finalisation/comparison_plots/` (for plots) and `src/results_finalisation/comparison_metrics/` (for numeric data).

Outputs:
- comparison_table.csv (aggregated metrics) saved in `comparison_metrics/`.
- All numeric data underlying each plot (ROC, PR, calibration curves, probability histograms, regression scatter, residuals) are saved as 12 CSVs in `comparison_metrics/`.
    - Classification: 2x ROC, 2x Precision-Recall, 2x Calibration, 2x Probability histograms
    - Regression: 1x Scatter, 1x Residual, 1x Residual KDE, 1x error-vs-truth
- All 14 figures (plots) are saved as PNGs in `comparison_plots/`:
    - Classification: 2x ROC, 2x Precision-Recall, 2x Calibration curves, 2x Probability histograms
    - Regression: 1x Regression scatter, 1x residual distributions, 1x error-vs-truth
    - 3x Metric comparison grouped bars for each target (3 separate charts: max_risk, median_risk, pct_time_high)
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

# Output folder for numeric metrics used in plots
METRICS_OUT_DIR = ROOT / "results_finalisation" / "comparison_metrics"
METRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output folder for all plots and comparison tables
PLOTS_OUT_DIR = ROOT / "results_finalisation" / "comparison_plots"
PLOTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output comparison table CSV summarising metrics for all tasks and models
COMPARISON_CSV = METRICS_OUT_DIR / "comparison_table.csv"

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
    # Important: use raw predictions for regression comparison with LightGBM
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
    - TCN uses "pct_time_high_raw_cal" key (explicit calibrated raw metric)
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
# Classification plots ( ROC / PR / Calibration / Histogram )
# -----------------------
"""
Generates and saves all classification visualisations and their underlying numeric data.

Purpose:
- Compare LightGBM vs TCN_refined on classification targets (`max_risk`, `median_risk`)
- Each plot’s numeric data is saved as a single combined CSV (both models side-by-side)
- This allows later quantitative interpretation and reproducibility

Visualisations produced:
1. ROC curve (overlay both models)
2. Precision–Recall curve (overlay both models)
3. Calibration (reliability) curve (overlay both models)
4. Probability histograms (side-by-side distributions)

All outputs:
- Plots → saved as PNGs in `comparison_plots/`
- Numeric data → saved as CSVs in `comparison_metrics/`

File naming convention:
    roc_<target>.csv, pr_<target>.csv, calibration_<target>.csv, prob_hist_<target>.csv
"""

print("Generating classification plots (ROC / PR / Calibration / Histograms)...")

# Iterate over both classification targets
for name, pcol in [("max_risk", "prob_max"), ("median_risk", "prob_median")]:
    # -----------------------
    # 1. Extract data
    # -----------------------
    # Derive ground-truth column (e.g. name="max_risk" → y_true_max)
    base = name.split("_")[0]
    y_true_col = f"y_true_{base}"

    # Ground-truth labels (verified identical across both models)
    y_true = df_lgb[y_true_col].astype(float).values

    # Extract both models' probability predictions
    y_prob_lgb = df_lgb[pcol].astype(float).values
    y_prob_tcn = df_tcn[pcol].astype(float).values

    # -----------------------
    # 2. Compute and save all numeric data
    # -----------------------
    """
    Each plot type is saved as one combined CSV for both models.
    Arrays are padded with NaNs to align row lengths where necessary.
    """

    # --- ROC curve ---
    fpr_lgb, tpr_lgb, _ = roc_curve(y_true, y_prob_lgb)
    fpr_tcn, tpr_tcn, _ = roc_curve(y_true, y_prob_tcn)
    auc_lgb = roc_auc_score(y_true, y_prob_lgb)
    auc_tcn = roc_auc_score(y_true, y_prob_tcn)
    prevalence = y_true.mean()

    pd.DataFrame({
        "fpr_LightGBM": fpr_lgb,
        "tpr_LightGBM": tpr_lgb,
        "fpr_TCN_refined": np.pad(fpr_tcn, (0, max(0, len(fpr_lgb)-len(fpr_tcn))), constant_values=np.nan),
        "tpr_TCN_refined": np.pad(tpr_tcn, (0, max(0, len(tpr_lgb)-len(tpr_tcn))), constant_values=np.nan),
        "auc_LightGBM": [auc_lgb]*len(fpr_lgb),
        "auc_TCN_refined": [auc_tcn]*len(fpr_lgb),
        "prevalence": [prevalence]*len(fpr_lgb)
    }).to_csv(METRICS_OUT_DIR / f"roc_{name}.csv", index=False)

    # --- Precision–Recall curve ---
    precision_lgb, recall_lgb, _ = precision_recall_curve(y_true, y_prob_lgb)
    precision_tcn, recall_tcn, _ = precision_recall_curve(y_true, y_prob_tcn)
    ap_lgb = average_precision_score(y_true, y_prob_lgb)
    ap_tcn = average_precision_score(y_true, y_prob_tcn)
    
    # Ensure both recall and precision arrays are padded to the same length.
    def pad_to(arr, n):
        return np.pad(arr, (0, max(0, n - len(arr))), constant_values=np.nan)[:n]

    max_len = max(len(recall_lgb), len(recall_tcn))

    pd.DataFrame({
        "recall_LightGBM": pad_to(recall_lgb, max_len),
        "precision_LightGBM": pad_to(precision_lgb, max_len),
        "recall_TCN_refined": pad_to(recall_tcn, max_len),
        "precision_TCN_refined": pad_to(precision_tcn, max_len),
        "ap_LightGBM": [ap_lgb]*max_len,
        "ap_TCN_refined": [ap_tcn]*max_len
    }).to_csv(METRICS_OUT_DIR / f"pr_{name}.csv", index=False)

    # --- Calibration curve (reliability diagram) ---
    frac_pos_lgb, mean_pred_lgb = calibration_curve(y_true, y_prob_lgb, n_bins=10, strategy="uniform")
    frac_pos_tcn, mean_pred_tcn = calibration_curve(y_true, y_prob_tcn, n_bins=10, strategy="uniform")
    brier_lgb = brier_score_loss(y_true, y_prob_lgb)
    brier_tcn = brier_score_loss(y_true, y_prob_tcn)
    ece_lgb = expected_calibration_error(y_true, y_prob_lgb)
    ece_tcn = expected_calibration_error(y_true, y_prob_tcn)
    # Cast the histogram arrays to float so pad_to function will work 
    n_samples_lgb = np.histogram(y_prob_lgb, bins=10, range=(0,1))[0].astype(float)
    n_samples_tcn = np.histogram(y_prob_tcn, bins=10, range=(0,1))[0].astype(float)

    # Determine max length between both models' calibration arrays
    max_len = max(len(mean_pred_lgb), len(mean_pred_tcn))

    # Pad all arrays to the same length
    def pad_to(arr, n):
        arr = np.asarray(arr, dtype=float)
        return np.pad(arr, (0, max(0, n - len(arr))), constant_values=np.nan)[:n]

    # Combine both models' calibration data into one unified CSV
    pd.DataFrame({
        "mean_pred_LightGBM": pad_to(mean_pred_lgb, max_len),
        "frac_pos_LightGBM": pad_to(frac_pos_lgb, max_len),
        "brier_LightGBM": pad_to([brier_lgb]*len(frac_pos_lgb), max_len),
        "ece_LightGBM": pad_to([ece_lgb]*len(frac_pos_lgb), max_len),
        "n_samples_LightGBM": pad_to(n_samples_lgb, max_len),
        "mean_pred_TCN_refined": pad_to(mean_pred_tcn, max_len),
        "frac_pos_TCN_refined": pad_to(frac_pos_tcn, max_len),
        "brier_TCN_refined": pad_to([brier_tcn]*len(frac_pos_tcn), max_len),
        "ece_TCN_refined": pad_to([ece_tcn]*len(frac_pos_tcn), max_len),
        "n_samples_TCN_refined": pad_to(n_samples_tcn, max_len)
    }).to_csv(METRICS_OUT_DIR / f"calibration_{name}.csv", index=False)

    # --- Probability histograms ---
    """
    - mean: average predicted probability
    - std: standard deviation of predicted probabilities
    - min / max: range of predicted probabilities
    - skew: asymmetry of the distribution
    - kurt: tail heaviness of the distribution

    """

    from scipy.stats import skew, kurtosis 

    # Function to compute key descriptive statistics of an array
    def summary_stats(arr):
        # Returns mean, standard deviation, minimum, maximum, skewness, and kurtosis
        return arr.mean(), arr.std(), arr.min(), arr.max(), skew(arr), kurtosis(arr)
    # Compute summary statistics for each model's predicted probabilities
    mean_lgb, std_lgb, min_lgb, max_lgb, skew_lgb, kurt_lgb = summary_stats(y_prob_lgb)
    mean_tcn, std_tcn, min_tcn, max_tcn, skew_tcn, kurt_tcn = summary_stats(y_prob_tcn)

    pd.DataFrame({
        "pred_prob_LightGBM": y_prob_lgb,
        "mean_LightGBM": [mean_lgb]*len(y_prob_lgb),
        "std_LightGBM": [std_lgb]*len(y_prob_lgb),
        "min_LightGBM": [min_lgb]*len(y_prob_lgb),
        "max_LightGBM": [max_lgb]*len(y_prob_lgb),
        "skew_LightGBM": [skew_lgb]*len(y_prob_lgb),
        "kurt_LightGBM": [kurt_lgb]*len(y_prob_lgb),
        "pred_prob_TCN_refined": y_prob_tcn,
        "mean_TCN_refined": [mean_tcn]*len(y_prob_tcn),
        "std_TCN_refined": [std_tcn]*len(y_prob_tcn),
        "min_TCN_refined": [min_tcn]*len(y_prob_tcn),
        "max_TCN_refined": [max_tcn]*len(y_prob_tcn),
        "skew_TCN_refined": [skew_tcn]*len(y_prob_tcn),
        "kurt_TCN_refined": [kurt_tcn]*len(y_prob_tcn)
    }).to_csv(METRICS_OUT_DIR / f"prob_hist_{name}.csv", index=False)

    # -----------------------
    # 3. Generate and save plots
    # -----------------------

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_true, y_prob_lgb, ax, "LightGBM")
    plot_roc(y_true, y_prob_tcn, ax, "TCN_refined")
    ax.set_title(f"ROC — {name}")
    ax.legend()
    fig.savefig(PLOTS_OUT_DIR / f"roc_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC plot → {PLOTS_OUT_DIR / f'roc_{name}.png'}")

    # ---- Precision–Recall ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_true, y_prob_lgb, ax, "LightGBM")
    plot_pr(y_true, y_prob_tcn, ax, "TCN_refined")
    ax.set_title(f"Precision–Recall — {name}")
    ax.legend()
    fig.savefig(PLOTS_OUT_DIR / f"pr_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR plot → {PLOTS_OUT_DIR / f'pr_{name}.png'}")

    # ---- Calibration curve ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_calibration_curve(y_true, y_prob_lgb, ax, "LightGBM")
    plot_calibration_curve(y_true, y_prob_tcn, ax, "TCN_refined")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # perfect calibration line
    ax.set_title(f"Calibration Curve — {name}")
    ax.legend()
    fig.savefig(PLOTS_OUT_DIR / f"calibration_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot → {PLOTS_OUT_DIR / f'calibration_{name}.png'}")

    # ---- Probability histograms ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Left: LightGBM
    axs[0].hist(y_prob_lgb, bins=10, range=(0.0, 1.0))
    axs[0].set_title("LightGBM probability histogram")
    axs[0].set_xlabel("Predicted probability")
    axs[0].set_ylabel("Count (test patients)")
    # Right: TCN
    axs[1].hist(y_prob_tcn, bins=10, range=(0.0, 1.0))
    axs[1].set_title("TCN_refined probability histogram")
    axs[1].set_xlabel("Predicted probability")
    fig.suptitle(f"Probability Distributions — {name}")
    fig.savefig(PLOTS_OUT_DIR / f"prob_hist_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probability histogram → {PLOTS_OUT_DIR / f'prob_hist_{name}.png'}")

print("Classification plots and numeric data saved successfully.")

# -----------------------
# Regression plots (scatter / residuals / error-vs-truth)
# -----------------------
"""
Generates and saves regression visualisations + their numeric data.

Purpose:
- To compare LightGBM vs TCN_refined performance on the continuous regression target (`pct_time_high`)
- All underlying numeric data are saved to CSVs (1 file per plot type) for later quantitative interpretation.
- This avoids reliance on visual estimation from the PNGs alone.

Outputs:
- `comparison_plots/` → PNG figures
- `comparison_metrics/` → CSVs containing numeric data for all plots

Visualisations produced:
1. True vs Predicted scatter (both models overlaid)
2. Residual histograms with KDE (both models side-by-side)
3. Error vs Truth scatter (residual vs true)

Important details:
- Explicit raw prediction columns:
    - LightGBM: y_true_reg, y_pred_reg
    - TCN: y_true_reg, y_pred_reg_raw (calibrated raw predictions)
- All CSVs combine both models’ data side-by-side for direct comparison.
"""

print("Generating regression plots (scatter / residuals / error-vs-truth)...")

# -----------------------
# 1. Extract regression columns
# -----------------------
# Convert all values to float arrays for numeric operations
y_true_lgb = df_lgb[LGB_COLS["y_true_reg"]].astype(float).values
y_pred_lgb = df_lgb[LGB_COLS["y_pred_reg"]].astype(float).values

y_true_tcn = df_tcn[TCN_COLS["y_true_reg"]].astype(float).values
y_pred_tcn = df_tcn[TCN_COLS["y_pred_reg_raw"]].astype(float).values

# Verify ground-truth alignment (sanity check)
if not np.allclose(y_true_lgb, y_true_tcn, atol=1e-8):
    print("WARNING: Regression ground-truth mismatch between LightGBM and TCN. Verify ordering!")

# -----------------------
# 2. Save numeric data for scatter + residuals
# -----------------------
"""
We save ONE CSV per plot type, combining both models’ data.
Each file contains columns for both LightGBM and TCN_refined.

Saved files:
- scatter_combined.csv → true vs predicted values for both models
- residuals_combined.csv → residuals + KDE data for both models
- error_vs_truth_combined.csv → true values vs residuals (for error-vs-truth plot)
"""

target_name = "pct_time_high"  # use explicit target

# --- Residuals (predicted - true) ---
res_lgb = y_pred_lgb - y_true_lgb
res_tcn = y_pred_tcn - y_true_tcn

# --- Compute summary stats for residuals ---
def summary_stats(arr):
    """Compute mean, std, min, max, skewness, kurtosis."""
    return arr.mean(), arr.std(), arr.min(), arr.max(), skew(arr), kurtosis(arr)

mean_res_lgb, std_res_lgb, min_res_lgb, max_res_lgb, skew_res_lgb, kurt_res_lgb = summary_stats(res_lgb)
mean_res_tcn, std_res_tcn, min_res_tcn, max_res_tcn, skew_res_tcn, kurt_res_tcn = summary_stats(res_tcn)

# --- Save scatter numeric data (True vs Pred) ---
pd.DataFrame({
    "y_true_LightGBM": y_true_lgb,
    "y_pred_LightGBM": y_pred_lgb,
    "y_true_TCN_refined": y_true_tcn,
    "y_pred_TCN_refined": y_pred_tcn
}).to_csv(METRICS_OUT_DIR / f"scatter_{target_name}.csv", index=False)

# --- Save residual numeric data (values + stats) ---
pd.DataFrame({
    "residual_LightGBM": res_lgb,
    "mean_res_LightGBM": [mean_res_lgb]*len(res_lgb),
    "std_res_LightGBM": [std_res_lgb]*len(res_lgb),
    "min_res_LightGBM": [min_res_lgb]*len(res_lgb),
    "max_res_LightGBM": [max_res_lgb]*len(res_lgb),
    "skew_res_LightGBM": [skew_res_lgb]*len(res_lgb),
    "kurt_res_LightGBM": [kurt_res_lgb]*len(res_lgb),
    "residual_TCN_refined": np.pad(res_tcn, (0, max(0, len(res_lgb) - len(res_tcn))), constant_values=np.nan)[:len(res_lgb)],
    "mean_res_TCN_refined": [mean_res_tcn]*len(res_lgb),
    "std_res_TCN_refined": [std_res_tcn]*len(res_lgb),
    "min_res_TCN_refined": [min_res_tcn]*len(res_lgb),
    "max_res_TCN_refined": [max_res_tcn]*len(res_lgb),
    "skew_res_TCN_refined": [skew_res_tcn]*len(res_lgb),
    "kurt_res_TCN_refined": [kurt_res_tcn]*len(res_lgb)
}).to_csv(METRICS_OUT_DIR / f"residuals_{target_name}.csv", index=False)

# --- Save KDE numeric data ---
grid_l = np.linspace(np.nanmin(res_lgb), np.nanmax(res_lgb), 200)
grid_t = np.linspace(np.nanmin(res_tcn), np.nanmax(res_tcn), 200)
kde_l = kde_1d(res_lgb, grid_l)
kde_t = kde_1d(res_tcn, grid_t)

pd.DataFrame({
    "grid_LightGBM": grid_l,
    "kde_LightGBM": kde_l,
    "grid_TCN_refined": np.pad(grid_t, (0, max(0, len(grid_l) - len(grid_t))), constant_values=np.nan)[:len(grid_l)],
    "kde_TCN_refined": np.pad(kde_t, (0, max(0, len(kde_l) - len(kde_t))), constant_values=np.nan)[:len(kde_l)]
}).to_csv(METRICS_OUT_DIR / f"residuals_kde_{target_name}.csv", index=False)


# --- Save error vs truth numeric data (with residuals stats) ---
pd.DataFrame({
    "y_true_LightGBM": y_true_lgb,
    "residual_LightGBM": res_lgb,
    "mean_res_LightGBM": [mean_res_lgb]*len(res_lgb),
    "std_res_LightGBM": [std_res_lgb]*len(res_lgb),
    "min_res_LightGBM": [min_res_lgb]*len(res_lgb),
    "max_res_LightGBM": [max_res_lgb]*len(res_lgb),
    "skew_res_LightGBM": [skew_res_lgb]*len(res_lgb),
    "kurt_res_LightGBM": [kurt_res_lgb]*len(res_lgb),
    "y_true_TCN_refined": y_true_tcn,
    "residual_TCN_refined": res_tcn,
    "mean_res_TCN_refined": [mean_res_tcn]*len(res_tcn),
    "std_res_TCN_refined": [std_res_tcn]*len(res_tcn),
    "min_res_TCN_refined": [min_res_tcn]*len(res_tcn),
    "max_res_TCN_refined": [max_res_tcn]*len(res_tcn),
    "skew_res_TCN_refined": [skew_res_tcn]*len(res_tcn),
    "kurt_res_TCN_refined": [kurt_res_tcn]*len(res_tcn)
}).to_csv(METRICS_OUT_DIR / f"error_vs_truth_{target_name}.csv", index=False)

# -----------------------
# 3. Generate and save plots
# -----------------------

# ---- Scatter: True vs Predicted (overlay both models) ----
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, y_pred_lgb, alpha=0.85, label="LightGBM")
ax.scatter(y_true_tcn, y_pred_tcn, alpha=0.85, label="TCN_refined", marker="x")

# Plot identity line for perfect predictions
mn = min(np.nanmin(y_true_lgb), np.nanmin(y_pred_lgb), np.nanmin(y_pred_tcn))
mx = max(np.nanmax(y_true_lgb), np.nanmax(y_pred_lgb), np.nanmax(y_pred_tcn))
ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray")

ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Predicted pct_time_high")
ax.set_title("Regression Scatter — pct_time_high")
ax.legend()
fig.savefig(PLOTS_OUT_DIR / "scatter_pct_time_high.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved regression scatter → {PLOTS_OUT_DIR / 'scatter_pct_time_high.png'}")

# ---- Residual histograms + KDE (side-by-side) ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Left: LightGBM residuals
axs[0].hist(res_lgb, bins=12, density=True, alpha=0.6)
axs[0].plot(grid_l, kde_l, lw=1.4)
axs[0].set_title("LightGBM residuals (pred - true)")
axs[0].set_xlabel("Residual")
axs[0].set_ylabel("Density")

# Right: TCN residuals
axs[1].hist(res_tcn, bins=12, density=True, alpha=0.6)
axs[1].plot(grid_t, kde_t, lw=1.4)
axs[1].set_title("TCN_refined residuals (pred - true)")
axs[1].set_xlabel("Residual")

fig.suptitle("Residual Distributions — pct_time_high")
fig.savefig(PLOTS_OUT_DIR / "residuals_pct_time_high.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved residual distributions → {PLOTS_OUT_DIR / 'residuals_pct_time_high.png'}")

# ---- Error vs Truth (residual vs true) ----
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_true_lgb, res_lgb, alpha=0.8, label="LightGBM")
ax.scatter(y_true_tcn, res_tcn, alpha=0.8, label="TCN_refined", marker="x")
ax.axhline(0, linestyle="--", color="gray")  # reference zero error line
ax.set_xlabel("True pct_time_high")
ax.set_ylabel("Residual (pred - true)")
ax.set_title("Error vs Truth — pct_time_high")
ax.legend()
fig.savefig(PLOTS_OUT_DIR / "error_vs_truth_pct_time_high.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved error-vs-truth → {PLOTS_OUT_DIR / 'error_vs_truth_pct_time_high.png'}")

print("Regression plots and numeric data saved successfully.")

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
    ax.legend(); fig.savefig(PLOTS_OUT_DIR/f"metrics_comparison_{tgt}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# Regression
sub = df_comp[df_comp["target"] == "pct_time_high"]
if not sub.empty:
    x = np.arange(len(sub)); width=0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x-width/2, sub["rmse"], width, label="RMSE")
    ax.bar(x+width/2, sub["r2"], width, label="R²")
    ax.set_xticks(x); ax.set_xticklabels(sub["model"])
    ax.set_title("Regression Metric Comparison — pct_time_high")
    ax.legend(); fig.savefig(PLOTS_OUT_DIR/"metrics_comparison_pct_time_high.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("All numeric metrics used in plots successfully saved to:")
print(f"   {METRICS_OUT_DIR}")
print("All plots and comparison tables successfully saved to:")
print(f"   {PLOTS_OUT_DIR}")