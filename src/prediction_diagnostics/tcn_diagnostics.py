# src/prediction_diagnostics/tcn_diagnostics.py

"""
tcn_diagnostics.py

Title: Temporal Convolutional Network (TCN) – Full Diagnostics

Purpose:
- Deep diagnostic evaluation of the TCN model on TEST and VALIDATION sets.
- Uses saved evaluation regression outputs (from tcn_predictions.csv)
  to ensure identical R² to evaluation.
- Recreates classification labels dynamically (max_risk > 2, median_risk == 2).
- Generates threshold sweeps, ROC-AUCs, histograms, scatterplots, residuals,
  and data-distribution diagnostics for both classification and regression.

Outputs:
Saved to: src/prediction_diagnostics/plots/
- Diagnostic plots (histograms, scatter, residuals, data distributions)
- Terminal metrics (F1, ROC-AUC, R², RMSE)
- Training label distributions
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from pathlib import Path
import sys

# -------------------------------
# PROJECT ROOT
# -------------------------------
# Defines the absolute root path to project.
# This ensures all relative paths work regardless of where you run the script.
PROJECT_ROOT = Path("/Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor")

# -------------------------------
# Dataset & model asset locations
# -------------------------------
TEST_DATA_DIR = PROJECT_ROOT / "src/ml_models_tcn/prepared_datasets"
TRAINING_CSV = PROJECT_ROOT / "data/processed_data/news2_features_patient.csv"
PATIENT_SPLITS_PATH = PROJECT_ROOT / "src/ml_models_tcn/deployment_models/preprocessing/patient_splits.json"
CONFIG_PATH = PROJECT_ROOT / "src/ml_models_tcn/trained_models/config.json"
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "src/ml_models_tcn/trained_models/tcn_best.pt"

# -------------------------------
# Output folders (Phase 4.5 Diagnostics)
# -------------------------------
# Results from evaluation (Phase 4) are read here:
RESULTS_DIR = PROJECT_ROOT / "src/prediction_evaluations/results"
EVAL_PRED_CSV = RESULTS_DIR / "tcn_predictions.csv"  # From evaluate_tcn_testset.py

# Evaluation outputs (we still read the evaluation CSV produced by evaluate_tcn_testset.py)
EVAL_RESULTS_DIR = PROJECT_ROOT / "src/prediction_evaluations" / "results"
EVAL_PRED_CSV = EVAL_RESULTS_DIR / "tcn_predictions.csv"   # read-only for diagnostics

# Diagnostics-specific output folder (Phase 4.5)
DIAG_DIR = PROJECT_ROOT / "src" / "prediction_diagnostics"      # top-level diagnostics folder
PLOT_DIR = DIAG_DIR / "plots"                                   # where diagnostics plots go
DIAG_RESULTS_DIR = DIAG_DIR / "results"                         # where diagnostics jsons go

# ensure directories exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load Config and Splits
# -------------------------------
# Loads model configuration (architecture details, kernel size, etc.)
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
print("Model architecture from training config:", cfg["model_architecture"])

# patient_df contains static, patient-level features (risk scores, % time high, etc.)
patient_df = pd.read_csv(TRAINING_CSV).set_index("subject_id")

# splits defines which patient IDs belong to train/val/test.
with open(PATIENT_SPLITS_PATH) as f:
    splits = json.load(f)
train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]

print(f"\nLoaded splits → Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# -------------------------------
# Load evaluation regression results
# -------------------------------
eval_df = pd.read_csv(EVAL_PRED_CSV)
y_true_reg_test = torch.tensor(eval_df["y_true_reg"].values, dtype=torch.float32)
y_pred_reg_test = torch.tensor(eval_df["y_pred_reg"].values, dtype=torch.float32)

# -------------------------------
# Recreate classification + validation regression targets
# -------------------------------
def recreate_y(df, ids):
    df_split = df.loc[ids].copy()
    y_max = torch.tensor((df_split["max_risk"] > 2).astype(float).values)
    y_median = torch.tensor((df_split["median_risk"] == 2).astype(float).values)
    y_reg = torch.tensor(df_split["pct_time_high"].values, dtype=torch.float32)
    return y_max, y_median, y_reg

y_test_max, y_test_median, _ = recreate_y(patient_df, test_ids)
y_val_max, y_val_median, y_val_reg = recreate_y(patient_df, val_ids)

# -------------------------------
# Load Model + Tensors
# -------------------------------
# Dynamically import TCNModel from src/ml_models_tcn. This requires modifying sys.path.
sys.path.append(str(PROJECT_ROOT / "src/ml_models_tcn"))
from tcn_model import TCNModel

# Loads test tensors (features and masks).
# These represent padded time-series data for each patient.
x_test = torch.load(TEST_DATA_DIR / "test.pt")
mask_test = torch.load(TEST_DATA_DIR / "test_mask.pt")

# Initialise model with parameters from config.json
feature_dim = x_test.shape[2]
model = TCNModel(
    num_features=feature_dim,
    num_channels=cfg["model_architecture"]["num_channels"],
    head_hidden=cfg["model_architecture"]["head_hidden"],
    kernel_size=cfg["model_architecture"]["kernel_size"],
    dropout=cfg["model_architecture"]["dropout"]
)

# Load trained weights from file
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
model.eval() # Set to evaluation mode
print("[INFO] Model loaded successfully.\n")

# -------------------------------
# Utility Functions
# -------------------------------
def get_predictions(x, mask):
    with torch.no_grad():
        outputs = model(x, mask)
    return (
        torch.sigmoid(outputs["logit_max"]).numpy(),
        torch.sigmoid(outputs["logit_median"]).numpy(),
        outputs["regression"].squeeze().numpy(),
    )

def plot_hist(prob, title, fname):
    plt.figure()
    plt.hist(prob, bins=20)
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname)
    plt.close()
    print(f"[SAVED] {fname}")

def regression_diagnostics(y_true, y_pred, title, prefix):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title(f"{title} Scatter")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{prefix}_scatter.png")
    plt.close()

    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{title} Residuals")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{prefix}_residuals.png")
    plt.close()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{title} → RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rmse, r2

def threshold_sweep(y_true, prob, name):
    thresholds = np.linspace(0.1, 0.9, 9)
    print(f"\nThreshold sweep: {name}")
    y_true_bin = np.asarray(y_true) >= 0.5
    for t in thresholds:
        preds = np.asarray(prob) >= t
        f1 = f1_score(y_true_bin.astype(int), preds.astype(int))
        acc = np.mean(preds == y_true_bin)
        print(f"  Th={t:.2f} → F1={f1:.3f}, Acc={acc:.3f}")

# -------------------------------
# Run Test Diagnostics
# -------------------------------
print("\n=== TEST SET DIAGNOSTICS ===")

# Classification
prob_max_test, prob_median_test, pred_reg_test_model = get_predictions(x_test, mask_test)
plot_hist(prob_max_test, "Test Max Risk Probabilities", "prob_hist_max_test.png")
threshold_sweep(y_test_max, prob_max_test, "Max Risk")

plot_hist(prob_median_test, "Test Median Risk Probabilities", "prob_hist_median_test.png")
threshold_sweep(y_test_median, prob_median_test, "Median Risk")

# Regression (from eval CSV)
rmse_reg, r2_reg = regression_diagnostics(
    y_true_reg_test.numpy(), y_pred_reg_test.numpy(),
    "Test Regression (Evaluation CSV)", "test_reg"
)

# -------------------------------
# Run Validation Diagnostics
# -------------------------------
print("\n=== VALIDATION SET DIAGNOSTICS ===")
val_x = torch.load(TEST_DATA_DIR / "val.pt")
val_mask = torch.load(TEST_DATA_DIR / "val_mask.pt")

prob_max_val, prob_median_val, pred_reg_val = get_predictions(val_x, val_mask)

plot_hist(prob_max_val, "Val Max Risk", "prob_hist_max_val.png")
plot_hist(prob_median_val, "Val Median Risk", "prob_hist_median_val.png")

rmse_val, r2_val = regression_diagnostics(
    y_val_reg.numpy(), pred_reg_val, "Validation Regression", "val_reg"
)

# -------------------------------
# Data Distribution Diagnostics
# -------------------------------
print("\n=== DATA DISTRIBUTIONS ===")

# --- Classification label balance ---
def plot_label_balance(y, title, fname):
    """Plots histogram for binary label balance."""
    plt.figure()
    plt.hist(y, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("Label value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname)
    plt.close()
    print(f"[SAVED] {fname}")

# Max Risk label stats
pos_max = int(y_test_max.sum())
neg_max = len(y_test_max) - pos_max
print(f"Test Max Risk → 1s: {pos_max}, 0s: {neg_max}, proportion positive: {y_test_max.mean():.3f}")
plot_label_balance(y_test_max.numpy(), "Test Max Risk Label Balance", "dist_test_max_risk_labels.png")

# Median Risk label stats
pos_med = int(y_test_median.sum())
neg_med = len(y_test_median) - pos_med
print(f"Test Median Risk → 1s: {pos_med}, 0s: {neg_med}, proportion positive: {y_test_median.mean():.3f}")
plot_label_balance(y_test_median.numpy(), "Test Median Risk Label Balance", "dist_test_median_risk_labels.png")

# --- Regression value distributions ---
plt.figure()
plt.hist(y_true_reg_test.numpy(), bins=20)
plt.title("True pct_time_high Distribution (Test)")
plt.xlabel("pct_time_high (true)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_DIR / "dist_test_pct_time_high_true.png")
plt.close()

plt.figure()
plt.hist(y_pred_reg_test.numpy(), bins=20)
plt.title("Predicted pct_time_high Distribution (Test)")
plt.xlabel("pct_time_high (predicted)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_DIR / "dist_test_pct_time_high_pred.png")
plt.close()

print("[SAVED] pct_time_high true/predicted distribution plots")
print("Data distribution analysis complete — imbalance and skew patterns visualised.\n")

# -------------------------------
# Training Label Distribution
# -------------------------------
# Checks the label distribution to identify imbalance or skew.
print("\n=== TRAINING LABEL DISTRIBUTION ===")
for col in ["max_risk", "median_risk", "pct_time_high"]:
    if col in patient_df.columns:
        print(f"\n{col}:")
        print(patient_df[col].describe())

# -------------------------------
# Diagnostic Summary
# -------------------------------

# Compute F1 at 0.5 threshold (explicitly for summary)
f1_max = f1_score(y_test_max.numpy(), (prob_max_test >= 0.5).astype(int))
f1_median = f1_score(y_test_median.numpy(), (prob_median_test >= 0.5).astype(int))

# Compute ROC-AUC (threshold-independent ranking metric)
# Guard against degenerate cases where all labels identical
try:
    roc_max = roc_auc_score(y_test_max, prob_max_test)
except ValueError:
    roc_max = float("nan")

try:
    roc_median = roc_auc_score(y_test_median, prob_median_test)
except ValueError:
    roc_median = float("nan")

# Regression values already computed earlier: rmse_reg, r2_reg, rmse_val, r2_val
print("\n=== SUMMARY ===")
print(f"Max Risk → F1 (0.5 threshold)={f1_max:.3f}, ROC-AUC={roc_max:.3f}")
print(f"Median Risk → F1 (0.5 threshold)={f1_median:.3f}, ROC-AUC={roc_median:.3f}")
print(f"Test Regression → RMSE={rmse_reg:.4f}, R²={r2_reg:.4f}")
print(f"Validation Regression → RMSE={rmse_val:.4f}, R²={r2_val:.4f}")
print("\nNote: Median Risk F1=0 or Regression R²<0 suggests class imbalance or label noise.")
print(f"All plots saved to: {PLOT_DIR.resolve()}")
print("\nDiagnostics completed successfully ✅")

# Save summary to JSON
summary = {
  "max_risk": {"f1_0.5": float(f1_max), "roc_auc": float(roc_max)},
  "median_risk": {"f1_0.5": float(f1_median), "roc_auc": float(roc_median)},
  "test_regression": {"rmse": float(rmse_reg), "r2": float(r2_reg)},
  "val_regression": {"rmse": float(rmse_val), "r2": float(r2_val)}
}
with open(DIAG_RESULTS_DIR / "tcn_diagnostics_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)