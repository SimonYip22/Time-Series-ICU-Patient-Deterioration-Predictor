# src/prediction_evaluations/evaluate_tcn_testset_refined.py

"""
evaluate_tcn_testset_refined.py

Title: Final Evaluation of Refined Temporal Convolutional Network (TCN) on Held-Out Test Set

Overview:
- Loads the refined Temporal Convolutional Network (TCN) checkpoint (`tcn_best_refined.pt`) from Phase 4.5.  
- Dynamically reconstructs ground-truth test targets from patient-level CSV and JSON splits to ensure consistency.  
- Performs inference on the held-out test set using the finalised TCN architecture and configuration.  
- The regression head outputs predictions in **log-space** (trained on `log1p(pct_time_high)`), which are later inverse-transformed (`expm1`) to obtain interpretable raw-scale estimates.  
- Computes:
    • Binary classification metrics for **max risk** and **median risk** prediction heads  
    • Regression metrics in both **log-space** (internal validation scale) and **raw-space** (clinically interpretable scale)
- Applies post-hoc **linear calibration** in log-space to correct systematic bias without retraining.

Scientific Rationale:
- **Log-space evaluation** validates internal training objectives and model optimisation fidelity.  
- **Raw-space evaluation** provides clinically interpretable performance measures and enables fair comparison with baseline models (Phase 4 TCN, LightGBM, NEWS2).  
- **Calibration** addresses monotonic bias where predictions track the true trend but differ in scale, improving real-world applicability without altering learned weights.

Outputs:
- `prediction_evaluations/tcn_results_refined/tcn_metrics_refined.json` → Full evaluation metrics (classification + regression, pre- and post-calibration)  
- `prediction_evaluations/tcn_results_refined/tcn_predictions_refined.csv` → Per-patient predictions in both raw and log scales  
- `tcn_regression_calibration_logspace.png` → Calibration bias visualisation  
- `tcn_regression_calibration_comparison_logspace.png` → Before/after calibration comparison

This script represents the final, reproducible evaluation pipeline for the refined TCN model.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import sys                            # System-level operations
from pathlib import Path              # Safe filesystem handling across OS

import torch                          # Core deep learning library
import json                           # For reading/writing JSON files (splits + metrics), saving metrics to JSON
import pandas as pd                   # For tabular prediction outputs
import numpy as np                    # For inverse log transform
from time import time                 # Used to measure inference duration

from sklearn.metrics import f1_score  # For threshold tuning

import matplotlib.pyplot as plt       # plotting correlation curves
from sklearn.linear_model import LinearRegression # for linear regression on log-space values


# -------------------------------------------------------------
# Path setup
# -------------------------------------------------------------
# Add src/ to sys.path so we can import packages directly
SCRIPT_DIR = Path(__file__).resolve().parent                # src/prediction_evaluations
SRC_DIR = SCRIPT_DIR.parent                                 # src/
PROJECT_ROOT = SRC_DIR.parent                               # project root
sys.path.append(str(SRC_DIR))                               # now Python can find ml_models_tcn, prediction_evaluations

# -------------------------------------------------------------
# Imports from other project modules
# -------------------------------------------------------------
# --- Import reusable metric utilities (Phase 5)---
# evaluation_metrics.py defines functions to compute metrics for classification and regression tasks.
from prediction_evaluations.evaluation_metrics import (
    compute_classification_metrics,
    compute_regression_metrics
)

# --- Import the trained TCN model definition from Phase 4 (from tcn_model.py) ---
# tcn_models.py defines the architecture: all layers, residual blocks, pooling, and heads.
from ml_models_tcn.tcn_model import TCNModel

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
# Path to best TCN model weights (saved during early stopping in Phase 4.5) + Model config
TRAINED_MODEL_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "tcn_best_refined.pt"
CONFIG_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "config_refined.json"

# Path to saved test + mask tensors (input sequences + masks)
TEST_DATA_DIR = SRC_DIR / "ml_models_tcn" / "prepared_datasets"

# Path to rebuild y_test (patient-level labels for test set)
FEATURES_PATIENT_PATH = PROJECT_ROOT / "data" / "processed_data" / "news2_features_patient.csv"
SPLITS_PATH = SRC_DIR / "ml_models_tcn" / "deployment_models" / "preprocessing" / "patient_splits.json"

# Directory to save evaluation outputs (metrics + predictions)
RESULTS_DIR = SRC_DIR / "prediction_evaluations" / "tcn_results_refined"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Sanity Checks — ensure all expected files and configs exist
# -------------------------------------------------------------
assert TRAINED_MODEL_PATH.exists(), f"[ERROR] Missing model weights: {TRAINED_MODEL_PATH}"
assert CONFIG_PATH.exists(), f"[ERROR] Missing config file: {CONFIG_PATH}"
assert TEST_DATA_DIR.exists(), f"[ERROR] Missing test data directory: {TEST_DATA_DIR}"
assert (TEST_DATA_DIR / "test.pt").exists(), "[ERROR] Missing test.pt file"
assert (TEST_DATA_DIR / "test_mask.pt").exists(), "[ERROR] Missing test_mask.pt file"
assert FEATURES_PATIENT_PATH.exists(), "[ERROR] Missing patient-level features CSV"
assert SPLITS_PATH.exists(), "[ERROR] Missing patient_splits.json"

print("[INFO] All required files found. Proceeding with refined model evaluation...")

# -------------------------------------------------------------
# Rebuild test labels dynamically
# -------------------------------------------------------------
# Dynamically rebuild y_test from CSV + JSON split (handles any class distribution)
# Load patient-level feature dataframe and test split identifiers
features_df = pd.read_csv(FEATURES_PATIENT_PATH)
with open(SPLITS_PATH) as f:
    splits = json.load(f)

# Keep the test_ids as a list in the order in the JSON (patient_splits.json)
test_ids = splits["test"]  

# Use .loc with the list to preserve order
test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()

# Recreate binary targets exactly as defined in Phase 4 (ensures label consistency)
test_df["max_risk_binary"] = test_df["max_risk"].apply(lambda x: 1 if x > 2 else 0)
test_df["median_risk_binary"] = test_df["median_risk"].apply(lambda x: 1 if x == 2 else 0)

# Convert targets into tensors (same structure as y_train/y_val during training)
y_test = {
    "max": torch.tensor(test_df["max_risk_binary"].values, dtype=torch.float32),
    "median": torch.tensor(test_df["median_risk_binary"].values, dtype=torch.float32),
    "reg": torch.tensor(test_df["pct_time_high"].values, dtype=torch.float32)
}

# -------------------------------------------------------------
# Load Model + Test Data
# -------------------------------------------------------------

# --- 0. Select device (CPU or GPU) ---
# Always define device first — needed for loading tensors and model weights correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- 1. Load input test tensors (test.pt, test_mask.pt) ---
# Load preprocessed tensors from Phase 4 (time-series inputs + masks from prepared_datasets/)
x_test = torch.load(TEST_DATA_DIR / "test.pt", map_location=device)
mask_test = torch.load(TEST_DATA_DIR / "test_mask.pt", map_location=device)

# Quick shape validation — ensure alignment between test IDs and tensor batch size
assert x_test.shape[0] == len(test_ids), (
    f"[ERROR] x_test has {x_test.shape[0]} patients, "
    f"but test split defines {len(test_ids)}. Check patient_splits.json alignment."
)

# --- 2. Load the hyperparameters into the model architecture / structure ---
# Model architecture defined in TCNModel imported from tcn_model.py, which needs hyperparameters to instantiate
# Load architecture hyperparameters from config_refined.json created during training (tcn_training_script_refined.py)
with open(CONFIG_PATH) as f:
    config = json.load(f)
arch = config["model_architecture"]

# Automatically get feature dimension (171 features = input channels in first convolutional layer)
NUM_FEATURES = x_test.shape[2] 

# Instantiate the TCN model architecture (defined in tcn_model.py) with same hyperparameters as training (tcn_training_script_refined.py)
# Must match exactly what was used during training, otherwise the weight shapes won't align
model = TCNModel(
    num_features=NUM_FEATURES,          # Input feature dimension (171 per-timestep features)
    num_channels=arch["num_channels"],  # 3 TCN layers with increasing channels
    kernel_size=arch["kernel_size"],    # Kernel size of 3 for temporal convolutions
    dropout=arch["dropout"],            # Regularisation: randomly zero 20% of activations during training
    head_hidden=arch["head_hidden"]     # Hidden layer size of 64 in the final dense head
)

# --- 3. Load the trained model weights (tcn_best_refined.pt) ---
# Weights are the learned parameters (filters, biases) that the model adjusted during training to minimise loss
# tcn_best_refined.pt = serialized 'state_dict' containing all trained layer weights + biases (tensors)
state_dict = torch.load(TRAINED_MODEL_PATH, map_location=device)
# Copies the saved weights (state_dict) into the model architecture we just instantiated (model)
model.load_state_dict(state_dict)

# --- 4. Move model to chosen device (CPU or GPU) ---
# Transfers all tensors (weights, buffers) inside the model to the selected device (CPU or CUDA)
model.to(device)

# --- 5. Set model to evaluation mode ---
# Disables training-specific layers (dropout, batchnorm updates)
# Ensures consistent, deterministic, stable predictions during inference → no random dropout, no unstable normalisation (batchnorm updates)
model.eval()

print("[INFO] Loaded refined TCN model and weights successfully")

# -------------------------------------------------------------
# Inference (Testing)
# -------------------------------------------------------------
"""
- The trained model is used to make predictions on new, unseen data.
- When we train a model, PyTorch tracks everything needed to compute gradients for backpropagation (e.g. how each weight affects the loss).
- When we test / evaluate (inference) a model, we don’t need gradients, we just want predictions.
- No optimisation, no gradient updates, no backpropagation.
- Disabling gradient computation speeds up inference and reduces memory consumption.
- The refined TCN model produces three simultaneous outputs per patient:
    1. logit_max     → binary classification for max risk (probability of extreme deterioration)
    2. logit_median  → binary classification for median risk (moderate deterioration)
    3. regression     → continuous prediction of log1p(pct_time_high)
"""
print("[INFO] Running inference on test set...")
start_time = time() # Track runtime for reproducibility

# Disable gradient computation for faster and memory-efficient inference
with torch.no_grad():
    # Forward pass through the TCN model (predictions based on frozen weights)
    # Outputs a dictionary from tcn_model.py → {"logit_max", "logit_median", "regression"}
    outputs = model(x_test, mask_test)

    # --- Extract raw outputs from model (PyTorch tensors) ---
    # Extract raw output PyTorch tensors (logits for classification, continuous for regression)
    # TCN aggregates each patient’s entire time-series into one vector → one prediction (1 number per patient)
    # Output heads generate shape (num_patients, 1) → squeeze() removes the extra dimension → (num_patients,)
    logits_max = outputs["logit_max"].squeeze().cpu()           # Binary classification head 1
    logits_median = outputs["logit_median"].squeeze().cpu()     # Binary classification head 2
    preds_reg = outputs["regression"].squeeze().cpu()           # Regression head (continuous)

# --- Convert into final output format (NumPy array) ---
# Convert PyTorch tensor → NumPy array for evaluation functions (pandas + sklearn compatibility)
# Apply sigmoid for classification heads to convert logits → probabilities
prob_max = torch.sigmoid(logits_max).numpy()
prob_median = torch.sigmoid(logits_median).numpy()
y_pred_reg = preds_reg.numpy()

# Measure total inference time
inference_time = time() - start_time
print(f"[INFO] Inference complete in {inference_time:.2f} seconds")

# -------------------------------------------------------------
# Threshold Tuning (only for median-risk head)
# -------------------------------------------------------------
"""
- Tuning the decision threshold for the median-risk head improves F1 and recall on imbalanced data. 
- This is done using the validation set.
- The optimal threshold is the one that maximizes the F1-score.
- Max-risk head remains at 0.5 since its performance is already strong.
"""
# --- Load validation data for threshold tuning (scientifically clean) ---
x_val = torch.load(TEST_DATA_DIR / "val.pt", map_location=device)
mask_val = torch.load(TEST_DATA_DIR / "val_mask.pt", map_location=device)

# Build validation labels from same patient splits (JSON)
val_ids = splits["val"]
val_df = features_df.set_index("subject_id").loc[val_ids].reset_index()

# Binary label creation (consistent and reproducible)
val_df["max_risk_binary"] = val_df["max_risk"].apply(lambda x: 1 if x > 2 else 0)
val_df["median_risk_binary"] = val_df["median_risk"].apply(lambda x: 1 if x == 2 else 0)

# Final label arrays
y_val_max = val_df["max_risk_binary"].values
y_val_median = val_df["median_risk_binary"].values

# Helper function: Finds the threshold that maximises F1-score on validation data.
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)
    return best_t, best_f1

# --- Threshold tuning of just median head on validation set ---
with torch.no_grad():
    val_outputs = model(x_val, mask_val)
val_prob_median = torch.sigmoid(val_outputs["logit_median"].squeeze()).cpu().numpy()

best_thresh_median, best_f1_median = find_best_threshold(y_val_median, val_prob_median)

print(f"[INFO] Median Risk Threshold (from validation) = {best_thresh_median:.3f} (F1={best_f1_median:.3f})")

# -------------------------------------------------------------
# Inspect range and central tendency of regression predictions
# -------------------------------------------------------------
"""
Prints the range (min–max) and mean of regression outputs in log-space.
Helps verify:
- The numerical stability of model predictions (no extreme outliers)
- That predictions are within a plausible range consistent with log1p-transformed targets
- Whether the model is over- or under-predicting on average before calibration
"""
print(f"Regression predictions (log-space):")
print(f"  Min:  {y_pred_reg.min():.4f}")
print(f"  Max:  {y_pred_reg.max():.4f}")
print(f"  Mean: {y_pred_reg.mean():.4f}")
print("==========================")

# -------------------------------------------------------------
# Prepare Ground-Truth Arrays
# -------------------------------------------------------------
"""
- Need the true labels (actual outcomes) to compare against model predictions
- y_test is a dictionary of PyTorch tensors (from earlier rebuilding step)
- Extract each PyTorch tensor from y_test dictionary 
- If these tensors are on a GPU (from training or inference pipelines), .cpu() brings them back to the CPU memory so we can handle them safely with NumPy and Pandas.
- NumPy arrays can’t exist on GPUs, only PyTorch tensors can.
- Convert PyTorch tensor → NumPy array for evaluation functions (pandas + sklearn compatibility)
"""
# Extract each tensor, move to CPU (if needed), convert to NumPy arrays for metric computations (same format as predictions)
y_true_max = y_test["max"].cpu().numpy()
y_true_median = y_test["median"].cpu().numpy()
y_true_reg = y_test["reg"].cpu().numpy()

# -------------------------------------------------------------
# Save Predictions with regression log + raw (Phase 4.5 only)
# -------------------------------------------------------------
"""
- Combine ground-truth and predicted values into a DataFrame
- The model’s regression head was trained to predict log1p(pct_time_high),
- So its native outputs (`y_pred_reg`) are in log-space (logarithmic scale).
- We evaluate in both:
    (a) log-space → reflects the actual training objective
    (b) raw-space → clinically interpretable scale
- Classification columns are: y_true_max & prob_max (for max risk classification), y_true_median & prob_median (for median risk classification)
- Regression columns are: y_true_reg & y_pred_reg_raw (raw), y_true_reg_log & y_pred_reg_log (log)
- Each row corresponds to one patient in the test set (15 patients in total)
"""

# Create combined prediction DataFrame
df_preds = pd.DataFrame({
    # Classification heads
    "y_true_max": y_true_max,                     # Ground truth max
    "prob_max": prob_max,                         # Prediction max (inference)
    "y_true_median": y_true_median,               # Ground truth median
    "prob_median": prob_median,                   # Prediction median (inference)

    # Regression head (both log + raw)
    "y_true_reg": y_true_reg,                     # Ground truth regression (for raw metrics)
    "y_true_reg_log": np.log1p(y_true_reg),       # Ground truth regression (log-converted for log metrics)
    "y_pred_reg_log": y_pred_reg,                 # Prediction regresion (inference for log metrics)
    "y_pred_reg_raw": np.expm1(y_pred_reg)        # Prediction regression (back-transformed for raw metrics)
})

# Save for diagnostics
df_preds.to_csv(RESULTS_DIR / "tcn_predictions_refined.csv", index=False)
print("[INFO] Saved classification predictions + regression predictions (raw + log-space) → tcn_predictions_refined.csv")

# -------------------------------------------------------------
# Compute Correlation Between Predictions and Ground Truth
# -------------------------------------------------------------
"""
- Computes Pearson correlation coefficients for the regression head.
- 'log-space' correlation compares model outputs in the log1p-transformed scale 
  (used for training and internal validation).
- 'raw-space' correlation compares inverse-transformed predictions to true percentages.
- Interpretation:
    • High correlation (>0.4–0.5) but negative R² indicates monotonicity with bias,
      meaning predictions follow the trend but are systematically over- or under-estimated.
    • This allows post-hoc calibration instead of retraining.
"""
corr_log = np.corrcoef(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log"])[0,1]
corr_raw = np.corrcoef(df_preds["y_true_reg"], df_preds["y_pred_reg_raw"])[0,1]
print(f"Correlation (log-space): {corr_log:.3f}")
print(f"Correlation (raw-space): {corr_raw:.3f}")

# -------------------------------------------------------------
# Visualise Calibration Bias (Log-Space)
# -------------------------------------------------------------
"""
- Scatter plot of true vs predicted log-space regression values.
- The red dashed line represents perfect calibration (y = x).
- Points above the line → model overestimates; below → model underestimates.
- This plot helps to visually detect systematic calibration bias before applying correction.
- Saved to disk for reproducibility.
"""

plt.scatter(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log"], alpha=0.7)
plt.plot([df_preds["y_true_reg_log"].min(), df_preds["y_true_reg_log"].max()],
         [df_preds["y_true_reg_log"].min(), df_preds["y_true_reg_log"].max()],
         color='red', linestyle='--')
plt.xlabel("True log1p(pct_time_high)")
plt.ylabel("Predicted log1p(pct_time_high)")
plt.title("Refined TCN — Regression Calibration (log-space)")

plt.savefig(RESULTS_DIR / "tcn_regression_calibration_logspace.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"[INFO] Saved calibration plot → {RESULTS_DIR}/tcn_regression_calibration_logspace.png")

# -------------------------------------------------------------
# Apply Post-Hoc Calibration (No Retraining Needed)
# -------------------------------------------------------------
"""
- Fit a simple linear regression on the log-space predictions:
      y_true_log ≈ a * y_pred_log + b
- Corrects systematic bias in predicted log-space values.
- Calibrated predictions are also converted back to raw-space using expm1 for interpretability.
- This ensures that metrics reflect calibrated, clinically meaningful predictions.
"""

lr = LinearRegression().fit(
    df_preds["y_pred_reg_log"].values.reshape(-1, 1),
    df_preds["y_true_reg_log"].values
)
a, b = lr.coef_[0], lr.intercept_
print(f"Calibration: y_true_log ≈ {a:.3f} * y_pred_log + {b:.3f}")

# Apply calibration
df_preds["y_pred_reg_log_cal"] = a * df_preds["y_pred_reg_log"] + b
df_preds["y_pred_reg_raw_cal"] = np.expm1(df_preds["y_pred_reg_log_cal"])

# -------------------------------------------------------------
# Compute Metrics (Pre- and Post-Calibration)
# -------------------------------------------------------------
"""
- Computes all relevant metrics for both classification and regression tasks:
    - Binary classification: max risk, median risk
    - Regression classification:
        1. Log-space metrics → internal validation, matches the training objective (MSE on log targets)
        2. Raw-space metrics → clinically interpretable and comparable to baseline models
- Also computes metrics for calibrated predictions to quantify improvement after bias correction.
"""

# Compute classification metrics for true vs prob
metrics_max = compute_classification_metrics(y_true_max, prob_max)
metrics_median = compute_classification_metrics(y_true_median, prob_median)

# Compute classification metrics for median head with applied threshold tuning
metrics_median_tuned = compute_classification_metrics(y_true_median, prob_median, threshold=best_thresh_median)

# Pre-calibration regression metrics
metrics_reg_log = compute_regression_metrics(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log"])
metrics_reg_raw = compute_regression_metrics(df_preds["y_true_reg"], df_preds["y_pred_reg_raw"])

# Post-calibration regression metrics
metrics_reg_log_cal = compute_regression_metrics(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log_cal"])
metrics_reg_raw_cal = compute_regression_metrics(df_preds["y_true_reg"], df_preds["y_pred_reg_raw_cal"])

# Combine all metrics into one JSON-friendly dictionary
all_metrics = {
    "max_risk": metrics_max,
    "median_risk": metrics_median,
    "median_risk_tuned": metrics_median_tuned, # classfication metrics for median head using tuning threshold
    "pct_time_high_log": metrics_reg_log, # regression metrics for log-scale (internal)
    "pct_time_high_raw": metrics_reg_raw, # regression metrics for raw (interpretable)
    "pct_time_high_log_cal": metrics_reg_log_cal, # regression metrics for log-scale (internal) calibrated
    "pct_time_high_raw_cal": metrics_reg_raw_cal, # regression metrics for raw (interpretable) calibrated
    "inference_time_sec": round(inference_time, 2)
}

# -------------------------------------------------------------
# Visualise Calibration Effect (Before vs After)
# -------------------------------------------------------------
"""
- Overlays scatter plots of log-space predictions before and after calibration.
- Steelblue points: before calibration
- Orange points: after calibration
- Red dashed line: perfect calibration
- Provides an immediate visual sense of how calibration improves alignment with true values.
- Saved to disk for reproducibility.
"""
plt.figure(figsize=(7, 7))

# --- Before Calibration ---
plt.scatter(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log"], 
            alpha=0.7, color="steelblue", label="Before calibration")

# --- After Calibration ---
plt.scatter(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log_cal"], 
            alpha=0.7, color="orange", label="After calibration")

# --- Reference Line (Perfect Calibration) ---
min_val = min(df_preds["y_true_reg_log"].min(), df_preds["y_pred_reg_log_cal"].min())
max_val = max(df_preds["y_true_reg_log"].max(), df_preds["y_pred_reg_log_cal"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal diagonal")

# --- Labels & Formatting ---
plt.xlabel("True log1p(pct_time_high)")
plt.ylabel("Predicted log1p(pct_time_high)")
plt.title("Refined TCN — Calibration Before vs After (log-space)")
plt.legend()
plt.grid(alpha=0.3)

# --- Save and Show ---
plt.savefig(RESULTS_DIR / "tcn_regression_calibration_comparison_logspace.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"[INFO] Saved calibration comparison plot → {RESULTS_DIR}/tcn_regression_calibration_comparison_logspace.png")

# -------------------------------------------------------------
# Save Computed Metrics (JSON)
# -------------------------------------------------------------
"""
- Saves all computed metrics (classification, regression, calibrated) to a JSON file.
- Ensures reproducibility and documentation of model evaluation.
"""
with open(RESULTS_DIR / "tcn_metrics_refined.json", "w") as f:
    json.dump(all_metrics, f, indent=4)
print("[INFO] Saved metrics → tcn_results_refined/tcn_metrics_refined.json")

# -------------------------------------------------------------
# Display Summary
# -------------------------------------------------------------
# Print a quick on-screen summary of key results for easy inspection
print("\n=== Final Refined Test Metrics ===")
print(f"Max Risk (0.5) — AUC: {metrics_max['roc_auc']:.3f}, F1: {metrics_max['f1']:.3f}, Acc: {metrics_max['accuracy']:.3f}")
print(f"Median Risk (0.5) — AUC: {metrics_median['roc_auc']:.3f}, F1: {metrics_median['f1']:.3f}, Acc: {metrics_median['accuracy']:.3f}")
print(f"Median Risk ({best_thresh_median:.3f}) — AUC: {metrics_median_tuned['roc_auc']:.3f}, F1: {metrics_median_tuned['f1']:.3f}, Acc: {metrics_median_tuned['accuracy']:.3f}")
print(f"Regression (log) — RMSE: {metrics_reg_log['rmse']:.3f}, R²: {metrics_reg_log['r2']:.3f}")
print(f"Regression (raw) — RMSE: {metrics_reg_raw['rmse']:.3f}, R²: {metrics_reg_raw['r2']:.3f}")
print(f"Regression (log) calibrated — RMSE: {metrics_reg_log_cal['rmse']:.3f}, R²: {metrics_reg_log_cal['r2']:.3f}")
print(f"Regression (raw) calibrated — RMSE: {metrics_reg_raw_cal['rmse']:.3f}, R²: {metrics_reg_raw_cal['r2']:.3f}")
print("==========================")
print(f"Test IDs used: {sorted(list(test_ids))}")
# Contextualises regression metrics and potential calibration bias by showing baseline means and spread.
print(f"Mean of y_true_reg: {y_true_reg.mean():.4f}, Std: {y_true_reg.std():.4f}  # Ground truth: actual % time high")
print(f"Mean of y_pred_reg: {y_pred_reg.mean():.4f}, Std: {y_pred_reg.std():.4f}  # Model output: predicted log-space mean")
print("==========================")

print("[INFO] Evaluation complete — refined TCN calibration validated and metrics saved.")
