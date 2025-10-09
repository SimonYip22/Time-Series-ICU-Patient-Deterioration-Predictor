"""
evaluate_tcn_testset.py

Title: Final Evaluation of Temporal Convolutional Network (TCN) on Held-Out Test Set

Summary:
- Loads the best-performing TCN checkpoint (tcn_best.pt) from Phase 4.
- Dynamically rebuilds the test targets (y_test) from patient-level CSV + split JSON.
- Runs inference on the unseen test set (x_test, mask_test).
- Computes classification + regression metrics via evaluation_metrics.py.
- Saves patient-level predictions (CSV) and aggregated metrics (JSON).
- Provides final, reproducible test-time performance for comparison vs. NEWS2 and LightGBM.
Output:
- results/tcn_metrics.json → summary of computed numeric performance metrics
- results/tcn_predictions.csv → row-wise per-patient outputs (predictions + ground truth) 
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import sys                            # System-level operations
from pathlib import Path              # Safe filesystem handling across OS

import torch                          # Core deep learning library
import json                           # For reading/writing JSON files (splits + metrics), saving metrics to JSON
import pandas as pd                   # For tabular prediction outputs
from time import time                 # Used to measure inference duration


# --- Add src/ to sys.path so we can import packages directly ---
SCRIPT_DIR = Path(__file__).resolve().parent               # src/prediction_evaluations
SRC_DIR = SCRIPT_DIR.parent                                 # src/
PROJECT_ROOT = SRC_DIR.parent                               # project root
sys.path.append(str(SRC_DIR))                               # now Python can find ml_models_tcn, prediction_evaluations

# --- Import reusable metric utilities (Phase 5 Step 0) ---
# evaluation_metrics.py defines functions to compute metrics for classification and regression tasks.
from prediction_evaluations.evaluation_metrics import (
    compute_classification_metrics,
    compute_regression_metrics
)

# --- Import the trained TCN model definition from Phase 4 (from tcn_model.py) ---
# tcn_models.py efines the architecture: all layers, residual blocks, pooling, and heads.
from ml_models_tcn.tcn_model import TCNModel

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
# Path to best TCN model checkpoint (saved during early stopping in Phase 4)
TRAINED_MODEL_PATH = SRC_DIR / "ml_models_tcn" / "trained_models" / "tcn_best.pt"

# Path to saved test tensors (input sequences + masks)
TEST_DATA_DIR = SRC_DIR / "ml_models_tcn" / "prepared_datasets"

# Path to rebuild y_test (patient-level labels for test set)
FEATURES_PATIENT_PATH = PROJECT_ROOT / "data" / "processed_data" / "news2_features_patient.csv"
SPLITS_PATH = SRC_DIR / "ml_models_tcn" / "deployment_models" / "preprocessing" / "patient_splits.json"

# Directory to save evaluation outputs (metrics + predictions)
RESULTS_DIR = SRC_DIR / "prediction_evaluations" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Dynamically rebuild y_test from patient-level data (same logic as phase 4 training)
# -------------------------------------------------------------
# Dynamically rebuild y_test from CSV + JSON split (handles any class distribution)
# Load patient-level feature dataframe and test split identifiers
features_df = pd.read_csv(FEATURES_PATIENT_PATH)
with open(SPLITS_PATH) as f:
    splits = json.load(f)
test_ids = set(splits["test"]) # Extract test patient IDs

# Filter dataframe to only include patients in test split
test_df = features_df[features_df["subject_id"].isin(test_ids)].reset_index(drop=True)

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

# --- 2. Create the model architecture / structure (tcn_model.py) ---
# Automatically get feature dimension (171 features = input channels in first convolutional layer)
NUM_FEATURES = x_test.shape[2] 
# Instantiate the TCN model architecture defined in tcn_model.py 
# Must match exactly what was used during training, otherwise the weight shapes won't align
model = TCNModel(
    num_features=NUM_FEATURES, # Input feature dimension (171 per-timestep features)
    num_channels=[64, 128, 128], # 3 TCN layers with increasing channels
    kernel_size=3,            # Kernel size of 3 for temporal convolutions
    dropout=0.2,            # Regularisation: randomly zero 20% of activations during training
    head_hidden=64          # Hidden layer size of 64 in the final dense head
)

# --- 3. Load the trained model weights (tcn_best.pt) ---
# tcn_best.pt = serialized 'state_dict' containing all trained layer weights + biases (tensors)
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

print("[INFO] Loaded TCN model and weights successfully")

# -------------------------------------------------------------
# Inference (Testing)
# -------------------------------------------------------------
"""
- The trained model is used to make predictions on new, unseen data.
- When we train a model, PyTorch tracks everything needed to compute gradients for backpropagation (e.g. how each weight affects the loss).
- When we test / evaluate (inference) a model, we don’t need gradients, we just want predictions.
- No optimisation, no gradient updates, no backpropagation.
- Disabling gradient computation speeds up inference and reduces memory consumption.
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
# Compute Metrics
# -------------------------------------------------------------
"""
- Use the reusable functions from evaluation_metrics.py to compute all metrics
- Functions take in both prediction and ground-truth arrays
- Compute standard metrics for classification and regression tasks
- The output is a dictionary of metric numbers that quantify how close the model’s predictions were to the ground truth.
"""
metrics_max = compute_classification_metrics(y_true_max, prob_max)
metrics_median = compute_classification_metrics(y_true_median, prob_median)
metrics_reg = compute_regression_metrics(y_true_reg, y_pred_reg)

# Combine all metrics into one JSON-friendly dictionary
all_metrics = {
    "max_risk": metrics_max,
    "median_risk": metrics_median,
    "pct_time_high": metrics_reg,
    "inference_time_sec": round(inference_time, 2)
}

# -------------------------------------------------------------
# Save Computed Metrics (JSON)
# -------------------------------------------------------------
# Write computed metrics to disk for documentation/reproducibility
with open(RESULTS_DIR / "tcn_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)
print("[INFO] Saved metrics → results/tcn_metrics.json")

# -------------------------------------------------------------
# Save Predictions and Ground-Truth Arrays (CSV)
# -------------------------------------------------------------
# Combine ground-truth and predicted values into a DataFrame
# Columns are: y_true_max & prob_max (for max risk classification), y_true_median & prob_median (for median risk classification), y_true_reg & y_pred_reg (for regression)
# Each row corresponds to one patient in the test set (15 patients in total)
df_preds = pd.DataFrame({
    "y_true_max": y_true_max,
    "prob_max": prob_max,
    "y_true_median": y_true_median,
    "prob_median": prob_median,
    "y_true_reg": y_true_reg,
    "y_pred_reg": y_pred_reg
})

# Save predictions for further analysis and reproducibility
df_preds.to_csv(RESULTS_DIR / "tcn_predictions.csv", index=False)
print("[INFO] Saved predictions → results/tcn_predictions.csv")

# -------------------------------------------------------------
# Display Summary
# -------------------------------------------------------------
# Print a quick on-screen summary of key results for easy inspection
print("\n=== Final Test Metrics ===")
print(f"Max Risk — AUC: {metrics_max['roc_auc']:.3f}, F1: {metrics_max['f1']:.3f}, Acc: {metrics_max['accuracy']:.3f}")
print(f"Median Risk — AUC: {metrics_median['roc_auc']:.3f}, F1: {metrics_median['f1']:.3f}, Acc: {metrics_median['accuracy']:.3f}")
print(f"Regression — RMSE: {metrics_reg['rmse']:.3f}, R²: {metrics_reg['r2']:.3f}")
print("==========================")