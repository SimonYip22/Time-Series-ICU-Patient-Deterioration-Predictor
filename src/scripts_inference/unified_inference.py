# src/scripts_inference/unified_inference.py

"""
run_inference.py

Title: Unified Inference & Interpretability Pipeline for LightGBM and TCN Models

Purpose:
- Lightweight deployment-ready inference script for both patient-level LightGBM and time-series TCN models.
- Performs batch inference on a test set (default) or single-patient inference when a patient ID is specified.
- Note on single-batch inference:
    - This script performs inference on the entire test set as a single batch.
    - For patient-level or streaming inference, additional looping and state management would be required.
    - This lightweight deployment script focuses on batch inference for reproducibility and simplicity.
- Outputs:
    1. CSVs of predictions for all targets (classification probabilities and regression outputs).
    2. Aggregated top-10 feature importances per model:
        - LightGBM: SHAP-based top features
        - TCN: Gradient × Input saliency-based top features
    3. Combined top-10 feature summary for convenience.
- Includes optional per-patient output to terminal for demonstration or single-case prediction.

Design:
1. Load patient-level test data and LightGBM models.
2. Load time-series test tensors and TCN model.
3. Perform batch inference and save results to CSV.
4. Compute interpretability summaries (top features).
5. Optionally, run single-patient inference and display results in terminal.

Notes:
- This script repurposes previous evaluation and interpretability scripts; no new model logic is introduced.
- Single-patient inference uses the same preprocessed test data and feature mapping.
"""


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
# --- Standard Libraries ---
import json
from pathlib import Path
from time import time

# --- Data Handling ---
import pandas as pd
import numpy as np

# --- ML Libraries ---
import joblib
import torch
import shap

# -------------------------------------------------------------
# Paths and setup
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Input data paths
DATA_PATH = PROJECT_ROOT / "data" / "processed_data" / "news2_features_patient.csv"
SPLITS_PATH = SRC_DIR / "ml_models_tcn" / "deployment_models" / "preprocessing" / "patient_splits.json"
TEST_TENSORS_DIR = SRC_DIR / "ml_models_tcn" / "prepared_datasets"

# Model paths
LIGHTGBM_MODELS_DIR = SRC_DIR / "prediction_evaluations" / "lightgbm_results"
TCN_MODEL_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "tcn_best_refined.pt"
TCN_CONFIG_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "config_refined.json"

# Output path
OUTPUT_DIR = SRC_DIR / "scripts_inference" / "deployment_lite_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Load Shared Resources & Display Valid Patient IDs
# -------------------------------------------------------------
# Load test patient IDs from JSON splits file
with open(SPLITS_PATH) as f:
    splits = json.load(f)
test_ids = splits["test"]

# Display valid test patient IDs for reference when running per-patient inference
print("\nValid test patient IDs:")
print(test_ids)
print("\nUse one of these IDs to run per-patient inference.")

# Load patient-level features CSV and subset to test patients
features_df = pd.read_csv(DATA_PATH)
test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()

# Recreate binary targets for LightGBM consistency with training
test_df["max_risk_binary"] = test_df["max_risk"].apply(lambda x: 1 if x > 2 else 0)
# Binary creation now includes NEWS2=3 as different test set may now include this value
test_df["median_risk_binary"] = test_df["median_risk"].apply(lambda x: 1 if x >= 2 else 0)

# Define features to exclude from model input
exclude_cols = ["subject_id", "max_risk", "median_risk", "pct_time_high",
                "max_risk_binary", "median_risk_binary"]
# Define feature columns for LightGBM inference
feature_cols = [c for c in test_df.columns if c not in exclude_cols]

# -------------------------------------------------------------
# 2. LightGBM Inference
# -------------------------------------------------------------
print("[INFO] Starting LightGBM inference...")

# Define targets and their task types
targets = [
    ("max_risk", "classification"),
    ("median_risk", "classification"),
    ("pct_time_high", "regression")
]

# Dictionary to hold LightGBM predictions
lightgbm_preds = {} 
# Dictionary to hold loaded LightGBM models for interpretability
models_dict = {}  

# Iterate over each target to load model and perform inference
for target, task in targets:
    model_file = LIGHTGBM_MODELS_DIR / f"{target}_retrained_model.pkl"
    assert model_file.exists(), f"Missing model: {model_file}"
    model = joblib.load(model_file)
    models_dict[target] = model  # Save model for SHAP interpretability

    # Prepare test features
    X_test = test_df[feature_cols] 

    # Perform prediction: probability for classification, direct prediction for regression
    if task == "classification":
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)
    lightgbm_preds[target] = preds

# Aggregate LightGBM predictions into a DataFrame
df_lightgbm = pd.DataFrame({
    "subject_id": test_df["subject_id"],
    "prob_max": lightgbm_preds["max_risk"],
    "prob_median": lightgbm_preds["median_risk"],
    "y_pred_reg": lightgbm_preds["pct_time_high"]
})

# Save LightGBM inference outputs to CSV
df_lightgbm.to_csv(OUTPUT_DIR / "lightgbm_inference_outputs.csv", index=False)
print(f"[INFO] Saved LightGBM inference outputs → {OUTPUT_DIR/'lightgbm_inference_outputs.csv'}")

# -------------------------------------------------------------
# 3. TCN Inference
# -------------------------------------------------------------
print("[INFO] Starting TCN inference...")

# Load TCNModel from tcn_model.py
from ml_models_tcn.tcn_model import TCNModel

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test tensors for time-series data and corresponding masks
x_test = torch.load(TEST_TENSORS_DIR / "test.pt", map_location=device)
mask_test = torch.load(TEST_TENSORS_DIR / "test_mask.pt", map_location=device)

# Sanity check: ensure tensor batch size matches number of test patient IDs
assert x_test.shape[0] == len(test_ids), "Mismatch between tensor batch size and patient IDs."

# Load TCN model configuration JSON
with open(TCN_CONFIG_PATH) as f:
    config = json.load(f)
arch = config["model_architecture"]
NUM_FEATURES = x_test.shape[2] # Number of input features for the TCN

# Instantiate TCN model with loaded configuration parameters
model = TCNModel(
    num_features=NUM_FEATURES,
    num_channels=arch["num_channels"],
    kernel_size=arch["kernel_size"],
    dropout=arch["dropout"],
    head_hidden=arch["head_hidden"]
)

# Load trained TCN model weights
state_dict = torch.load(TCN_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict) 
model.to(device)

# Set model to evaluation mode
model.eval() 

# Perform inference with no gradient computation
start = time()
with torch.no_grad():
    outputs = model(x_test, mask_test) 

# Extract logits and regression outputs, move to CPU and numpy for processing
logits_max = outputs["logit_max"].squeeze().cpu()
logits_median = outputs["logit_median"].squeeze().cpu()
regression = outputs["regression"].squeeze().cpu()

# Convert logits to probabilities via sigmoid
prob_max = torch.sigmoid(logits_max).numpy()
prob_median = torch.sigmoid(logits_median).numpy()

# Convert regression outputs back from log scale (expm1)
reg_raw = np.expm1(regression.numpy())
elapsed = time() - start
print(f"[INFO] TCN inference completed in {elapsed:.2f}s")

# Aggregate TCN predictions into DataFrame
df_tcn = pd.DataFrame({
    "subject_id": test_df["subject_id"],
    "prob_max": prob_max,
    "prob_median": prob_median,
    "y_pred_reg_raw": reg_raw
})

# Save TCN inference outputs to CSV
df_tcn.to_csv(OUTPUT_DIR / "tcn_inference_outputs.csv", index=False)
print(f"[INFO] Saved TCN inference outputs → {OUTPUT_DIR/'tcn_inference_outputs.csv'}")

# -------------------------------------------------------------
# 4. Combined Summary
# -------------------------------------------------------------
print("\n=== Deployment Lite Inference Completed ===")
print(f"LightGBM predictions: {df_lightgbm.shape[0]} patients")
print(f"TCN predictions:      {df_tcn.shape[0]} patients")
print(f"Output directory:     {OUTPUT_DIR}")
print("===========================================")

# -------------------------------------------------------------
# 5. LightGBM Interpretability - SHAP (Top 10 features)
# -------------------------------------------------------------
def compute_lightgbm_shap_top10(models_dict, X_input):
    """
    Computes mean absolute SHAP values per feature for each LightGBM model.
    Returns a concatenated DataFrame of top 10 features per target.

    Parameters:
    - models_dict: dict of {target_name: trained LightGBM model}
    - X_input: DataFrame of input features for SHAP computation

    Returns:
    - DataFrame with columns: feature, mean_abs_shap, target
    """
    # Initialise list to hold results
    results = []
    # Iterate over each model to compute SHAP values
    for target, model in models_dict.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        # For binary classification, shap_values is a list; take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # Compute mean absolute SHAP values per feature
        mean_abs = np.abs(shap_values).mean(axis=0)

        # Create DataFrame of top 10 features
        df = pd.DataFrame({
            "feature": X_input.columns,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).head(10)
        df["target"] = target
        results.append(df)
    return pd.concat(results, ignore_index=True)

# Run SHAP summary for LightGBM using loaded models and test features
print("[INFO] Computing SHAP Top-10 features (LightGBM)...")
lightgbm_top10 = compute_lightgbm_shap_top10(models_dict, test_df[feature_cols])
# Save LightGBM top-10 features to CSV
lightgbm_top10.to_csv(OUTPUT_DIR / "top10_features_lightgbm.csv", index=False)
print(f"[INFO] Saved LightGBM top-10 features → {OUTPUT_DIR/'top10_features_lightgbm.csv'}")

# -------------------------------------------------------------
# 6. TCN Interpretability - Gradient x Input Saliency (Top 10 Features)
# -------------------------------------------------------------

# Define feature columns for TCN saliency analysis (same as LightGBM features here for consistency)
feature_cols_tcn = feature_cols.copy()

# Define targets and corresponding model output keys for saliency computation
tcn_targets = [
    ("max_risk", "logit_max"),
    ("median_risk", "logit_median"),
    ("pct_time_high", "regression")
]

def compute_tcn_saliency_top10(model, x_tensor, mask_tensor, feature_cols, targets, device):
    """
    Computes |grad × input| saliency for each model head, averages across
    patients and timesteps, and returns top 10 features per target.

    Parameters:
    - model: trained TCN model
    - x_tensor: input tensor for test set (batch_size, seq_len, features)
    - mask_tensor: mask tensor indicating valid timesteps
    - feature_cols: list of feature names corresponding to input features
    - targets: list of tuples (target_name, model_output_key)
    - device: torch device

    Returns:
    - DataFrame with top 10 features per target based on mean absolute saliency
    """
    def grad_input_saliency(x_batch, mask_batch, head_key):
        """
        Computes gradient × input saliency for a batch and a specific output head.
        """
        # Enable gradient tracking
        x = x_batch.clone().detach().to(device)
        x.requires_grad = True
        # Forward pass
        outputs = model(x, mask_batch)
        # Select the relevant output head
        out = outputs[head_key].squeeze() 
        
        # Compute gradients
        grads = []

        # Iterate over batch to compute gradients for each sample
        for i in range(out.shape[0]):
            if x.grad is not None:
                x.grad.zero_()
            # Backward pass for the i-th output
            out[i].backward(retain_graph=True)
            # Store gradient for the i-th sample
            grads.append(x.grad[i].detach().cpu().numpy())
        # Stack gradients and compute |grad × input|
        grads = np.stack(grads, axis=0)
        saliency = np.abs(grads * x.detach().cpu().numpy())
        return saliency

    results = []
    batch_size = 4  # Process in small batches to manage memory during gradient computation
    n_test = x_tensor.shape[0] 

    # Iterate over each target to compete saliency
    for target_name, head_key in targets:
        print(f"[INFO] Computing TCN saliency for {target_name}...")
        all_saliencies = []
        # Batch processing
        for i in range(0, n_test, batch_size):
            xb = x_tensor[i:i+batch_size].to(device)
            mb = mask_tensor[i:i+batch_size].to(device)
            # Compute saliency for the batch
            sal_b = grad_input_saliency(xb, mb, head_key)
            all_saliencies.append(sal_b)
        all_saliencies = np.concatenate(all_saliencies, axis=0)
        # Average saliency across patients and timesteps per feature
        feature_mean = all_saliencies.mean(axis=(0, 1))
        # Create DataFrame of top 10 features
        df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_saliency": feature_mean
        }).sort_values("mean_abs_saliency", ascending=False).head(10)
        df["target"] = target_name
        results.append(df)
    return pd.concat(results, ignore_index=True)

# Run TCN saliency summary using loaded model and test tensors
print("[INFO] Computing Gradient×Input Saliency Top-10 features (TCN)...")
tcn_top10 = compute_tcn_saliency_top10(model, x_test, mask_test, feature_cols_tcn, tcn_targets, device)
tcn_top10.to_csv(OUTPUT_DIR / "top10_features_tcn.csv", index=False)
print(f"[INFO] Saved TCN top-10 features → {OUTPUT_DIR/'top10_features_tcn.csv'}")

# -------------------------------------------------------------
# 7. Merge Feature Summaries
# -------------------------------------------------------------
# Combine LightGBM and TCN top 10 feature summaries into one file for convenience
combined_summary = pd.concat([
    lightgbm_top10.assign(model="LightGBM"),
    tcn_top10.assign(model="TCN")
])
combined_summary.to_csv(OUTPUT_DIR / "top10_features_summary.csv", index=False)

print(f"[INFO] ✅ Combined interpretability summary saved → {OUTPUT_DIR/'top10_features_summary.csv'}")

# -------------------------------------------------------------
# 8. Single-Patient Inference
# -------------------------------------------------------------
def run_single_patient_inference(patient_id):
    """
    Prints LightGBM and TCN predictions for a single patient in the terminal.
    """
    if patient_id not in test_df["subject_id"].values:
        print(f"[ERROR] Patient ID {patient_id} not in test set.")
        return

    # Find index of the patient in the test DataFrame
    idx = test_df.index[test_df["subject_id"] == patient_id][0]

    # LightGBM predictions
    print(f"\n--- LightGBM predictions for patient {patient_id} ---")
    for target, task in targets:
        val = lightgbm_preds[target][idx]
        print(f"{target}: {val:.4f}")

    # TCN predictions
    print(f"\n--- TCN predictions for patient {patient_id} ---")
    print(f"prob_max: {prob_max[idx]:.4f}")
    print(f"prob_median: {prob_median[idx]:.4f}")
    print(f"y_pred_reg_raw: {reg_raw[idx]:.4f}")
