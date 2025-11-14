# src/scripts_inference/unified_inference.py

"""
unified_inference.py

Title: Unified Inference & Interpretability Pipeline for LightGBM and TCN Models

Purpose:
- Provides batch and per-patient inference for patient-level LightGBM and timestamp-level TCN models.
- Performs predictions on a test set (batch) or for a single patient (interactive CLI).
- Lightweight, deployment-ready script focusing on reproducibility; streaming or incremental inference requires extra handling.

Workflow:
1. Load patient-level features (LightGBM input).
2. Load LightGBM models and perform predictions.
3. Load TCN tensors, model architecture/config, and perform timestamp-level inference.
4. Save per-model prediction outputs to CSV.
5. Run interpretability:
   - LightGBM: SHAP-based top-10 feature importance.
   - TCN: Gradient × Input saliency top-10 features.
6. Combine interpretability summaries for convenience.
7. Optional CLI loop for single-patient predictions.

Outputs:
1. Per-model CSVs of predictions for all targets (classification probabilities and regression outputs).
2. Top-10 feature importance per model (LightGBM SHAP, TCN saliency).
3. Combined top-10 feature summary CSV for both models.
4. Optional terminal output for single-patient predictions.

Notes:
- Script repurposes evaluation and interpretability code; no new model logic is introduced.
- Single-patient inference uses the same preprocessed data and feature mapping as batch inference.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
# --- Standard Libraries ---
import json
import sys
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
PADDING_CONFIG_PATH = SRC_DIR / "ml_models_tcn" / "deployment_models" / "preprocessing" / "padding_config.json"
TEST_TENSORS_DIR = SRC_DIR / "ml_models_tcn" / "prepared_datasets"

# Model paths
LIGHTGBM_MODELS_DIR = SRC_DIR / "prediction_evaluations" / "lightgbm_results"
TCN_MODEL_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "tcn_best_refined.pt"
TCN_CONFIG_PATH = SRC_DIR / "prediction_diagnostics" / "trained_models_refined" / "config_refined.json"

# Output path
OUTPUT_DIR = SRC_DIR / "scripts_inference" / "deployment_lite_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Load Test Data and Patient Features for LightGBM
# -------------------------------------------------------------
# Load test patient IDs from JSON splits file
with open(SPLITS_PATH) as f:
    splits = json.load(f)
test_ids = splits["test"]

# Load patient-level features CSV and subset to test patients
features_df = pd.read_csv(DATA_PATH)
test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()

# Define features to exclude from model input
exclude_cols = ["subject_id", "max_risk", "median_risk", "pct_time_high"]

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

# Clip negative regression predictions to 0
df_lightgbm["y_pred_reg"] = df_lightgbm["y_pred_reg"].clip(lower=0)

# Save LightGBM inference outputs to CSV
df_lightgbm.to_csv(OUTPUT_DIR / "lightgbm_inference_outputs.csv", index=False)
print("[INFO] Saved LightGBM inference outputs → lightgbm_inference_outputs.csv")

# -------------------------------------------------------------
# 3. TCN Inference
# -------------------------------------------------------------
print("[INFO] Starting TCN inference...")

# Load TCNModel from tcn_model.pyw
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))
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

# Clip negative values
reg_raw = np.clip(reg_raw, a_min=0, a_max=None)

# Report elapsed time for TCN inference
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
print("[INFO] Saved TCN inference outputs → tcn_inference_outputs.csv")

# Print Summary to Terminal
print("\n=== Deployment Lite Inference Completed ===")
print(f"LightGBM predictions: {df_lightgbm.shape[0]} patients")
print(f"TCN predictions:      {df_tcn.shape[0]} patients")
print(f"Outputs saved in 'deployment_lite_outputs/' folder")
print("===========================================")

# -------------------------------------------------------------
# 4. LightGBM Interpretability - SHAP (Top 10 features)
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

# -------------------------------------------------------------
# 5. TCN Interpretability - Gradient x Input Saliency (Top 10 Features)
# -------------------------------------------------------------
# Load padding configuration to get temporal feature names
with open(PADDING_CONFIG_PATH) as f:
    padding_config = json.load(f)
tcn_feature_names = padding_config["feature_cols"]

# Define targets and corresponding model output keys for saliency computation
tcn_targets = [
    ("max_risk", "logit_max"),
    ("median_risk", "logit_median"),
    ("pct_time_high", "regression")
]

def compute_tcn_saliency_top10(model, x_tensor, mask_tensor, feature_cols, targets, device):
    """
    Compute the top-10 most important features per output head of a TCN model
    using the |gradient × input| saliency method.

    Parameters:
    - model (torch.nn.Module): Trained TCN model.
    - x_tensor (torch.Tensor): Input time-series tensor of shape (n_samples, seq_len, n_features).
    - mask_tensor (torch.Tensor): Mask tensor indicating valid timesteps (same shape as x_tensor[:, :, 0]).
    - feature_cols (list of str): Names of features corresponding to tensor indices.
    - targets (list of tuples): Each tuple is (target_name, model_output_key) specifying
      which output heads to compute saliency for.
    - device (torch.device): PyTorch device to perform computation on.

    Returns:
    - pandas.DataFrame: Concatenated top-10 features per target with columns:
        'feature', 'mean_abs_saliency', 'target'.
    """
    results = []
    batch_size = 4

    # Number of test samples
    n_test = x_tensor.shape[0]

    # Iterate over each target head
    for target_name, head_key in targets:
        print(f"[INFO] Computing TCN saliency for {target_name}...")
        all_saliencies = []

        # Loop through batches
        for i in range(0, n_test, batch_size):

            # Prepare batch data
            xb = x_tensor[i:i+batch_size].clone().detach().to(device)
            mb = mask_tensor[i:i+batch_size].to(device)

            # Enable gradient tracking
            xb.requires_grad = True
            # Forward pass
            outputs = model(xb, mb)
            # Extract output for the current head
            out = outputs[head_key].squeeze()

            grads = []
            # Compute gradients for each sample in the batch
            for j in range(out.shape[0]):
                if xb.grad is not None:
                    xb.grad.zero_()
                # Backward pass for the j-th sample
                out[j].backward(retain_graph=True)
                # Store gradients
                grads.append(xb.grad[j].detach().cpu().numpy())

            # Stack gradients and compute |grad × input|
            grads = np.stack(grads, axis=0)
            sal = np.abs(grads * xb.detach().cpu().numpy())
            # Append batch saliencies
            all_saliencies.append(sal)

        # Concatenate all batches
        all_saliencies = np.concatenate(all_saliencies, axis=0)

        # Average over patients and timesteps
        feature_mean = all_saliencies.mean(axis=(0, 1))

        # Build dataframe
        df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_saliency": feature_mean
        }).sort_values("mean_abs_saliency", ascending=False).head(10)

        # Add target name column
        df["target"] = target_name
        results.append(df)

    return pd.concat(results, ignore_index=True)

# Run TCN saliency summary using loaded model and test tensors
print("[INFO] Computing Gradient×Input Saliency Top-10 features (TCN)...")
tcn_top10 = compute_tcn_saliency_top10(model, x_test, mask_test, tcn_feature_names, tcn_targets, device)

# -------------------------------------------------------------
# 6. Merge Feature Summaries
# -------------------------------------------------------------
# Combine LightGBM and TCN top 10 feature summaries into one file for convenience
combined_summary = pd.concat([
    lightgbm_top10.assign(model="LightGBM"),
    tcn_top10.assign(model="TCN")
])
combined_summary.to_csv(OUTPUT_DIR / "top10_features_summary.csv", index=False)
print("[INFO] ✅ Combined interpretability summary saved → top10_features_summary.csv")

# -------------------------------------------------------------
# 7. Interactive CLI: Single-Patient Inference
# -------------------------------------------------------------
"""
- Provides a command-line interface to view per-patient predictions.
- Users can enter a valid patient ID to display LightGBM and TCN predictions for that patient. 
- The loop continues until the user exits.
"""
def run_single_patient_inference(patient_id):
    """
    Display LightGBM and TCN predictions for a single patient.

    Parameters:
    - patient_id (int or str): The ID of the patient to compute predictions for.

    Behavior:
    - Validates the patient ID against the test set.
    - Prints predicted probabilities for classification targets (LightGBM and TCN).
    - Prints regression output (LightGBM and TCN).
    - Alerts if the patient ID is invalid.
    """
    # Check if patient ID is valid
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

if __name__ == "__main__":

    print("\nBatch inference complete.")

    # Display available patient IDs for reference
    print("Available patient IDs for per-patient inference:")
    print(test_ids)

    # Simple interactive loop for single-patient inference
    while True:
        user_input = input("\nEnter a patient ID for per-patient inference (or 'no' to exit): ").strip()

        if user_input.lower() == "no":
            print("Exiting per-patient inference.")
            break

        if not user_input.isdigit():
            print("Please enter a numeric patient ID.")
            continue

        patient_id = int(user_input)

        if patient_id not in test_ids:
            print(f"Patient ID {patient_id} is not in the test set: {test_ids}")
            continue

        run_single_patient_inference(patient_id)
