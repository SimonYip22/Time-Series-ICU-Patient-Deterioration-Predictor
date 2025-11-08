# src/results_finalisation/saliency_analysis_tcn.py
"""
saliency_analysis_tcn.py

Title: Compute temporal saliency maps for the refined TCN model on test set.

Summary
Purpose:
- Computes temporal saliency maps for the refined TCN model on the held-out test set (15 patients)  
- This provides model interpretability by quantifying when and which input features most strongly influenced each prediction target.
- This step (Phase 6 Step 4) extends interpretability beyond feature-level influence (SHAP in LightGBM) to temporal reasoning in the TCN.  
- It enables clinically meaningful visualisation of when deterioration-relevant signals (e.g. SpO₂, respiratory rate) drive predictions.
- Strengthens model transparency and trustworthiness for clinical AI evaluation.
Concept:
- Uses gradient × input (|grad * input|) saliency mapping:
  - For each patient sequence, computes the gradient of the model output with respect to every input feature at every timestep.
  - The magnitude of this value represents feature importance over time.
- Complements the LightGBM SHAP analysis (Phase 6 Step 3) by adding temporal interpretability for the sequence model.
Inputs:
- Trained model checkpoint: `tcn_best_refined.pt`
- Model configuration: `config_refined.json`
- Preprocessed test tensors: `test.pt`, `test_mask.pt`
- Preprocessing metadata: `padding_config.json`, `standard_scaler.pkl`, `patient_splits.json`
Outputs:
- All outputs saved to: `src/results_finalisation/interpretability_tcn/`
- Per Target (max_risk, median_risk, pct_time_high):
    1. Per-patient saliency arrays (.npz) → `patient_saliency_{target}.npz`  
        - Keys: `"patient_{id}"`  
        - Values: saliency matrix (max_seq_len × n_features)
    2. Per-patient saliency heatmaps (.png) → `{target}_patient_{i:02d}_heatmap.png`  
        - Shows feature importance over time for each test patient.
    3. Global mean saliency heatmap (.png) → `{target}_mean_heatmap.png`  
        - Average |grad × input| importance across all patients.
    4. Top-10 feature ranking (.csv) → `{target}_top10_saliency.csv`  
        - Mean absolute saliency aggregated across all timepoints and patients.
"""
# -----------------------
# Imports
# -----------------------

import json
from pathlib import Path
import numpy as np

# Core deep learning library (model loading, tensor ops)
import torch
import matplotlib.pyplot as plt
import pandas as pd

# For progress bars when looping through patients or batches
from tqdm import tqdm

# Import the TCN model architecture definition
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add src/ to path

from ml_models_tcn.tcn_model import TCNModel

# -----------------------
# Path Directories
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# === TCN Model ===
TRAINED_MODEL_PATH = SCRIPT_DIR.parent.parent / "src" / "prediction_diagnostics" / "trained_models_refined" / "tcn_best_refined.pt"
CONFIG_PATH = SCRIPT_DIR.parent.parent / "src" / "prediction_diagnostics" / "trained_models_refined" / "config_refined.json"

# === TCN data and preprocessing directories ===
TEST_DATA_DIR = SCRIPT_DIR.parent.parent / "src" / "ml_models_tcn" / "prepared_datasets"
TCN_DIR = SCRIPT_DIR.parent.parent / "src" / "ml_models_tcn" / "deployment_models" / "preprocessing"

# === Preprocessing artifacts ===
SPLITS_PATH = TCN_DIR / "patient_splits.json"
PADDING_PATH = TCN_DIR / "padding_config.json"
SCALER_PATH = TCN_DIR / "standard_scaler.pkl"

# === Model tensors ===
TEST_TENSOR = TEST_DATA_DIR / "test.pt"
MASK_TENSOR = TEST_DATA_DIR / "test_mask.pt"

# === Output directory ===
RESULTS_DIR = SCRIPT_DIR / "interpretability_tcn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sanity Checks
# -----------------------------
assert TRAINED_MODEL_PATH.exists(), f"[ERROR] Missing trained model file: {TRAINED_MODEL_PATH}"
assert CONFIG_PATH.exists(), f"[ERROR] Missing model configuration: {CONFIG_PATH}"
assert (TEST_DATA_DIR / "test.pt").exists(), f"[ERROR] Missing test tensor: {TEST_DATA_DIR / 'test.pt'}"
assert (TEST_DATA_DIR / "test_mask.pt").exists(), f"[ERROR] Missing test mask tensor: {TEST_DATA_DIR / 'test_mask.pt'}"
assert SPLITS_PATH.exists(), f"[ERROR] Missing patient splits file: {SPLITS_PATH}"
assert PADDING_PATH.exists(), f"[ERROR] Missing padding config file: {PADDING_PATH}"
assert SCALER_PATH.exists(), f"[ERROR] Missing standard scaler file: {SCALER_PATH}"

print("[INFO] All required input files found. Ready to proceed.")

# -----------------------------
# 1. Load Model, Config, and Test Data
# -----------------------------
# --- Load device (cpu or gpu) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- Load config (architecture & parameters) ---
with open(CONFIG_PATH) as f:
    config = json.load(f)
arch = config["model_architecture"]

# --- Load padding/feature configuration ---
with open(PADDING_PATH) as f:
    pad_cfg = json.load(f)
feature_cols = pad_cfg['feature_cols']
MAX_SEQ_LEN = pad_cfg['max_seq_len']
target_cols = pad_cfg["target_cols"]

# --- Load test tensors ---
x_test = torch.load(TEST_TENSOR, map_location=device)
mask_test = torch.load(MASK_TENSOR, map_location=device)

# --- Validate test tensor shapes (shape should match padding config) ---
n_test, seq_len, n_features = x_test.shape
assert seq_len == MAX_SEQ_LEN, f"Expected seq_len {MAX_SEQ_LEN}, got {seq_len}"
assert n_features == len(feature_cols), "Feature dimension mismatch with padding_config"
print(f"[INFO] Loaded test data: {x_test.shape}, mask: {mask_test.shape}")

# --- Rebuild model architecture (from config) ---
model = TCNModel(
    num_features=n_features,
    num_channels=arch['num_channels'],
    kernel_size=arch['kernel_size'],
    dropout=arch['dropout'],
    head_hidden=arch['head_hidden']
).to(device)

# --- Load trained model weights (tcn_best_refined.pt) ---
state_dict = torch.load(TRAINED_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

# --- Set model to eval mode ---
model.eval()
print("[INFO] Loaded TCN model and moved to device.")

# -----------------------------
# 2. Define Targets & Saliency Function
# -----------------------------
"""
Purpose:
- This section defines what to explain (TARGETS) and how to compute saliency (compute_saliency_for_batch). 
- Saliency maps quantify how each input feature at each timestep influenced the model's output for that prediction head.
"""
# --- Target heads to explain ---
"""
- Each target corresponds to one model output head.
    - "logit_max" and "logit_median" → classification heads (BCEWithLogitsLoss)
    - "regression" → continuous regression head (MSELoss)
- We compute saliency separately for each head to understand which features (and when in time) drove each type of prediction.
"""
TARGETS = [
    ("max_risk", "logit_max"),       # classification
    ("median_risk", "logit_median"), # classification
    ("pct_time_high", "regression")  # regression
]

# --- Saliency Computation Helper ---
def compute_saliency_for_batch(model, x_batch, mask_batch, head_key):
    """
    Goal:
    - To measure how much each input feature at each time point influenced the model’s prediction, for every patient in a batch.
    - Computes |grad * input| saliency for a mini-batch of test patients. 

    Concept:
    - "Saliency" = gradient of model output wrt each input feature × the input value itself.
    - The gradient shows sensitivity: how much a small change in the feature would change the output.
    - Multiplying by input magnitude weights it by how active that feature was.
    - Taking absolute value removes directionality (we care about strength of influence).
    Args:
    - model: trained TCN model
    - x_batch: tensor (B, T, F) → batch of patient sequences
    - mask_batch: tensor (B, T) → binary mask (1 = real timestep, 0 = padding)
    - head_key: string ('logit_max', 'logit_median', 'regression') → which model output to explain
    Returns:
    - np.ndarray of shape (B, T, F) = |gradient * input|
    → saliency for each feature at each timestep for each patient
    """
    x = x_batch.clone().detach().to(device) # Clone tensor to avoid modifying original
    x.requires_grad = True # Enable gradient tracking on inputs (B, T, F), track how the model’s output changes if any of these values change
    mask = mask_batch.to(device) # Move mask to device

    outputs = model(x, mask) # Forward pass, model returns a dict of outputs
    out = outputs[head_key].squeeze() # Select target head output tensor, shape (B,) (one value per patient).

    grads = []

    # Loop over batch to compute gradients for each patient
    for i in range(out.shape[0]): 
        if x.grad is not None:
            x.grad.zero_() 
        scalar = out[i] # Select scalar output for patient i
        scalar.backward(retain_graph=True) # Backprop to compute gradients wrt inputs, so x.grad gets shape (B, T, F), one gradient for every input value of every patient.
        grads.append(x.grad[i].detach().cpu().numpy()) # Shape (T, F)
    grads = np.stack(grads, axis=0) # Shape (B, T, F), stack all per-pateint gradients of output wrt each input feature and timestep.

    saliency = grads * x.detach().cpu().numpy() # Combine gradients with input values (both shapes are the same so multiplication is valid)
    return np.abs(saliency) # Return absolute value of saliency (we care about magnitude only)
                            # Final saliency array: (B, T, F), Each row = one patient, Each column (within that) = one timepoint × feature pair.

# -----------------------------
# 3. Compute Interpretable Saliency Outputs
# -----------------------------
"""
Purpose:
- Compute fully interpretable, quantitative saliency outputs for each model target head.
- Avoid per-patient heatmaps because 171 features × 96 timesteps is not interpretable.
- Focus on CSV outputs and concise heatmaps for the top features.
- For each target head (max_risk, median_risk, pct_time_high):
    1. Compute per-patient |grad * input| saliency (shape: n_test × T × F)
    2. Generate feature-level mean + std CSVs → overall importance per feature
    3. Generate temporal mean profile CSV → how feature importance changes over time
    4. Generate top-5 features temporal profile CSV → interpretable temporal patterns of key features
    5. Generate global mean heatmap PNG for top 10 features → visual summary only
    6. Perform diagnostics → sanity checks, correlations between heads, NaNs, min/max
Rationale:
- Per-patient heatmaps are removed because 171 features × 96 timesteps is not interpretable.
- CSV outputs provide numerical, fully interpretable data for analysis.
"""

# Batch size for gradient computation
batch_size = 4

# --- Loop over all targets (three model heads) ---
for target_name, head_key in TARGETS:
    print(f"\n[INFO] ===== Saliency for target: {target_name} ({head_key}) =====")

    # -----------------------------
    # 3A. Compute per-patient saliency for all batches
    # -----------------------------
    """
    - Compute |∂y/∂x * x| for each patient sequence
    - Result: 3D array (n_test, T, F)
      - n_test = number of patients in test set
      - T = timesteps (max sequence length)
      - F = number of input features
    - This is the fundamental data from which all interpretable summaries are derived.
    """
    per_patient_saliency = []

    for i in tqdm(range(0, n_test, batch_size), desc=f"Processing {target_name}"):
        xb = x_test[i:i+batch_size].to(device)
        mb = mask_test[i:i+batch_size].to(device)
        sal_b = compute_saliency_for_batch(model, xb, mb, head_key) 
        per_patient_saliency.append(sal_b)

    per_patient_saliency = np.concatenate(per_patient_saliency, axis=0) # Stack all batches → (n_test, T, F)
    print(f"[INFO] Full saliency array shape: {per_patient_saliency.shape}")

    # -----------------------------
    # 3B. Feature-level mean + std saliency CSV
    # -----------------------------
    """
    - Aggregates saliency over all patients and timesteps
    - Computes mean and standard deviation per feature
    - CSV provides interpretable numerical values for feature importance
    """
    feature_mean = per_patient_saliency.mean(axis=(0, 1))  # (F,)
    feature_std = per_patient_saliency.std(axis=(0, 1))    # (F,)
    df_features = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_saliency": feature_mean,
        "std_abs_saliency": feature_std
    }).sort_values("mean_abs_saliency", ascending=False)

    df_features.to_csv(RESULTS_DIR / f"{target_name}_feature_saliency.csv", index=False)
    print(f"[INFO] Saved feature-level saliency CSV → {target_name}_feature_saliency.csv")

    # -----------------------------
    # 3C. Temporal mean profile CSV
    # -----------------------------
    """
    - Computes mean saliency across features for each timestep
    - Reveals when during the sequence the model is most sensitive
    - CSV: timestep × mean_abs_saliency
    """
    temporal_mean = per_patient_saliency.mean(axis=(0, 2))  # (T,)

    df_temporal = pd.DataFrame({
        "timestep": np.arange(MAX_SEQ_LEN),
        "mean_abs_saliency": temporal_mean
    })
    df_temporal.to_csv(RESULTS_DIR / f"{target_name}_temporal_saliency.csv", index=False)
    print(f"[INFO] Saved temporal saliency CSV → {target_name}_temporal_saliency.csv")

    # -----------------------------
    # 3D. Top-5 features temporal profile CSV
    # -----------------------------
    """
    - Identify the top 5 features by mean absolute saliency
    - Compute temporal profile (mean over patients) for these features
    - CSV: timestep × top features → interpretable temporal trends
    """
    top_features_idx = feature_mean.argsort()[::-1][:5]  # indices of top 5 features
    top_features = [feature_cols[i] for i in top_features_idx]
    temporal_top = per_patient_saliency[:, :, top_features_idx].mean(axis=0)  # (T, 5)
    df_top_temporal = pd.DataFrame(temporal_top, columns=top_features)
    df_top_temporal.insert(0, "timestep", np.arange(MAX_SEQ_LEN))
    df_top_temporal.to_csv(RESULTS_DIR / f"{target_name}_top_features_temporal.csv", index=False)
    print(f"[INFO] Saved top-5 features temporal CSV → {target_name}_top_features_temporal.csv")

    # -----------------------------
    # 3E. Global mean heatmap PNG (top 10 features)
    # -----------------------------
    """
    - For visualisation only; shows mean saliency of top 10 features over time
    - Provides a visual summary but is not needed for quantitative interpretation
    """
    top10_idx = feature_mean.argsort()[::-1][:10]
    top10_features = [feature_cols[i] for i in top10_idx]

    # Compute mean saliency over patients for top 10 features
    mean_top10 = per_patient_saliency[:, :, top10_idx].mean(axis=0)  # (T, 10)

    # Apply log transform to enhance visibility of small differences
    plot_data = np.log1p(mean_top10)  # log(1 + x) to handle zeros safely

    # Percentile-based color scaling to reduce influence of outliers
    vmin, vmax = np.percentile(plot_data, [5, 95])

    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        plot_data.T,
        aspect='auto',
        interpolation='nearest',
        cmap='plasma',   # perceptually uniform colormap
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(im, label='mean |grad * input| (log scale)')
    plt.ylabel('Top 10 Features')
    plt.yticks(ticks=np.arange(len(top10_features)), labels=top10_features, fontsize=8)
    plt.xlabel('Time (timestep)')
    plt.title(f"Mean Saliency Heatmap — {target_name} — Top 10 Features")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{target_name}_mean_heatmap.png", dpi=200)
    plt.close()
    print(f"[INFO] Saved global mean heatmap → {target_name}_mean_heatmap.png")

    # -----------------------------
    # 3F. Diagnostics
    # -----------------------------
    """
    - Provides sanity checks on the saliency outputs:
        - NaN count, mean, max values
        - Correlations between model heads to identify potential redundancy or overlap
    - Ensures outputs are valid and interpretable numerically
    """
    print("\n[DIAGNOSTICS] ==============================")
    print("Saliency stats:")
    print(" - NaN count:", np.isnan(per_patient_saliency).sum())
    print(" - Mean value:", np.nanmean(per_patient_saliency))
    print(" - Max value:", np.nanmax(per_patient_saliency))

    # Head correlations
    outputs = model(x_test, mask_test)
    max_out = outputs["logit_max"].detach().cpu().numpy().ravel()
    med_out = outputs["logit_median"].detach().cpu().numpy().ravel()
    reg_out = outputs["regression"].detach().cpu().numpy().ravel()
    print("Correlation between head outputs:")
    print("max ↔ median:", np.corrcoef(max_out, med_out)[0, 1])
    print("max ↔ regression:", np.corrcoef(max_out, reg_out)[0, 1])

print("\n[INFO] ✅ TCN saliency computation complete.")