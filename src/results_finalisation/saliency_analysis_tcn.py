# src/results_finalisation/saliency_analysis_tcn.py
"""
saliency_analysis_tcn.py

Title: Compute temporal saliency maps for the refined TCN model on the held-out test set.

Outputs
-------
- per-patient saliency arrays (npz):  interpretability_tcn/patient_saliency_{target}.npz  (keys: "patient_{id}" -> (max_seq_len, n_features))
- per-patient heatmap PNGs:    interpretability_tcn/{target}_patient_{i:02d}_heatmap.png
- global mean heatmap PNG:    interpretability_tcn/{target}_mean_heatmap.png
- top-10 feature CSV:         interpretability_tcn/{target}_top10_saliency.csv
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
Target heads to explain
- Each target corresponds to one model output head.
    - "logit_max" and "logit_median" → classification heads (BCEWithLogitsLoss)
    - "regression" → continuous regression head (MSELoss)
- We compute saliency separately for each head to understand which features (and when in time) drove each type of prediction.
"""
# --- Target heads to explain ---
TARGETS = [
    ("max_risk", "logit_max"),       # classification
    ("median_risk", "logit_median"), # classification
    ("pct_time_high", "regression")  # regression
]

# --- Saliency Computation Helper ---
def compute_saliency_for_batch(model, x_batch, mask_batch, head_key):
    """
    Compute |grad * input| saliency for a mini-batch of test patients.

    Concept:
    - "Saliency" = gradient of model output wrt each input feature × the input value itself.
    - The gradient shows *sensitivity*: how much a small change in the feature would change the output.
    - Multiplying by input magnitude weights it by how active that feature was.
    - Taking absolute value removes directionality (we care about *strength* of influence).
    Args:
    - model: trained TCN model
    - x_batch: tensor (B, T, F) → batch of patient sequences
    - mask_batch: tensor (B, T) → binary mask (1 = real timestep, 0 = padding)
    - head_key: string ('logit_max', 'logit_median', 'regression') → which output to explain
    Returns:
    - np.ndarray of shape (B, T, F) = |gradient * input|
    → saliency for each feature at each timestep for each patient
    """
    x = x_batch.clone().detach().to(device)
    x.requires_grad = True
    mask = mask_batch.to(device)

    outputs = model(x, mask)
    out = outputs[head_key].squeeze()

    grads = []
    for i in range(out.shape[0]):
        if x.grad is not None:
            x.grad.zero_()
        scalar = out[i]
        scalar.backward(retain_graph=True)
        grads.append(x.grad[i].detach().cpu().numpy())
    grads = np.stack(grads, axis=0)

    saliency = grads * x.detach().cpu().numpy()
    return np.abs(saliency)



# =============================================================
# 3. Generate Per-Patient & Global Saliency Outputs
# =============================================================
batch_size = 4

for target_name, head_key in TARGETS:
    print(f"\n[INFO] ===== Saliency for target: {target_name} ({head_key}) =====")

    # --- Compute per-patient saliency ---
    per_patient_saliency = []
    for i in tqdm(range(0, n_test, batch_size), desc=f"Processing {target_name}"):
        xb = x_test[i:i+batch_size].to(device)
        mb = mask_test[i:i+batch_size].to(device)
        sal_b = compute_saliency_for_batch(model, xb, mb, head_key)
        per_patient_saliency.append(sal_b)

    per_patient_saliency = np.concatenate(per_patient_saliency, axis=0)
    print(f"[INFO] Saliency shape: {per_patient_saliency.shape}")

    # --- Save per-patient arrays ---
    save_npz = RESULTS_DIR / f"patient_saliency_{target_name}.npz"
    np.savez_compressed(save_npz, **{f"patient_{i}": per_patient_saliency[i] for i in range(n_test)})
    print(f"[INFO] Saved saliency arrays → {save_npz}")

    # =========================================================
    # 3A. Patient-Level Heatmaps
    # =========================================================
    for i in range(n_test):
        arr = per_patient_saliency[i]
        mask_np = mask_test[i].cpu().numpy().astype(bool)
        arr[~mask_np, :] = np.nan

        plt.figure(figsize=(14, 6))
        plt.imshow(arr.T, aspect='auto', interpolation='nearest')
        plt.colorbar(label='|grad * input|')
        plt.ylabel('Feature')
        plt.yticks(np.arange(len(feature_cols)), feature_cols, fontsize=6)
        plt.xlabel('Time (timestep)')
        plt.title(f"Saliency Heatmap — {target_name} — Patient {i}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{target_name}_patient_{i:02d}_heatmap.png", dpi=200)
        plt.close()

    print(f"[INFO] Saved {n_test} patient-level heatmaps for {target_name}")

    # =========================================================
    # 3B. Global Mean Heatmap
    # =========================================================
    mean_saliency = np.nanmean(per_patient_saliency, axis=0)
    plt.figure(figsize=(14, 6))
    plt.imshow(mean_saliency.T, aspect='auto', interpolation='nearest')
    plt.colorbar(label='mean |grad * input|')
    plt.ylabel('Feature')
    plt.yticks(np.arange(len(feature_cols)), feature_cols, fontsize=6)
    plt.xlabel('Time (timestep)')
    plt.title(f"Mean Saliency — {target_name} — Averaged Across Patients")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{target_name}_mean_heatmap.png", dpi=200)
    plt.close()
    print(f"[INFO] Saved global mean heatmap for {target_name}")

    # =========================================================
    # 3C. Top-10 Feature Ranking
    # =========================================================
    feature_mean_sal = per_patient_saliency.mean(axis=(0, 1))
    df_top = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_saliency": feature_mean_sal})
        .sort_values("mean_abs_saliency", ascending=False)
        .head(10)
    )
    df_top.to_csv(RESULTS_DIR / f"{target_name}_top10_saliency.csv", index=False)
    print(f"[INFO] Saved Top-10 Features → {RESULTS_DIR / f'{target_name}_top10_saliency.csv'}")

print("\n[INFO] ✅ TCN saliency computation complete.")