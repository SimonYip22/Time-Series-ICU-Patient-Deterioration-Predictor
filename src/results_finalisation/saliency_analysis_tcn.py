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
import torch
import matplotlib.pyplot as plt
import pandas as pd


from tqdm import tqdm


from ml_models_tcn.tcn_model import TCNModel

# -----------------------
# Path Directories
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# === Input directories ===
TRAINED_MODEL_PATH = SCRIPT_DIR.parent.parent / "src" / "prediction_diagnostics" / "trained_models_refined" / "tcn_best_refined.pt"
CONFIG_PATH = SCRIPT_DIR.parent.parent / "src" / "prediction_diagnostics" / "trained_models_refined" / "config_refined.json"
TEST_DATA_DIR = SCRIPT_DIR.parent.parent / "src" / "ml_models_tcn" / "prepared_datasets" # test tensors
TCN_DIR = SCRIPT_DIR.parent.parent / "src" / "ml_models_tcn" / "deployment_models" / "preprocessing"
SPLITS_PATH = TCN_DIR / "patient_splits.json"
PADDING_PATH = TCN_DIR / "padding_config.json"
SCALER_PATH = TCN_DIR / "standard_scaler.pkl"

# === Output directory ===
RESULTS_DIR = SCRIPT_DIR / "interpretability_tcn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Sanity Checks
# -----------------------------
assert TRAINED_MODEL_PATH.exists(), f"Missing model: {TRAINED_MODEL_PATH}"
assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
assert (TEST_DATA_DIR / "test.pt").exists(), f"Missing test tensor: {TEST_DATA_DIR / 'test.pt'}"
assert (TEST_DATA_DIR / "test_mask.pt").exists(), f"Missing test mask: {TEST_DATA_DIR / 'test_mask.pt'}"
assert SPLITS_PATH.exists(), f"Missing splits: {SPLITS_PATH}"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------
# Load Model + Config + Test Tensors
# -----------------------------
# Load model configuration (refined)
with open(CONFIG_PATH) as f:
    config = json.load(f)
arch = config["model_architecture"]

with open(SCALER_DIR / "padding_config.json", 'r') as f:
    pad_cfg = json.load(f)

feature_cols = pad_cfg['feature_cols']
MAX_SEQ_LEN = pad_cfg['max_seq_len']
target_cols = pad_cfg["target_cols"]

# Load tensors (x_test: [n_patients, seq_len, n_features], mask_test similar)
x_test = torch.load(TEST_DATA_DIR / "test.pt", map_location=device)
mask_test = torch.load(TEST_DATA_DIR / "test_mask.pt", map_location=device)

n_test, seq_len, n_features = x_test.shape
print(f"[INFO] Loaded x_test: {x_test.shape}, mask: {mask_test.shape}")
assert seq_len == MAX_SEQ_LEN, f"Expected seq_len {MAX_SEQ_LEN}, got {seq_len}"
assert n_features == len(feature_cols), "Feature dimension mismatch with padding_config"

# --- Reload model architecture and weights ---
arch = config['model_architecture']
model = TCNModel(
    num_features=n_features,
    num_channels=arch['num_channels'],
    kernel_size=arch['kernel_size'],
    dropout=arch['dropout'],
    head_hidden=arch['head_hidden']
)

state_dict = torch.load(TRAINED_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print('[INFO] Loaded TCN model')

# Targets we will explain and corresponding model output keys
TARGETS = [
    ("max", "logit_max"),       # binary classification -> use logit (pre-sigmoid)
    ("median", "logit_median"),
    ("pct_time_high", "regression")  # regression head
]

# --- Helper: compute gradient*input saliency for a batch of patients ---
def compute_saliency_for_batch(model, x_batch, mask_batch, head_key):
    """Compute gradient * input saliency for a batch.
    x_batch: torch tensor (B, T, F) requires_grad=False
    mask_batch: torch tensor (B, T) — not used directly in gradient calc but kept for clarity
    head_key: string key returned by model(x, mask) -> selects which scalar to compute gradient of

    Returns: saliency_abs: numpy array shape (B, T, F) of absolute(grad * input)
    """
    # Ensure we run on device
    x = x_batch.clone().detach().to(device)
    x.requires_grad = True

    mask = mask_batch.to(device) if mask_batch is not None else None

    outputs = model(x, mask)
    out = outputs[head_key]  # shape (B, 1) or (B,)
    out = out.squeeze()

    # For multi-output batch: compute gradient of each scalar with respect to inputs
    # We'll compute a vector-Jacobian product that results in gradients of shape (B, T, F)
    grads = []
    for i in range(out.shape[0]):
        # Zero existing grads
        if x.grad is not None:
            x.grad.zero_()
        scalar = out[i]
        # Backprop scalar
        scalar.backward(retain_graph=True)
        g = x.grad[i].detach().cpu().numpy().copy()  # (T, F)
        grads.append(g)
    grads = np.stack(grads, axis=0)  # (B, T, F)

    x_np = x.detach().cpu().numpy()  # (B, T, F)

    saliency = grads * x_np  # elementwise grad * input
    saliency_abs = np.abs(saliency)
    return saliency_abs

# --- Main loop: compute per-patient saliency for each target ---
for target_name, head_key in TARGETS:
    print(f"[INFO] Computing saliency for target: {target_name} (head: {head_key})")

    # We'll compute per-patient saliency in batches to be memory-friendly
    batch_size = 4
    per_patient_saliency = []  # list of (T, F) arrays

    with torch.no_grad():
        # NOTE: we need gradients; temporarily enable gradient context by switching to requires_grad path inside helper
        # We'll compute in small batches and re-enable grad per-batch using compute_saliency_for_batch
        for i in range(0, n_test, batch_size):
            xb = x_test[i:i+batch_size].to(device)
            mb = mask_test[i:i+batch_size].to(device)
            # compute with gradient tracking ON inside helper
            sal_b = compute_saliency_for_batch(model, xb, mb, head_key)
            per_patient_saliency.append(sal_b)

    per_patient_saliency = np.concatenate(per_patient_saliency, axis=0)  # shape (n_test, T, F)
    print(f"[INFO] Saliency array shape for {target_name}: {per_patient_saliency.shape}")

    # Save per-patient saliency arrays into npz (keyed by patient index)
    save_npz = RESULTS_DIR / f"patient_saliency_{target_name}.npz"
    npz_dict = {f"patient_{i}": per_patient_saliency[i] for i in range(per_patient_saliency.shape[0])}
    np.savez_compressed(save_npz, **npz_dict)
    print(f"[INFO] Saved per-patient saliency arrays → {save_npz}")

    # --- Generate patient-level heatmap PNGs (optional: small number, here save all test patients) ---
    for i in range(per_patient_saliency.shape[0]):
        arr = per_patient_saliency[i]  # (T, F)
        # Optionally zero-out padded timesteps using mask
        mask_np = mask_test[i].cpu().numpy().astype(bool)
        # Masked heatmap: set padded rows to NaN for plotting transparency
        plot_arr = arr.copy()
        if mask_np.shape[0] == plot_arr.shape[0]:
            plot_arr[~mask_np, :] = np.nan

        plt.figure(figsize=(14, 6))
        plt.imshow(plot_arr.T, aspect='auto', interpolation='nearest')
        plt.colorbar(label='|grad * input|')
        plt.ylabel('Feature (index)')
        plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, fontsize=6)
        plt.xlabel('Time (timestep)')
        plt.title(f"Saliency heatmap — {target_name} — patient {i}")
        plt.tight_layout()
        out_png = RESULTS_DIR / f"{target_name}_patient_{i:02d}_heatmap.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

    print(f"[INFO] Saved per-patient heatmaps for {target_name} (n={per_patient_saliency.shape[0]})")

    # --- Global mean heatmap (mean over patients) ---
    mean_over_patients = np.nanmean(per_patient_saliency, axis=0)  # (T, F)
    plt.figure(figsize=(14, 6))
    plt.imshow(mean_over_patients.T, aspect='auto', interpolation='nearest')
    plt.colorbar(label='mean |grad * input|')
    plt.ylabel('Feature (index)')
    plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, fontsize=6)
    plt.xlabel('Time (timestep)')
    plt.title(f"Mean Saliency heatmap — {target_name} — mean across test patients")
    plt.tight_layout()
    out_mean_png = RESULTS_DIR / f"{target_name}_mean_heatmap.png"
    plt.savefig(out_mean_png, dpi=200)
    plt.close()
    print(f"[INFO] Saved mean heatmap → {out_mean_png}")

    # --- Top-10 features by mean absolute saliency (averaged over patients & time) ---
    # Compute mean abs saliency per feature: first mean over patients and time
    feature_mean_sal = per_patient_saliency.mean(axis=(0, 1))  # (F,)
    df_top = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_saliency': feature_mean_sal
    }).sort_values('mean_abs_saliency', ascending=False)

    df_top.to_csv(RESULTS_DIR / f"{target_name}_top10_saliency.csv", index=False)
    print(f"[INFO] Saved top-10 feature saliency CSV → {RESULTS_DIR / f'{target_name}_top10_saliency.csv'}")

print('[INFO] TCN saliency computation complete')
