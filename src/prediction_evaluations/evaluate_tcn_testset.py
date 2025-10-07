"""
evaluate_tcn_testset.py

Title: Final Evaluation of Temporal Convolutional Network (TCN) on Held-Out Test Set

Summary:
- Loads the best TCN model checkpoint (tcn_best.pt).
- Runs inference on the unseen test set (x_test, mask_test).
- Computes classification + regression metrics via evaluation_metrics.py.
- Saves patient-level predictions (CSV) and aggregated metrics (JSON).
- Provides reproducible test-time performance for reporting and comparison (Phase 5 Step 1).
Output:
- 
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import torch
import json
import pandas as pd
from pathlib import Path
from time import time
from evaluation_metrics import compute_classification_metrics, compute_regression_metrics
from ml_models_tcn.tcn_model import TCNModel  

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR = SCRIPT_DIR.parent
TRAINED_MODEL_PATH = BASE_DIR / "ml_models_tcn" / "trained_models" / "tcn_best.pt"
TEST_DATA_DIR = BASE_DIR / "ml_models_tcn" / "data_tensors"

RESULTS_DIR = BASE_DIR / "prediction_evaluations" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Load Model + Test Data
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load test tensors (produced in Phase 4)
x_test = torch.load(TEST_DATA_DIR / "x_test.pt", map_location=device)
mask_test = torch.load(TEST_DATA_DIR / "mask_test.pt", map_location=device)
y_test = torch.load(TEST_DATA_DIR / "y_test.pt", map_location=device)  # dict or tuple depending on pipeline

# Load best TCN model weights
model = TCNModel.load_from_checkpoint(TRAINED_MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# -------------------------------------------------------------
# Inference (No Grad)
# -------------------------------------------------------------
print("[INFO] Running inference on test set...")
start_time = time()

with torch.no_grad():
    outputs = model(x_test, mask_test)
    # Expect dictionary-style outputs: {"max": logits1, "median": logits2, "reg": reg_pred}
    logits_max = outputs["max"].squeeze().cpu()
    logits_median = outputs["median"].squeeze().cpu()
    preds_reg = outputs["reg"].squeeze().cpu()

# Apply sigmoid for classification heads to get probabilities
prob_max = torch.sigmoid(logits_max).numpy()
prob_median = torch.sigmoid(logits_median).numpy()
y_pred_reg = preds_reg.numpy()

inference_time = time() - start_time
print(f"[INFO] Inference complete in {inference_time:.2f} seconds")

# -------------------------------------------------------------
# Prepare Ground Truth
# -------------------------------------------------------------
# y_test may be stored as dict → handle accordingly
if isinstance(y_test, dict):
    y_true_max = y_test["max"].cpu().numpy()
    y_true_median = y_test["median"].cpu().numpy()
    y_true_reg = y_test["reg"].cpu().numpy()
else:
    raise ValueError("y_test format not recognised — expected dictionary with keys ['max', 'median', 'reg'].")

# -------------------------------------------------------------
# Compute Metrics
# -------------------------------------------------------------
metrics_max = compute_classification_metrics(y_true_max, prob_max)
metrics_median = compute_classification_metrics(y_true_median, prob_median)
metrics_reg = compute_regression_metrics(y_true_reg, y_pred_reg)

# Combine into one structure
all_metrics = {
    "max_risk": metrics_max,
    "median_risk": metrics_median,
    "pct_time_high": metrics_reg,
    "inference_time_sec": round(inference_time, 2)
}

# -------------------------------------------------------------
# Save Metrics (JSON)
# -------------------------------------------------------------
with open(RESULTS_DIR / "tcn_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)
print("[INFO] Saved metrics → results/tcn_metrics.json")

# -------------------------------------------------------------
# Save Predictions (CSV)
# -------------------------------------------------------------
df_preds = pd.DataFrame({
    "y_true_max": y_true_max,
    "prob_max": prob_max,
    "y_true_median": y_true_median,
    "prob_median": prob_median,
    "y_true_reg": y_true_reg,
    "y_pred_reg": y_pred_reg
})
df_preds.to_csv(RESULTS_DIR / "tcn_predictions.csv", index=False)
print("[INFO] Saved predictions → results/tcn_predictions.csv")

# -------------------------------------------------------------
# Display Summary
# -------------------------------------------------------------
print("\n=== Final Test Metrics ===")
print(f"Max Risk — AUC: {metrics_max['roc_auc']:.3f}, F1: {metrics_max['f1']:.3f}, Acc: {metrics_max['accuracy']:.3f}")
print(f"Median Risk — AUC: {metrics_median['roc_auc']:.3f}, F1: {metrics_median['f1']:.3f}, Acc: {metrics_median['accuracy']:.3f}")
print(f"Regression — RMSE: {metrics_reg['rmse']:.3f}, R²: {metrics_reg['r2']:.3f}")
print("==========================")