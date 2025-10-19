# src/prediction_evaluations/evaluate_lightgbm_testset.py
"""
evaluate_lightgbm_testset.py

Title: Re-train and Evaluate Final LightGBM Models

Summary:
- Rebuilds train/test splits from patient_splits.json + news2_features_patient.csv, maintains exact order.
- Recreates binary targets (max_risk_binary, median_risk_binary).
- Loads tuned best hyperparameters (best_params.json).
- Retrains one LightGBM per target on the train split (70 patients) with best hyperparams.
- Evaluates each model on test split (15 patients) to produce predictions.
- Computes metrics using compute_classification_metrics + compute_regression_metrics functions from evaluation_metrics.py.
- Saves models, predictions, metrics, and a summary text file to lightgbm_results/.
- Prints clear progress logs and key metrics for reproducibility and transparency.

Outputs (saved to lightgbm_results/):
- 3x {target}_retrained_model.pkl → retrained LightGBM model files (.pkl)
- lightgbm_predictions.csv → test set true and predicted values CSV
- lightgbm_metrics.json → classification/regression performance metrics JSON
- training_summary.txt → hyperparameters + test metrics summary
"""

#-------------------------------------------------------------
# 0. Imports
#-------------------------------------------------------------
import pandas as pd
import lightgbm as lgb
import joblib
import json
from pathlib import Path

# Import functions from evaluation_metrics.py 
from evaluation_metrics import compute_classification_metrics, compute_regression_metrics

# -------------------------------------------------------------
# 0. Directories and Files
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Input files
DATA_PATH = SCRIPT_DIR.parent.parent / "data" / "processed_data" / "news2_features_patient.csv"
SPLITS_PATH = SCRIPT_DIR.parent / "ml_models_tcn" / "deployment_models" / "preprocessing"/ "patient_splits.json"
HYPERPARAMS_PATH = SCRIPT_DIR.parent / "ml_models_lightgbm" / "hyperparameter_tuning_runs" / "best_params.json"

# Output directory
RESULTS_DIR = SCRIPT_DIR.parent / "prediction_evaluations" / "lightgbm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output files
PREDICTIONS_CSV = RESULTS_DIR / "lightgbm_predictions.csv"
METRICS_JSON = RESULTS_DIR / "lightgbm_metrics.json"
SUMMARY_TXT = RESULTS_DIR / "training_summary.txt"

# -------------------------------------------------------------
# 1. Load Hyperparameters
# -------------------------------------------------------------
"""
Step 1: Load model hyperparameters.

- Reads the tuned LightGBM hyperparameters from `best_params.json`, which stores the
  optimal configuration for each prediction target (`max_risk`, `median_risk`, `pct_time_high`).
- Validates that all expected targets are present to prevent runtime errors.
- These parameters are later used to initialize and retrain each model on the patient-level data.
"""

print("\n[STEP] Loading hyperparameters...")

# Load hyperparameters file
with open(HYPERPARAMS_PATH, "r") as f:
    best_params = json.load(f)

# Ensure expected keys
TARGETS = ["max_risk", "median_risk", "pct_time_high"]
for t in TARGETS:
    if t not in best_params:
        raise KeyError(f"Missing hyperparameters for target '{t}' in {HYPERPARAMS_PATH}")

# -------------------------------------------------------------
# 2. Load Data and Splits
# -------------------------------------------------------------
"""
Step 2: Load patient-level data and fixed train/test splits.

- Loads the engineered patient-level NEWS2 feature dataset from `news2_features_patient.csv`.
- Reads `patient_splits.json` to reconstruct the exact patient IDs assigned to
  the training and test sets, preserving deterministic ordering for evaluation.
- Recomputes binary classification targets:
    - `max_risk_binary`: 1 if max_risk > 2
    - `median_risk_binary`: 1 if median_risk == 2
- Excludes non-feature columns to construct the final model input matrix.
"""

print("\n[STEP] Loading patient-level data...")

# Load patient-level features CSV into dataframe
df = pd.read_csv(DATA_PATH)

# Load train/val/test split IDs JSON
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

# Load patient IDs for train and test
train_ids = splits["train"]
test_ids = splits["test"]

# Build train/test DataFrames (preserve order for test)
train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
test_df = df.set_index("subject_id").loc[test_ids].reset_index()

# --- Recreate test targets dynamically ---
# Recreate binary targets for train + test (consistent with Phase 3 + 4)
for d in (train_df, test_df):
    d["max_risk_binary"] = d["max_risk"].apply(lambda x: 1 if x > 2 else 0)
    d["median_risk_binary"] = d["median_risk"].apply(lambda x: 1 if x == 2 else 0)

# Feature columns (everything except subject_id + original targets)
exclude_cols = ["subject_id", "max_risk", "median_risk", "pct_time_high",
                "max_risk_binary", "median_risk_binary"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"[INFO] Loaded {len(train_df)} train and {len(test_df)} test patients.")
print(f"[INFO] Using {len(feature_cols)} feature columns.\n")

# -------------------------------------------------------------
# 3. Train LightGBM Models
# -------------------------------------------------------------
"""
Step 3: Retrain LightGBM models on the training split.

- Iterates through each prediction target (`max_risk`, `median_risk`, `pct_time_high`).
- Initialises the LightGBM model (classifier or regressor) using the corresponding tuned parameters.
- Trains each model on patient-level aggregated features and target labels.
- Saves the retrained model to disk as a .pkl file in `lightgbm_results/`.
- Logs the model type, sample count, and file path for reproducibility.
"""

# List of targets to loop through
targets = [
    ("max_risk", "classification"),
    ("median_risk", "classification"),
    ("pct_time_high", "regression")
]

print("[STEP] Training LightGBM models...\n")

# Loop through each target
for target, task_type in targets:

    print(f"[INFO] Training {task_type} model for target: {target}")
    
    # Training patient data
    X_train = train_df[feature_cols]

    # Training patient true target values
    y_train = (
        train_df["max_risk_binary"] if target == "max_risk" else
        train_df["median_risk_binary"] if target == "median_risk" else
        train_df["pct_time_high"]
    ).to_numpy()

    # Best tuned hyperparameters for each target
    params = best_params[target]

    # Initialise model
    if task_type == "classification":
        model = lgb.LGBMClassifier(**params, random_state=42, class_weight="balanced", verbose=-1)
    else:
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

    # Train model with the training data and true values
    model.fit(X_train, y_train)
    
    # Save retrained model per target (.pkl)
    model_file = RESULTS_DIR / f"{target}_retrained_model.pkl"
    joblib.dump(model, model_file)

    print(f"[TRAINED] {target} ({task_type}) → {X_train.shape[0]} samples | saved: {model_file}")

# -------------------------------------------------------------
# 4. Evaluate Models on Test Set
# -------------------------------------------------------------
"""
Step 4: Evaluate trained models on the test set.

- Loads each saved model and applies it to the test split.
- Generates predictions:
    - For classification: predicted probabilities (`y_prob`)
    - For regression: continuous predictions (`y_pred`)
- Stores predictions and corresponding true values in `preds_dict` for downstream analysis.
"""

print("\n[STEP] Evaluating models on test set...\n")

# Create dictionary to store predictions + corresponding true values
preds_dict = {}

# Loop through targets
for target, task_type in targets:
    print(f"[INFO] Evaluating {task_type} model for target: {target}")

    # Load trained model (.pkl)
    model = joblib.load(RESULTS_DIR / f"{target}_retrained_model.pkl")

    # Test patient data
    X_test = test_df[feature_cols]

    # Test patient true values
    y_test = (
        test_df["max_risk_binary"] if target == "max_risk" else
        test_df["median_risk_binary"] if target == "median_risk" else
        test_df["pct_time_high"]
    ).to_numpy()

    # --- Generate predictions based on test patient data ---
    # Save each targets prediction + true values into dictionary
    if task_type == "classification":
        y_prob = model.predict_proba(X_test)[:, 1]
        preds_dict[target] = {"y_true": y_test, "prob": y_prob}
    else:
        y_pred = model.predict(X_test)
        preds_dict[target] = {"y_true": y_test, "y_pred": y_pred}

print("\n[INFO] Completed evaluation for all models.\n")

# -------------------------------------------------------------
# 5. Save Predictions CSV
# -------------------------------------------------------------
"""
Step 5: Save model predictions to CSV.

- Aggregates true values and predictions for all three targets into a single table (`df_preds`).
- Each row corresponds to one patient in the test set.
- Ensures alignment between y_true and prediction columns for interpretability.
- Saves the results to `lightgbm_predictions.csv`.
"""

print("\n[INFO] Saving LightGBM predictions...")

# Each column is a target value (6), each row is a patient (15)
df_preds = pd.DataFrame({
    "y_true_max": preds_dict["max_risk"]["y_true"],
    "prob_max": preds_dict["max_risk"]["prob"],
    "y_true_median": preds_dict["median_risk"]["y_true"],
    "prob_median": preds_dict["median_risk"]["prob"],
    "y_true_reg": preds_dict["pct_time_high"]["y_true"],
    "y_pred_reg": preds_dict["pct_time_high"]["y_pred"]
})

# Save to CSV
df_preds.to_csv(PREDICTIONS_CSV, index=False)
print(f"[INFO] Saved LightGBM testset predictions → {PREDICTIONS_CSV}")

# -------------------------------------------------------------
# 6. Compute Metrics
# -------------------------------------------------------------
"""
Step 6: Compute and report model performance metrics.

- Uses helper functions from evaluation_metrics.py to compute classification (AUROC, F1, Precision, etc.)
  and regression (R², RMSE) metrics.
- Prints formatted metrics to console for immediate review.
- Saves full metrics as JSON (`lightgbm_metrics.json`) for downstream analysis.
"""

print("[STEP] Computing performance metrics...\n")

# Compute metrics using imported functions on true vs prediction values
metrics_max = compute_classification_metrics(preds_dict["max_risk"]["y_true"], preds_dict["max_risk"]["prob"])
metrics_median = compute_classification_metrics(preds_dict["median_risk"]["y_true"], preds_dict["median_risk"]["prob"])
metrics_reg = compute_regression_metrics(preds_dict["pct_time_high"]["y_true"], preds_dict["pct_time_high"]["y_pred"])

# Save computed metrics into dictionary
metrics = {
    "max_risk": metrics_max,
    "median_risk": metrics_median,
    "pct_time_high": metrics_reg,
}

# Print evaluation metrics for all targets
for target, m in metrics.items():
    print(f"[RESULTS] {target}:")
    for k, v in m.items():
        print(f"  {k:<12}: {v:.4f}")
    print("")

# Save metrics JSON
with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"[INFO] Metrics saved → {METRICS_JSON}")

# -------------------------------------------------------------
# 7. Save Evaluation Summary
# -------------------------------------------------------------
"""
Step 7: Save concise model evaluation summary.

- Summarizes dataset info, test set metrics, and tuned hyperparameters per target.
- Written to `training_summary.txt` for easy inspection or inclusion in supplementary materials.
- Provides a reproducible record of which parameters generated each set of metrics.
"""

with open(SUMMARY_TXT, "w") as f:
    f.write("=== LightGBM Evaluation Summary ===\n")
    f.write(f"Data source: {DATA_PATH.name}\n")
    f.write(f"Splits: {len(train_df)} train, {len(test_df)} test patients\n\n")

    # Loop through targets
    for target in TARGETS:
        f.write(f"--- Target: {target} ---\n")

        # Print evaluation metrics
        f.write("Test set metrics:\n")
        for k, v in metrics[target].items():
            f.write(f"{k:<12}: {v:.4f}\n")

        # Print hyperparameters
        f.write("\nBest hyperparameters:\n")
        for k, v in best_params[target].items():
            f.write(f"  {k}: {v}\n")

        f.write("\n")

print(f"[INFO] Training summary saved to {SUMMARY_TXT}")
print("\n[INFO] DONE — All models trained, evaluated, and saved successfully.\n")