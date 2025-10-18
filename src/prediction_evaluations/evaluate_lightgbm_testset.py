# src/prediction_evaluations/evaluate_lightgbm_testset.py

"""
evaluate_final_lightgbm.py

Title: Evaluate Final Deployment-Ready LightGBM Models

Summary:

- Rebuilds train/test from patient_splits.json + news2_features_patient.csv
- Recreates binary targets (max_risk_binary, median_risk_binary)
- Loads best hyperparameters (best_params.json)
- Retrains one LightGBM per target on the TRAIN split with best hyperparams
- Evaluates on TEST split using evaluation_metrics.compute_* functions
- Saves: predictions CSV, metrics JSON, retrained model pkls, training_summary.txt
"""

#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd
import lightgbm as lgb
import joblib
import json
from pathlib import Path
import numpy as np
from evaluation_metrics import compute_classification_metrics, compute_regression_metrics

# -------------------------------------------------------------
# Directories and Files
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Input directories
DATA_PATH = SCRIPT_DIR.parent.parent / "data" / "processed_data" / "news2_features_patient.csv"
SPLITS_PATH = SCRIPT_DIR.parent / "ml_models_tcn" / "deployment_models" / " preprocessing"/ "patient_splits.json"
DEPLOY_DIR = SCRIPT_DIR.parent / "ml_models_lightgbm" / "deployment_models" 
HYPERPARAMS_PATH = SCRIPT_DIR.parent / "ml_models_lightgbm" / "hyperparameter_tuning_runs" / "best_params.json"

# Output directory
RESULTS_DIR = SCRIPT_DIR.parent.parent / "lightgbm_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output files
PREDICTIONS_CSV = RESULTS_DIR / "lightgbm_predictions.csv"
METRICS_JSON = RESULTS_DIR / "lightgbm_metrics.json"
SUMMARY_TXT = RESULTS_DIR / "training_summary.txt"

# -------------------------------------------------------------
# Load hyperparameters
# -------------------------------------------------------------

with open(HYPERPARAMS_PATH, "r") as f:
    best_params = json.load(f)

# Ensure expected keys
TARGETS = ["max_risk", "median_risk", "pct_time_high"]
for t in TARGETS:
    if t not in best_params:
        raise KeyError(f"Missing hyperparameters for target '{t}' in {HYPERPARAMS_PATH}")

# -------------------------------------------------------------
# Load patient-level features
# -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

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

# -------------------------------------------------------------
# Train & Evaluate LightGBM
# -------------------------------------------------------------
targets = [
    ("max_risk", "classification"),
    ("median_risk", "classification"),
    ("pct_time_high", "regression")
]

results = {}
preds_list = []


for target, task_type in targets:
    print(f"[INFO] Training {task_type} for target {target}")
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Prepare targets
    if target == "max_risk":
        y_train = train_df["max_risk_binary"].to_numpy()
        y_test = test_df["max_risk_binary"].to_numpy()
    elif target == "median_risk":
        y_train = train_df["median_risk_binary"].to_numpy()
        y_test = test_df["median_risk_binary"].to_numpy()
    else:
        y_train = train_df["pct_time_high"].to_numpy()
        y_test = test_df["pct_time_high"].to_numpy()


    # Train
    params = best_params[target]
    model = train_lightgbm(X_train, y_train, task_type, params)

    path = output_dir / f"{target}_evaluation_model.pkl"
    # Save model
    model_file = save_model(model, RESULTS_DIR, target)

    # --- Helper fucntion ---
    def train_lightgbm(X, y, task_type, params):
        if task_type == "classification":
            model = lgb.LGBMClassifier(**params, random_state=42, class_weight="balanced")
        else:
            model = lgb.LGBMRegressor(**params, random_state=42)
        model.fit(X, y)
        return model

    # Predict and evaluate
    if task_type == 'classification':
        y_prob = model.predict_proba(X_test)[:, 1]
        preds_dict[target] = {"y_true": y_test, "prob": y_prob, "y_pred_label": y_pred_label}
    else:
        y_pred = model.predict(X_test)
        preds_dict[target] = {"y_true": y_test, "y_pred": y_pred}

    results[target] = {
        "metrics": metrics,
        "model_file": str(model_file),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "params_used": params
    }

    print(f"[INFO] {target} metrics: {metrics}")

# -------------------------------------------------------------
# Save All Predictions (One Row per Subject)
# -------------------------------------------------------------
"""
- Saves all true and predicted values from LightGBM test set inference.
- Order matches exactly with X_test / y_test (deterministic split, no CV).
- Includes classification (max, median) and regression outputs.
"""

print("\n[INFO] Saving LightGBM predictions...")

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
# Compute Metrics 
# -------------------------------------------------------------
metrics_max = compute_classification_metrics(preds_df["y_true_max"], preds_df["prob_max"])
metrics_median = compute_classification_metrics(preds_df["y_true_median"], preds_df["prob_median"])
metrics_reg = compute_regression_metrics(preds_df["y_true_reg"], preds_df["y_pred_reg"])

all_metrics = {
    "max_risk": metrics_max,
    "median_risk": metrics_median,
    "pct_time_high": metrics_reg,
}

METRICS_JSON = RESULTS_DIR / "lightgbm_metrics.json"
with open(METRICS_JSON, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"[INFO] Metrics saved → {METRICS_JSON}")

# -------------------------------------------------------------
# Save concise evaluation summary
# -------------------------------------------------------------
SUMMARY_TXT = RESULTS_DIR / "training_summary.txt"

with open(SUMMARY_TXT, "w") as f:
    f.write("=== LightGBM Evaluation Summary ===\n")
    f.write(f"Data source: {DATA_PATH.name}\n")
    f.write(f"Splits: {len(train_df)} train, {len(test_df)} test patients\n\n")

    for target in TARGETS:
        f.write(f"--- Target: {target} ---\n")

        # Select correct metrics key
        test_metrics = results[target]["metrics"]

        f.write("Test set metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nBest hyperparameters:\n")
        for k, v in best_params[target].items():
            f.write(f"  {k}: {v}\n")

        f.write("\n")

print(f"[INFO] Training summary saved to {SUMMARY_TXT}")
print("[INFO] DONE")