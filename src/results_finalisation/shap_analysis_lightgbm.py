#src/results_finalisation/shap_analysis_lightgbm.py

"""
shap_analysis_lightgbm.py

Purpose:
- Performs final interpretability analysis for LightGBM models trained in Phase 5.  
- Computes SHAP (SHapley Additive exPlanations) values to quantify individual feature contributions and overall feature importance across the three prediction targets.
Inputs:
- `news2_features_patient.csv` — patient-level aggregated features
- `patient_splits.json` — defines 70-patient training set used for SHAP computation
- `{target}_retrained_model.pkl` — final LightGBM models per target
Workflow:
1. Load final LightGBM `.pkl` models from `src/prediction_evaluations/lightgbm_results/`.
2. Load patient-level features (`news2_features_patient.csv`) and training split from `patient_splits.json`.
3. Use `TreeExplainer` to compute SHAP values for all input features.
4. Aggregate mean absolute SHAP values to produce per-feature importance.
5. Save:
   - `{target}_shap_summary.csv`: numeric SHAP importances (all features)
   - `{target}_shap_summary.png`: bar plot of top 10 influential features
Outputs:
- Stored in `src/results_finalisation/interpretability_lightgbm/`:
    - `{target}_shap_summary.csv` — numerical SHAP values and mean absolute importance
    - `{target}_shap_summary.png` — summary bar plot (top 10 features)
"""

#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd
import numpy as np

# Model interpretability – computes SHAP values (feature attributions)
import shap

import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import json

#-------------------------------------------------------------
# Directories
#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Input directories 
DATA_PATH = SCRIPT_DIR.parent.parent / "data/processed_data/news2_features_patient.csv"
MODEL_DIR = SCRIPT_DIR.parent.parent / "src/prediction_evaluations/lightgbm_results"
SPLITS_PATH = SCRIPT_DIR.parent.parent / "src/ml_models_tcn/deployment_models/preprocessing/patient_splits.json"

# Output directories
OUTPUT_DIR = SCRIPT_DIR / "interpretability_lightgbm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#-------------------------------------------------------------
# Load patient features + training split
#-------------------------------------------------------------
# Load patient features CSV 
df = pd.read_csv(DATA_PATH)

# Load patient split IDs
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

# Load training split IDs
train_ids = splits["train"]
# Load feature columns for the training patients
train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)

#-------------------------------------------------------------
# Recreate binary targets + define feature columns 
#-------------------------------------------------------------
# Recreate binary targets consistent with previous phases
train_df["max_risk_binary"] = train_df["max_risk"].apply(lambda x: 1 if x == 3 else 0)
train_df["median_risk_binary"] = train_df["median_risk"].apply(lambda x: 1 if x == 2 else 0)

# Exclude non-feature columns (exclude ID + target columns)
exclude_cols = ["subject_id", "max_risk", "median_risk", "pct_time_high",
                "max_risk_binary", "median_risk_binary"]
# feature_cols now contains only model input features
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

#-------------------------------------------------------------
# Loop over targets and compute SHAP
#-------------------------------------------------------------
TARGETS = ["max_risk", "median_risk", "pct_time_high"]

# Loop each target by loading the correct .pkl model 
for target in TARGETS:
    print(f"Computing SHAP values for target: {target}")

    # Load LightGBM model for specififc target
    model_path = MODEL_DIR / f"{target}_retrained_model.pkl"
    model = joblib.load(model_path)

    # Select features and target values
    X_train = train_df[feature_cols]

    # --- Compute SHAP values safely using TreeExplainer ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # --- Handle classifiers vs regressors correctly ---
    if isinstance(shap_values, list):  
        # Binary classification → two arrays (class 0, class 1), we only want the array for class 1
        shap_array = shap_values[1]
    else:
        # Regression → single array
        shap_array = shap_values

    print(f"[INFO] Target: {target}")
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] SHAP array shape: {shap_array.shape}")

    #---------------------------------------------------------
    # Compute mean absolute SHAP importance
    #---------------------------------------------------------
    # Compute mean for target array
    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    print(f"[INFO] Mean |SHAP| sum: {mean_abs_shap.sum():.4f}")

    # Create DataFrame 
    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    # Save numeric SHAP importance
    shap_importance.to_csv(OUTPUT_DIR / f"{target}_shap_summary.csv", index=False)

    # Debugging
    if target in ["max_risk", "median_risk"]:
        preds = model.predict_proba(X_train)[:, 1]  # positive class
    else:
        preds = model.predict(X_train)              # regression: predicted values
    print("Unique predicted values:", np.unique(preds))

    #---------------------------------------------------------
    # Plot Top 10 SHAP feature importances
    #---------------------------------------------------------
    # Plot Top 10 SHAP features for target
    top_features = shap_importance.head(10)
    plt.figure(figsize=(10,6))
    plt.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value| (feature impact)")
    plt.title(f"Top 10 SHAP Features for {target}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{target}_shap_summary.png")
    plt.close()

print("SHAP analysis complete.")

