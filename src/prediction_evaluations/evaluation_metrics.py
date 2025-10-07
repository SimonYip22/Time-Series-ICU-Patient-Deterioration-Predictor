"""
evaluation_metrics.py

Title: Centralised Metric Computation Utilities

Summary:
- Provides reusable functions for computing standard evaluation metrics for:
    1. Binary classification tasks (ROC-AUC, F1, Accuracy, Precision, Recall)
    2. Regression tasks (RMSE, R²)
- Used across Phase 5 evaluation scripts for TCN, LightGBM, and NEWS2 baselines.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
)

# -------------------------------------------------------------
# Classification Metrics
# -------------------------------------------------------------
"""
Computes classification metrics for binary tasks.
Args:
- y_true (array-like): Ground-truth binary labels (0/1)
- y_prob (array-like): Predicted probabilities (0–1)
- threshold (float): Decision threshold to binarise probabilities (default = 0.5)
Returns:
- dict: containing ROC-AUC, F1, Accuracy, Precision, Recall
"""
def compute_classification_metrics(y_true, y_prob, threshold=0.5):

    # Convert to NumPy arrays (handles PyTorch tensors, lists, etc.)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Convert predicted probabilities to binary predictions
    #  → threshold=0.5 means predictions ≥0.5 are class 1, otherwise 0
    y_pred = (y_prob >= threshold).astype(int)

    # Initialise dictionary to store metrics
    metrics = {}

    # --- ROC-AUC ---
    # Measures model’s ability to rank positive vs negative cases correctly
    # Some edge cases fail (e.g. if only one class is present), so handle safely
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    # --- Other standard classification metrics ---
    # F1 = harmonic mean of precision & recall
    # Accuracy = proportion of correct predictions
    # Precision = TP / (TP + FP) — "when model predicts positive, how often correct?"
    # Recall = TP / (TP + FN) — "how many real positives did model catch?"
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    # Return all results as a dictionary
    return metrics

# -------------------------------------------------------------
# Regression Metrics
# -------------------------------------------------------------
"""
Computes regression metrics (RMSE, R²)
Args:
- y_true (array-like): Ground-truth continuous labels
- y_pred (array-like): Model-predicted continuous values
Returns:
- dict: containing RMSE and R²
"""
def compute_regression_metrics(y_true, y_pred):

    # Convert to NumPy arrays for safe computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # RMSE = sqrt(MSE) → measures average prediction error magnitude
    # R² = coefficient of determination → proportion of variance explained by the model
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }

    # Return both metrics
    return metrics