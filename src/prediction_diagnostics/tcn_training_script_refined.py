# src/prediction_diagnostics/tcn_training_script_refined.py

"""
tcn_training_script_refined.py

Title: Refined Temporal Convolutional Network (TCN) Retraining Script — Phase 4.5

Summary:
- Implements the refined retraining phase for the Temporal Convolutional Network (TCN) on patient-level ICU time-series data.
- Extends Phase 4 by introducing two controlled, data-level corrections:
    1. **Class weighting** for `median_risk_binary` to address severe class imbalance.
    2. **Log-transform** of `pct_time_high` regression targets to stabilise variance and improve generalisation.
- Retains the same model architecture, hyperparameters, and optimiser setup for scientific comparability.
- Uses PyTorch with GPU acceleration if available.
- Employs multi-task learning:
    - BCEWithLogitsLoss for classification (weighted for median head)
    - MSELoss for regression (on log-transformed target)
- Uses DataLoader for efficient mini-batch training and validation.
- Tracks and records both training and validation losses across epochs.

Key Features:
- Early stopping (based on validation loss)
- Learning rate scheduling (ReduceLROnPlateau)
- Full reproducibility via fixed random seeds across Python, NumPy, and PyTorch.
- Deterministic behaviour (CuDNN disabled benchmarking)
- Automatic saving of all model artefacts, configuration, and training history; ensures reproducibility, traceability, and interpretability.

Outputs (Phase 4.5):
1. **Trained model weights** → Best-performing model (lowest validation loss) → `trained_models_refined/tcn_best_refined.pt`
2. **Training configuration** → hyperparameters, architecture, optimiser, etc. → `trained_models_refined/config_refined.json`
3. **Training history** → train/validation losses per epoch → `trained_models_refined/training_history_refined.json`
4. **Console logs**:
    - Epoch-wise training and validation loss tracking
    - pos_weight computation for class imbalance
    - Early stopping trigger (if activated)
    - Verification of target tensor integrity and shapes

Purpose:
- Provides a fully controlled and scientifically validated retraining pipeline for temporal deep learning in ICU data.
- Ensures that metric improvements arise solely from targeted data corrections (class weighting and log-transform), not from architectural or hyperparameter changes.
- Establishes a reproducible, traceable baseline for downstream model comparison and evaluation.
"""
# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import torch                                                # core PyTorch library
import torch.nn as nn                                       # neural network modules                    
from torch.utils.data import TensorDataset                  # wraps tensors (features + masks + targets) into a dataset that PyTorch can iterate over.
from torch.utils.data import DataLoader                     # handles batching, shuffling, and parallel loading.
import sys                                                  
from pathlib import Path                                    # handles filesystem paths in a clean, cross-platform way.
import pandas as pd                                         # data manipulation and analysis (loading CSVs, DataFrames).
import joblib                                               # save and load scalers (StandardScaler) and preprocessing objects.                            
import json                                                 # Read/write JSON files.
import random                                               # Python built-in random module for setting seeds        
import numpy as np                                          # numerical operations, array manipulations

# -------------------------------------------------------------
# Updated Directory Structure (Phase 4.5)
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent                                # → src/prediction_diagnostics
PROJECT_ROOT = SCRIPT_DIR.parent                                            # → src/
sys.path.append(str(PROJECT_ROOT / "ml_models_tcn"))                        # ensure TCNModel can be imported
from tcn_model import TCNModel                                              # TCN architecture implementation (class we defined in tcn_model.py)

# === Input paths ===
DATA_DIR = PROJECT_ROOT / "ml_models_tcn" / "prepared_datasets"             # → src/ml_models_tcn/prepared_datasets
SCALER_DIR = PROJECT_ROOT / "ml_models_tcn" / "deployment_models" / "preprocessing"  # → src/ml_models_tcn/deployment_models/preprocessing
PATIENT_FEATURES_PATH = PROJECT_ROOT.parent / "data" / "processed_data" / "news2_features_patient.csv"  # → data/processed_data/news2_features_patient.csv

# === Output paths Updated (Refined Phase 4.5) ===
MODEL_SAVE_DIR = SCRIPT_DIR / "trained_models_refined"                      # → src/prediction_diagnostics/trained_models_refined
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

best_model_path = MODEL_SAVE_DIR / "tcn_best_refined.pt"
config_path = MODEL_SAVE_DIR / "config_refined.json"
history_path = MODEL_SAVE_DIR / "training_history_refined.json"

print(f"[INFO] All refined training outputs will be saved to: {MODEL_SAVE_DIR}")

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# All key parameters for training 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available, else CPU
BATCH_SIZE = 32                                             # Number of patient sequences processed in one forward/backward pass.
EPOCHS = 50                                                 # Number of complete passes through the training dataset. Multiple epochs → the model gradually adjusts weights to reduce loss.
LR = 1e-3                                                   # Learning rate for the Adam optimizer. Controls how much to change the model in response to estimated error each time the model weights are updated.
EARLY_STOPPING_PATIENCE = 7                                 # Monitors a validation metric (e.g., ROC-AUC or RMSE). If metric does not improve for 7 epochs, training stops early to avoid overfitting.

# -------------------------------------------------------------
# Reproducibility / Random Seed
# -------------------------------------------------------------
# Ensures that random operations (data shuffling, weight initialisation) produce the same results
# Final trained model and results can be consistently reproduced.
SEED = 42
random.seed(SEED) # fixes randomness from Python’s built-in random number generator (e.g. when DataLoader shuffles batches or any code internally uses random.choice).
np.random.seed(SEED) # ensures NumPy operations that use randomness (e.g. any preprocessing or augmentation) are deterministic.

# PyTorch seeds ensure weight initialisation, dropout, and any random sampling from PyTorch are reproducible.
torch.manual_seed(SEED)              # seeds CPU operations
torch.cuda.manual_seed(SEED)         # seeds current GPU
torch.cuda.manual_seed_all(SEED)     # seeds all GPUs if multiple

# Enforce deterministic behaviour in PyTorch
torch.backends.cudnn.deterministic = True  # forces deterministic convolution algorithms
torch.backends.cudnn.benchmark = False     # disables autotuner for benchmarking → ensures reproducibility

# -------------------------------------------------------------
# Define configuration dictionary before training (Refined Phase 4.5)
# -------------------------------------------------------------
config_data = {
    "phase": "4.5 - Refined Retraining",
    "device": DEVICE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "model_architecture": {
        "num_channels": [64, 64, 128],
        "head_hidden": 64,
        "kernel_size": 3,
        "dropout": 0.2
    },
    "optimizer": "Adam",
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 3,
        "factor": 0.5
    },
    "loss_functions": {
        "max_risk": "BCEWithLogitsLoss (unweighted)",
        "median_risk": "BCEWithLogitsLoss(pos_weight)",
        "pct_time_high": "MSELoss (on log1p-transformed target)"
    },
    "data_transformations": {
        "regression_target": "log1p(y) applied during training; expm1(y_pred) at inference",
        "class_weighting": "dynamic pos_weight computed from training data (num_neg / num_pos)"
    },
    "data_paths": {
        "prepared_datasets": str(DATA_DIR),
        "scaler_dir": str(SCALER_DIR),
        "patient_features_path": str(SCRIPT_DIR.parent.parent / 'data/processed-data/news2_features_patient.csv'),
        "splits_json": str(SCALER_DIR / "patient_splits.json")
    },
    "model_save_dir": str(MODEL_SAVE_DIR),
    "outputs": {
        "weights": str(MODEL_SAVE_DIR / "tcn_best_refined.pt"),
        "history": str(MODEL_SAVE_DIR / "training_history_refined.json"),
        "config": str(MODEL_SAVE_DIR / "config_refined.json")
    },
    "notes": "Refined retraining adds class weighting (median_risk) and log-transform (pct_time_high); all else unchanged for controlled comparison."
}

print("[INFO] Base configuration initialised.")

# -------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------
# ---- Function to Load Dataset Splits ----
def load_split(split_name):
    x = torch.load(DATA_DIR / f"{split_name}.pt")           # Load sequence tensor 
    mask = torch.load(DATA_DIR / f"{split_name}_mask.pt")   # Load corresponding mask tensor
    return x, mask                                          # The function returns both x and mask for a given split

# ---- Load the Dataset Splits using load_split() ----
# x_train, x_val, x_test → input sequences for training, validation, and testing.
# mask_train, mask_val, mask_test → masks to ignore padded timesteps during model computation and loss.
x_train, mask_train = load_split("train")
x_val, mask_val = load_split("val")
x_test, mask_test = load_split("test")

# ---- Load Target Column Names ----
# Load target column names from padding_config.json to identify which columns are targets
with open(SCALER_DIR / "padding_config.json") as f:
    config = json.load(f)

target_cols = ["max_risk_binary", "median_risk_binary", "pct_time_high"] # target columns in patient-level dataframe (used the binary versions for classification tasks)

# -------------------------------------------------------------
# Load true patient-level targets (Refined Phase 4.5)
# -------------------------------------------------------------
"""
This step rebuilds the outcome tensors with:
- Class weighting computed from training split (for median_risk)
- Log1p transformation applied to regression target (pct_time_high)
"""

PATIENT_FEATURES_PATH = SCRIPT_DIR.parent.parent / "data/processed_data/news2_features_patient.csv"
SPLITS_PATH = SCALER_DIR / "patient_splits.json"

# Load patient-level outcomes
patient_df = pd.read_csv(PATIENT_FEATURES_PATH).set_index("subject_id")


# --- Binary target reconstruction (same logic as Phase 4) ---
patient_df["max_risk_binary"] = patient_df["max_risk"].apply(lambda x: 1 if x > 2 else 0) # collapse [0,1,2]=0; [3]=1 (0=not-high, 1=high).
patient_df["median_risk_binary"] = patient_df["median_risk"].apply(lambda x: 1 if x == 2 else 0) # collapse [0,1]=0; [2]=1 (0=low, 1=medium).

# --- Load consistent patient splits ---
with open(SPLITS_PATH) as f:
    patient_splits = json.load(f)

# ---- Build target tensors for each split ----
train_ids = patient_splits["train"]
val_ids = patient_splits["val"]
test_ids = patient_splits["test"]

# --- Helper: Build target tensors (with regression log-transform) ---
def get_targets(split_ids, apply_log=False):
    df_split = patient_df.loc[split_ids, target_cols].copy()
    if apply_log:
        df_split["pct_time_high"] = np.log1p(df_split["pct_time_high"])  # log-transform regression target
        
        # Debug/info print to confirm that regression target has been successfully log-transformed.
        print(f"[INFO] Log-transform applied to regression target: min={df_split['pct_time_high'].min():.3f}, max={df_split['pct_time_high'].max():.3f}")
    return (
        torch.tensor(df_split["max_risk_binary"].values, dtype=torch.float32),
        torch.tensor(df_split["median_risk_binary"].values, dtype=torch.float32),
        torch.tensor(df_split["pct_time_high"].values, dtype=torch.float32),
    )

# --- Create tensors for each split ---
# get_targets() returns 3 tensors per split (max, median, regression targets), this gets repeated for train, val, test splits, now we have tensors for all 3 targets in all 3 splits
# apply_log=True applies log1p transform to regression target 
y_train_max, y_train_median, y_train_reg = get_targets(train_ids, apply_log=True)
y_val_max, y_val_median, y_val_reg = get_targets(val_ids, apply_log=True)
y_test_max, y_test_median, y_test_reg = get_targets(test_ids, apply_log=True)

# --- Compute class weights (for median risk) ---
# Directly fixes BCE loss imbalance by scaling the minority class contribution.
# pos_weight = num_neg / num_pos
# This makes the model pay more attention to the minority class (medium risk) during training.
pos_weight = (y_train_median == 0).sum() / (y_train_median == 1).sum()
pos_weight = pos_weight.to(DEVICE)  # move to GPU if available
print(f"[INFO] Computed pos_weight for median_risk = {pos_weight:.3f}")

# --- Verify tensor shapes ---
# Each tensor shape should match number of patients in each split
print("[INFO] Targets loaded (Refined Phase 4.5):")
print(" - train:", y_train_max.shape, y_train_median.shape, y_train_reg.shape) # 70
print(" - val:", y_val_max.shape, y_val_median.shape, y_val_reg.shape) # 15
print(" - test:", y_test_max.shape, y_test_median.shape, y_test_reg.shape) # 15

# -------------------------------------------------------------
# TensorDatasets & DataLoaders
# -------------------------------------------------------------
"""
TensorDataset is a PyTorch utility.
- It bundles multiple tensors (inputs + masks + labels) together so that they can be indexed as a unit.
DataLoader 
- Takes the dataset and feeds it to the model in mini-batches during training/validation.
Why we need this:
- Training on the entire dataset at once (full batch) is too memory-heavy.
- Mini-batching allows: More efficient GPU/CPU usage, smoother gradient estimates (stochastic gradient descent), the possibility to shuffle → avoids learning artefacts from data order.
"""

# Each sample (patient) in train_dataset is a tuple containing 5 tensors: (input sequence, mask, max target, median target, regression target)
train_dataset = TensorDataset(x_train, mask_train, y_train_max, y_train_median, y_train_reg)
val_dataset = TensorDataset(x_val, mask_val, y_val_max, y_val_median, y_val_reg)

# batch_size=BATCH_SIZE: 32 → each training step sees 32 patients at once.
# shuffle=True: randomises order each epoch → prevents overfitting to sequence of data.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # For validation we set shuffle=False (keep stable evaluation).

# -------------------------------------------------------------
# Model
# -------------------------------------------------------------
# build TCN with 3 convolutional blocks (64 → 64 → 128 filters), followed by a dense head of size 64, and made it ready to train on GPU/CPU.
# x_train is of shape (num_patients, max_seq_len, num_features), shape[2] is num_features (171)
# num_features=feature_dim into mdoel so it knows dimensions of each input vector
feature_dim = x_train.shape[2]
model = TCNModel(num_features=feature_dim,      # the size of the input feature vector at each timestep (171).
                 num_channels=[64, 64, 128],    # defines residual block architecture
                 head_hidden=64                 # Size of the dense hidden layer after pooling.
                 ).to(DEVICE)                   # moves model to GPU if available, else CPU

# -------------------------------------------------------------
# Training Logic Setup: Updated Losses & Optimiser (Phase 4.5)
# -------------------------------------------------------------
# Each task needs its own loss function:
criterion_max = nn.BCEWithLogitsLoss()
criterion_median = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # weighted BCE
criterion_reg = nn.MSELoss()  # still MSE, but regression targets now log-transformed

# Optimiser
# Adam = Adaptive Moment Estimation, updates all trainable weights of model based on computed gradients from backpropagation.
# LR = 1e-3, a common starting learning rate for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Schedular monitora validation metric (val loss here). If no improvement for 3 epochs, reduce LR by factor of 0.5.
# Helps model converge better by taking smaller steps when progress stalls.
# mode="min" → because we want to minimise loss.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# -------------------------------------------------------------
# Training Loop with Validation
# -------------------------------------------------------------
best_val_loss = float("inf")        # validation loss starts as infinity, so any first real validation loss will be smaller and count as “best”.
patience_counter = 0                # counts how many consecutive epochs the validation loss has not improved.

# Store training progress for visualisation
train_losses = []
val_losses = []

# ---- Epoch Loop ----
# epoch = one full pass over the training dataset.
# run multiple epochs (10–100+) so the model gradually learns.
for epoch in range(1, EPOCHS + 1):
    # set model to training mode (activates dropout, batch norm updates, etc.).
    model.train()
    total_loss = 0.0
    # ---- Mini-batch Loop ----
    # iterate over mini-batches from train_loader
    for x_batch, mask_batch, y_max, y_median, y_reg in train_loader:
        # Move data to GPU/CPU
        x_batch = x_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        y_max = y_max.to(DEVICE)
        y_median = y_median.to(DEVICE)
        y_reg = y_reg.to(DEVICE)

        optimizer.zero_grad()                     # reset old gradients 
        outputs = model(x_batch, mask_batch)      # forward pass: get model predictions for the batch

        # Calculate losses for 3 tasks
        loss_max = criterion_max(outputs["logit_max"], y_max)
        loss_median = criterion_median(outputs["logit_median"], y_median)
        loss_reg = criterion_reg(outputs["regression"], y_reg)

        # Combine losses (simple sum, equal weighting)
        loss = loss_max + loss_median + loss_reg

        # Backward pass (backpropagation)
        loss.backward()                                                     # compute gradient of the loss with respect to every trainable parameter in the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping to prevent exploding gradients
        optimizer.step()                                                    # update model weights (based on computed gradients), moving them in a direction that decreases the loss.
        
        # Accumulate loss for averaging  
        # ensures you sum up loss weighted by batch size.
        # You get the average training loss across all patients.
        total_loss += loss.item() * x_batch.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    # ---- Validation phase ----
    # set model to evaluation mode (disables dropout, batch norm updates, etc.)
    # Same as training, but no gradients or optimiser step.
    # You only compute validation loss → tells you how well the model generalises to unseen patients.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():             # disable gradient computation (saves memory, speeds up)
        for x_batch, mask_batch, y_max, y_median, y_reg in val_loader:
            x_batch = x_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)
            y_max = y_max.to(DEVICE)
            y_median = y_median.to(DEVICE)
            y_reg = y_reg.to(DEVICE)

            outputs = model(x_batch, mask_batch)
            loss_max = criterion_max(outputs["logit_max"], y_max)
            loss_median = criterion_median(outputs["logit_median"], y_median)
            loss_reg = criterion_reg(outputs["regression"], y_reg)

            loss = loss_max + loss_median + loss_reg
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    # ---- Scheduler step ----
    # If avg_val_loss hasn’t improved for a while, it automatically lowers the learning rate to make training more fine-grained.
    scheduler.step(avg_val_loss)

    # ---- Record losses for later visualisation ----
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # ---- Progress reporting ----
    # track learning across epochs.
    # Normally, train loss ↓ and val loss ↓ if model learns well.
    # If train ↓ but val ↑, that’s overfitting.
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # --- Save model to new Phase 4.5 folder ---
        torch.save(model.state_dict(), best_model_path) # If validation loss improves → save model, final model state will be best one.
    else:
        patience_counter += 1                               # No improvement → increment counter.
        if patience_counter >= EARLY_STOPPING_PATIENCE:     # If no improvement for 7 epochs → stop training (saves time, avoids overfitting).
            print(f"Early stopping at epoch {epoch}")
            break

# -------------------------------------------------------------
# Update configuration dictionary after training completes (Refined Phase 4.5)
# -------------------------------------------------------------
config_data["pos_weight_median_risk"] = float(pos_weight.cpu().item()) # records the actual numeric value of pos_weight that was computed dynamically from training set.
config_data["final_val_loss"] = float(best_val_loss) # records the best validation loss achieved during training, how well the model fit the validation data

# Confirms where the files actually went 
config_data["outputs_confirmed"] = { 
    "best_model": str(best_model_path),
    "training_history": str(history_path),
    "config": str(config_path)
}

# Save final config
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)
print(f"[INFO] Final refined training configuration saved to {config_path}")

# -------------------------------------------------------------
# Save updated training history for visualisation (Phase 4.5)
# -------------------------------------------------------------
# Save training loss curves
with open(history_path, "w") as f:
    json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=4)
print(f"[INFO] Refined training history saved to {history_path}")

# Model weights already saved in training loop at best epoch
print(f"[INFO] Best refined model saved to {best_model_path}")