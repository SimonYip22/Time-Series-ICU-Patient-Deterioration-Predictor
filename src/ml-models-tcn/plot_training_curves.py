"""
plot_training_curves.py

Title: Plotting Training vs Validation Loss Curves (per epoch)

Summary:
- Loads the recorded training and validation losses from `training_history.json`.
- Plots the loss curves across epochs to visualise:
    - Learning progression
    - Convergence behaviour
    - Early stopping point and potential overfitting onset.
- Provides annotated insights into where the model performed best and began overfitting.
- Helps assess whether the model converged smoothly and generalised well.

Output:
- `plots/loss_curve.png` — high-resolution PNG comparing training and validation losses.
- Console confirmation message with save location.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import json                     # To load saved training history (losses per epoch)
import matplotlib.pyplot as plt  # For visualising learning curves
from pathlib import Path         # For safe and cross-platform file paths

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent                # Get current script directory
HISTORY_PATH = SCRIPT_DIR / "trained_models" / "training_history.json"  # Path to saved training history

# -------------------------------------------------------------
# Load Training History
# -------------------------------------------------------------
# The training script stored losses into a JSON file (train/val loss per epoch).
# We load them here to visualise how loss changed over time.
with open(HISTORY_PATH) as f:
    history = json.load(f)

# Extract train and validation losses as lists of floats
train_loss = history["train_loss"]
val_loss = history["val_loss"]

# Identify best (lowest) validation loss epoch — the point of best generalisation
best_epoch = val_loss.index(min(val_loss))
best_val = min(val_loss)

# -------------------------------------------------------------
# Plot Training vs Validation Loss
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))  # Set figure size for readability
plt.plot(train_loss, label="Train Loss", linewidth=2)      # Plot training curve
plt.plot(val_loss, label="Validation Loss", linewidth=2)   # Plot validation curve
plt.xlabel("Epoch")                                        # X-axis label
plt.ylabel("Loss")                                         # Y-axis label
plt.title("Training vs Validation Loss per Epoch")         # Title for clarity
plt.legend()                                               # Show legend
plt.grid(True, linestyle="--", alpha=0.5)                  # Add grid for easier interpretation
plt.tight_layout()                                         # Adjust layout to prevent cutoff labels

# -------------------------------------------------------------
# Add Key Annotations (for interpretability)
# -------------------------------------------------------------
# Draw a vertical red dashed line at the best validation epoch
plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label="Best (Early Stop)")

# Add a red dot marking the exact lowest validation loss point
plt.scatter(best_epoch, best_val, color="red", s=60, zorder=5)

# Text annotation showing the epoch number and loss value
plt.text(best_epoch + 0.3, best_val + 0.02,
         f"Best epoch = {best_epoch}\nVal loss = {best_val:.3f}",
         color="red", fontsize=10, verticalalignment="bottom")

# Optional annotation: if more epochs exist after best epoch, show "overfitting region"
if len(val_loss) > best_epoch + 2:
    plt.text(best_epoch + 2, max(val_loss) * 0.95,
             "↑ Overfitting region",
             color="gray", fontsize=10, alpha=0.8)

# -------------------------------------------------------------
# Save Plot
# -------------------------------------------------------------
# Create /plots directory if not already present
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Save figure in high resolution
plt.savefig(PLOT_DIR / "loss_curve.png", dpi=300)

# Optionally display plot interactively
plt.show()

# Close figure to free memory
plt.close()

# Console confirmation
print("[INFO] Saved plot to plots/loss_curve.png")