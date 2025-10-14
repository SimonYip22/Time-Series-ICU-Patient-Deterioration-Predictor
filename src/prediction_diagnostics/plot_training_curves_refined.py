"""
plot_training_curves_refined.py

Title: Plotting Training vs Validation Loss Curves (per epoch) on Refined Model Outputs

Summary:
- Loads recorded training and validation losses from both:
    - Phase 4 → `ml_models_tcn/trained_models/training_history.json`
    - Phase 4.5 → `prediction_diagnostics/trained_models_refined/training_history_refined.json`
- Generates two plots:
    1. Refined (Phase 4.5) learning curve → training vs validation loss.
    2. Baseline vs Refined comparison → overlay showing convergence differences.
- Visualises:
    - Learning progression and stability
    - Early stopping point and best validation epoch
    - Onset of overfitting (if present)
- Annotates minima (best validation loss) for both phases.
- Prints console confirmations with save locations for reproducibility.

Outputs:
- `loss_plots/loss_curve_refined.png` → standalone refined training/validation loss
- `loss_plots/loss_curve_comparison.png` → overlaid baseline vs refined comparison
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
SCRIPT_DIR = Path(__file__).resolve().parent

# Phase 4 paths
HISTORY_ORIGINAL = SCRIPT_DIR.parent / "ml_models_tcn" / "trained_models" / "training_history.json"

# Phase 4.5 paths
HISTORY_REFINED = SCRIPT_DIR / "trained_models_refined" / "training_history_refined.json"

# Output directory
PLOT_DIR = SCRIPT_DIR / "loss_plots"
PLOT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Load Training Histories
# -------------------------------------------------------------
# The training scripts stored losses into a JSON file (train/val loss per epoch).
# We load them here to visualise how loss changed over time for both Phase 4 and Phase 4.5.
with open(HISTORY_ORIGINAL) as f:
    history_original = json.load(f)
with open(HISTORY_REFINED) as f:
    history_refined = json.load(f)

# Original (Phase 4): Extract train and validation losses as lists of floats
train_loss_original = history_original["train_loss"]
val_loss_original = history_original["val_loss"]

# Refined (Phase 4.5): Extract train and validation losses as lists of floats
train_loss_refined = history_refined["train_loss"]
val_loss_refined = history_refined["val_loss"]


# Original (Phase 4): Identify best (lowest) validation loss epoch — the point of best generalisation
best_epoch_original = val_loss_original.index(min(val_loss_original))
best_val_original = min(val_loss_original)

# Original (Phase 4.5): Identify best (lowest) validation loss epoch — the point of best generalisation
best_epoch_refined = val_loss_refined.index(min(val_loss_refined))
best_val_refined = min(val_loss_refined)


# -------------------------------------------------------------
# Plot 1 – Refined Model (Phase 4.5 only)
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))  # Set figure size for readability
plt.plot(train_loss_refined, label="Train Loss", linewidth=2)           # Plot training curve
plt.plot(val_loss_refined, label="Validation Loss", linewidth=2)        # Plot validation curve
plt.xlabel("Epoch")                                                     # X-axis label
plt.ylabel("Loss (MSE log-space)")                                      # Y-axis label
plt.title("Training vs Validation Loss per Epoch (Refined TCN Model Run)", pad=40)  # Title for clarity
plt.text(0.5, 1.02,
         "Note: Regression component uses log-transformed targets; loss values not numerically comparable to baseline (Phase 4).",
         ha="center", va="bottom", fontsize=9, color="gray", transform=plt.gca().transAxes)
plt.legend()                                                            # Show legend
plt.grid(True, linestyle="--", alpha=0.5)                               # Add grid for easier interpretation
plt.tight_layout()                                                      # Adjust layout to prevent cutoff labels

# -------------------------------------------------------------
# Plot 1 –Add Key Annotations for interpretability (Phase 4.5)
# -------------------------------------------------------------
# Draw a vertical red dashed line at the best validation epoch
plt.axvline(best_epoch_refined, color="red", linestyle="--", alpha=0.7, label="Best (Early Stop)")

# Add a red dot marking the exact lowest validation loss point
plt.scatter(best_epoch_refined, best_val_refined, color="red", s=60, zorder=5)

# Text annotation showing the epoch number and loss value
plt.text(best_epoch_refined + 0.3, best_val_refined + 0.02,
         f"Best epoch = {best_epoch_refined}\nVal loss = {best_val_refined:.3f}",
         color="red", fontsize=10, verticalalignment="bottom")

# Optional annotation: if more epochs exist after best epoch, show "overfitting region"
if len(val_loss_refined) > best_epoch_refined + 2:
    plt.text(best_epoch_refined + 2, max(val_loss_refined) * 0.95,
             "↑ Overfitting region",
             color="gray", fontsize=10, alpha=0.8)
    
# -------------------------------------------------------------
# Plot 1 – Save Refined Plot (Phase 4.5)
# -------------------------------------------------------------
# Save refined figure in high resolution
plt.savefig(PLOT_DIR / "loss_curve_refined.png", dpi=300)

# Optionally display plot interactively
plt.show()

# Close figure to free memory
plt.close()

# Console confirmation
print("[INFO] Saved refined loss plot → loss_plots/loss_curve_refined.png")
    
# -------------------------------------------------------------
# Plot 2 – Comparison: Phase 4 vs Phase 4.5
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Phase 4 curves (dashed, lighter)
plt.plot(train_loss_original, label="Baseline — Train (raw)", linestyle="--", color="blue", alpha=0.6)
plt.plot(val_loss_original, label="Baseline — Val (raw)", linestyle="--", color="orange", alpha=0.6)

# Phase 4.5 curves (solid, darker)
plt.plot(train_loss_refined, label="Refined — Train (log)", linewidth=2, color="blue")
plt.plot(val_loss_refined, label="Refined — Val (log)", linewidth=2, color="orange")

plt.xlabel("Epoch")
plt.ylabel("Composite Loss (raw vs log-space — compare trends only)")
plt.title("TCN Training vs Validation Loss — Baseline vs Refined (Trend Comparison Only)", pad=40)
plt.text(0.5, 1.02,
         "Note: Regression losses differ by scale (Phase 4 raw vs Phase 4.5 log-space).",
         ha="center", va="bottom", fontsize=9, color="gray", transform=plt.gca().transAxes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# -------------------------------------------------------------
# Plot 2 - Add Key Annotations for interpretability (Comparison)
# -------------------------------------------------------------
# Annotate minima
plt.scatter(best_epoch_original, best_val_original, color="orange", s=50, zorder=5)
plt.scatter(best_epoch_refined, best_val_refined, color="red", s=50, zorder=5)
plt.text(best_epoch_original + 0.3, best_val_original + 0.02,
         f"Baseline best\n(Epoch {best_epoch_original})",
         color="orange", fontsize=9, verticalalignment="bottom")
plt.text(best_epoch_refined + 0.3, best_val_refined + 0.02,
         f"Refined best\n(Epoch {best_epoch_refined})",
         color="red", fontsize=9, verticalalignment="bottom")

# Reserve just enough space below the plot (minimal padding)
plt.subplots_adjust(bottom=0.5)

# Add the disclaimer just below the x-axis, centred and compact
plt.figtext(
    0.5, 0.05,  # slightly lower and tighter
    "Disclaimer: Phase 4.5 regression losses are log-transformed; visual overlay is for trend comparison only.",
    ha="center", va="bottom", fontsize=9, color="gray"
)

# -------------------------------------------------------------
# Plot 2 – Save Comparison Plot (Phase 4 vs Phase 4.5)
# -------------------------------------------------------------
# Adjust layout to leave space for disclaimer
plt.tight_layout(rect=[0.02, 0.14, 0.98, 0.92])  

# Save combined plot
plt.savefig(PLOT_DIR / "loss_curve_comparison.png", dpi=300)
plt.show()
plt.close()

print("[INFO] Saved combined comparison plot → loss_plots/loss_curve_comparison.png")



