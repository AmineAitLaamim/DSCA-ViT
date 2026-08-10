# DSCA-ViT — Spatial Bias: Initialization vs Trained (02b)
# ============================================================
# PURPOSE:
#   Focused diagnostic: did the learned spatial bias matrix actually
#   change from its initialization during Stage 2 training?
#
#   We directly compare:
#     1. The ORIGINAL initialized spatial bias (reconstructed using the
#        exact DSCA-ViT implementation logic)
#     2. The TRAINED spatial bias stored in the Stage 2 checkpoint
#
#   This is NOT a general attention-analysis notebook.
#   We do NOT retrain, do NOT modify the model/checkpoint, and do NOT
#   use test labels for any optimization.
#
# CHECKPOINT (exactly):
#   /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth
#
# OUTPUT:
#   .../DSCA_ViT/Results/Spatial_Bias_Initialization_vs_Trained/
# ============================================================

# ============================================================
# Cell 1 — Mount Google Drive + Clone / Pull Repository
# ============================================================

from google.colab import drive
import os

drive.mount("/content/drive")

if os.path.exists("/content/drive/MyDrive"):
    print("✅ Google Drive mounted successfully.")
else:
    raise RuntimeError("❌ Google Drive was not mounted correctly.")

import subprocess

REPO_URL = "https://github.com/AmineAitLaamim/DSCA-ViT.git"
REPO_DIR = "/content/DSCA-ViT"

if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("✅ Repository cloned.")
else:
    print("Pulling latest changes...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print("✅ Repository updated.")

import sys
sys.path.insert(0, REPO_DIR)

print(f"REPO_DIR: {REPO_DIR}")


# ============================================================
# Cell 2 — Install Dependencies
# ============================================================

subprocess.run(
    ["pip", "install", "timm", "pyyaml", "seaborn", "scipy", "--quiet"],
    check=True
)
print("✅ Dependencies installed.")


# ============================================================
# Cell 3 — Imports & Reproducibility
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"PyTorch Version : {torch.__version__}")
print(f"Device          : {device}")
print("=" * 60)


# ============================================================
# Cell 4 — Configuration
# ============================================================

BACKBONE_NAME   = "DSCA_ViT"
MODEL_ID        = "dsca_vit_b16"
NUM_CLASSES     = 4
IMAGE_SIZE      = 224
GRID            = 14                 # patch grid
N_PATCH         = GRID * GRID        # 196 spatial patches
NUM_TOKENS      = N_PATCH + 1        # 197 = 1 CLS + 196 patches

# Checkpoint (weights-only file from Stage 2)
CHECKPOINT_PATH = "/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth"

# Output folder
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
EXPERIMENT_DIR  = os.path.join(CHECKPOINT_ROOT, BACKBONE_NAME)
RESULTS_DIR     = os.path.join(EXPERIMENT_DIR, "Results")
OUT_DIR         = os.path.join(RESULTS_DIR, "Spatial_Bias_Initialization_vs_Trained")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Spatial Bias Diagnostic Configuration")
print("=" * 60)
print(f"Model           : {BACKBONE_NAME}")
print(f"Checkpoint      : {CHECKPOINT_PATH}")
print(f"Output Dir      : {OUT_DIR}")
print("=" * 60)


# ============================================================
# Cell 5 — Build Model + Load Stage 2 Checkpoint
# ============================================================

from models import DSCAViT

# Build the SAME architecture/configuration used when the checkpoint was trained
model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.1,
    classifier_dropout=0.1,
)
model = model.to(device)

assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found:\n{CHECKPOINT_PATH}"
print(f"✅ Loading checkpoint:\n    {CHECKPOINT_PATH}")

state = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
    print("✅ Loaded full checkpoint (model_state_dict).")
else:
    model.load_state_dict(state)
    print("✅ Loaded weights-only state_dict.")

model.eval()

counts = model.count_parameters()
print("=" * 60)
print("DSCA-ViT Parameter Summary")
print("=" * 60)
for name, count in counts.items():
    print(f"  {name:<20} : {count:>12,}")
print("=" * 60)


# ============================================================
# Cell 6 — Locate the ACTUAL Spatial Bias Parameter
# ============================================================

# Automatically inspect the model to find the spatial-bias parameter.
# We do NOT assume a name — we search for it.
bias_param = None
bias_path = None

for name, param in model.named_parameters():
    if "bias_matrix" in name or ("spatial_bias" in name and "bias" in name):
        bias_param = param
        bias_path = name
        break

if bias_param is None:
    # Fallback: search all parameters for one with shape (197, 197)
    for name, param in model.named_parameters():
        if tuple(param.shape) == (NUM_TOKENS, NUM_TOKENS):
            bias_param = param
            bias_path = name
            break

assert bias_param is not None, "Could not locate the spatial bias parameter."

print("=" * 60)
print("Spatial Bias Parameter (auto-located)")
print("=" * 60)
print(f"  Module path    : {bias_path}")
print(f"  Parameter name : {bias_path.split('.')[-1]}")
print(f"  Shape          : {tuple(bias_param.shape)}")
print(f"  requires_grad  : {bias_param.requires_grad}")
print(f"  dtype          : {bias_param.dtype}")
print(f"  device         : {bias_param.device}")
print(f"  min            : {bias_param.data.min().item():.6f}")
print(f"  max            : {bias_param.data.max().item():.6f}")
print(f"  mean           : {bias_param.data.mean().item():.6f}")
print(f"  std            : {bias_param.data.std().item():.6f}")
print("=" * 60)

# Verify the parameter is in the checkpoint
ckpt_key = bias_path  # state_dict key == module path
if isinstance(state, dict) and "model_state_dict" in state:
    ckpt_state = state["model_state_dict"]
else:
    ckpt_state = state

print("Checkpoint key verification:")
print(f"  Exact checkpoint key : {ckpt_key}")
print(f"  Key present          : {ckpt_key in ckpt_state}")
if ckpt_key in ckpt_state:
    ckpt_tensor = ckpt_state[ckpt_key]
    print(f"  Checkpoint shape     : {tuple(ckpt_tensor.shape)}")
    print(f"  Model param shape    : {tuple(bias_param.shape)}")
    print(f"  Shapes match         : {tuple(ckpt_tensor.shape) == tuple(bias_param.shape)}")
print("=" * 60)


# ============================================================
# Cell 7 — Experiment 1: Reconstruct the Initial Bias
# ============================================================

# The most faithful way to reconstruct the initialization is to instantiate
# a fresh SpatialBiasMatrix with the SAME constructor arguments used by the
# architecture (gamma=0.1, beta=1.0, num_tokens=197) and read its bias_matrix.
# This reproduces the EXACT implementation logic (including CLS handling).

from models.cross_attention import SpatialBiasMatrix

# Inspect the actual constructor args used by the model
bca = model.cross_attention
print("BidirectionalCrossAttention spatial bias module:")
print(f"  num_tokens : {bca.spatial_bias.num_tokens}")
print(f"  grid_size  : {bca.spatial_bias.grid_size}")

# Reconstruct initialization using the exact same class + defaults
fresh_bias_module = SpatialBiasMatrix(
    num_tokens=NUM_TOKENS,
    gamma=0.1,
    beta=1.0,
)
B_init = fresh_bias_module.bias_matrix.data.cpu().numpy()   # (197, 197)

print("=" * 60)
print("Experiment 1 — Reconstructed Initial Bias")
print("=" * 60)
print(f"  Grid size              : {GRID}")
print(f"  Number of spatial patches : {N_PATCH}")
print(f"  CLS token included     : Yes (index 0, row/col = 0)")
print(f"  Distance metric        : Euclidean on {GRID}x{GRID} grid")
print(f"  Initialization gamma   : 0.1 (from constructor)")
print(f"  Initialization beta    : 1.0 (diagonal bonus)")
print(f"  Initial matrix shape   : {B_init.shape}")
print(f"  CLS row all zero       : {np.allclose(B_init[0, :], 0.0)}")
print(f"  CLS col all zero       : {np.allclose(B_init[:, 0], 0.0)}")
print(f"  Diagonal (patch)       : {np.diag(B_init[1:, 1:]).mean():.6f}")
print("=" * 60)


# ============================================================
# Cell 8 — Experiment 2: Initial vs Trained Statistics
# ============================================================

B_trained = bias_param.data.cpu().numpy()   # (197, 197)

# Delta
delta = B_trained - B_init

def stats(name, arr):
    print(f"  {name:<12} min={arr.min():.6f}  max={arr.max():.6f}  "
          f"mean={arr.mean():.6f}  std={arr.std():.6f}")

print("=" * 60)
print("Experiment 2 — Initial vs Trained Statistics")
print("=" * 60)
print("  Initial bias:")
stats("B_init", B_init)
print("  Trained bias:")
stats("B_trained", B_trained)
print("  Delta (B_trained - B_init):")
stats("delta", delta)
print()
print(f"  Mean absolute change      : {np.abs(delta).mean():.6f}")
print(f"  Max absolute change       : {np.abs(delta).max():.6f}")
print(f"  RMSE / RMS change         : {np.sqrt((delta**2).mean()):.6f}")
print(f"  Relative Frobenius change : {np.linalg.norm(delta) / np.linalg.norm(B_init):.6f}")

# Pearson correlation + R² between flattened B_init and B_trained
flat_init = B_init.flatten()
flat_trained = B_trained.flatten()
pearson_r, _ = pearsonr(flat_init, flat_trained)
r2 = pearson_r ** 2
print(f"  Pearson correlation       : {pearson_r:.6f}")
print(f"  R²                        : {r2:.6f}")
print("=" * 60)


# ============================================================
# Cell 9 — Experiment 3: Fit Gamma to Initial and Trained Bias
# ============================================================

# Build the 196x196 patch distance matrix (Euclidean on 14x14 grid)
rows = np.repeat(np.arange(GRID), GRID)
cols = np.tile(np.arange(GRID), GRID)
patch_coords = np.stack([rows, cols], axis=1)   # (196, 2)

dist_matrix = np.zeros((N_PATCH, N_PATCH), dtype=np.float64)
for i in range(N_PATCH):
    for j in range(N_PATCH):
        dist_matrix[i, j] = np.sqrt(((patch_coords[i] - patch_coords[j]) ** 2).sum())

# Patch-only bias (exclude CLS row/col 0)
B_init_patch = B_init[1:, 1:]       # (196, 196)
B_trained_patch = B_trained[1:, 1:] # (196, 196)

# Fit B(i,j) = beta - gamma * d(i,j) over valid spatial pairs.
# Exclude the diagonal (where beta applies, not the distance penalty).
mask = dist_matrix > 0
d_vals = dist_matrix[mask]
b_init_vals = B_init_patch[mask]
b_trained_vals = B_trained_patch[mask]

def fit_gamma_beta(d, b):
    """Fit b = beta - gamma*d via linear regression. Returns (gamma, beta, R2)."""
    # slope of b vs d = -gamma
    A = np.vstack([d, np.ones_like(d)]).T
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    slope, intercept = coef[0], coef[1]
    gamma = -slope
    beta = intercept
    pred = beta - gamma * d
    ss_res = np.sum((b - pred) ** 2)
    ss_tot = np.sum((b - b.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return gamma, beta, r2

gamma_init, beta_init, r2_init = fit_gamma_beta(d_vals, b_init_vals)
gamma_trained, beta_trained, r2_trained = fit_gamma_beta(d_vals, b_trained_vals)

gamma_change = gamma_trained - gamma_init
rel_gamma_change = (gamma_trained - gamma_init) / gamma_init

print("=" * 60)
print("Experiment 3 — Fitted Gamma/Beta")
print("=" * 60)
print(f"  Initial : gamma={gamma_init:.6f}  beta={beta_init:.6f}  R²={r2_init:.6f}")
print(f"  Trained : gamma={gamma_trained:.6f}  beta={beta_trained:.6f}  R²={r2_trained:.6f}")
print(f"  gamma_change        : {gamma_change:.6f}")
print(f"  relative_gamma_change: {rel_gamma_change:.6f} ({rel_gamma_change*100:.2f}%)")
print("=" * 60)


# ============================================================
# Cell 10 — Experiment 4: Visual Comparison (3 panels)
# ============================================================

# Use the SAME color scale for panels A and B; diverging centered at 0 for C.
vmin_ab = min(B_init.min(), B_trained.min())
vmax_ab = max(B_init.max(), B_trained.max())
vmax_delta = np.abs(delta).max()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

im0 = axes[0].imshow(B_init, cmap="RdBu_r", vmin=vmin_ab, vmax=vmax_ab)
axes[0].set_title("A) Initialized Spatial Bias")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(B_trained, cmap="RdBu_r", vmin=vmin_ab, vmax=vmax_ab)
axes[1].set_title("B) Trained Spatial Bias")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(delta, cmap="RdBu_r", vmin=-vmax_delta, vmax=vmax_delta)
axes[2].set_title("C) Difference (Trained - Initial)")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("Spatial Bias: Initialization vs Trained", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part1_bias_initial_vs_trained.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part1_bias_initial_vs_trained.png")


# ============================================================
# Cell 11 — Experiment 5: Bias Difference Distribution
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(delta.flatten(), bins=100, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(0.0, color="red", linestyle="--", label="0 (no change)")
ax.set_xlabel("B_trained - B_init")
ax.set_ylabel("Count")
ax.set_title("Histogram of Bias Difference")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part2_bias_difference_histogram.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part2_bias_difference_histogram.png")

abs_delta = np.abs(delta)
print("=" * 60)
print("Experiment 5 — Difference Distribution")
print("=" * 60)
print("  Percentage of entries with |delta| < threshold (diagnostic only):")
for thr in [1e-3, 1e-2, 5e-2, 0.1]:
    pct = float((abs_delta < thr).mean() * 100)
    print(f"    |delta| < {thr:<6}: {pct:6.2f}%")
print("=" * 60)


# ============================================================
# Cell 12 — Experiment 6: Initial vs Trained Scatter
# ============================================================

rmse = np.sqrt((delta**2).mean())

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(flat_init, flat_trained, s=2, alpha=0.3, color="steelblue")
lims = [min(flat_init.min(), flat_trained.min()), max(flat_init.max(), flat_trained.max())]
ax.plot(lims, lims, "r--", label="y = x")
ax.set_xlabel("B_init")
ax.set_ylabel("B_trained")
ax.set_title(f"Initial vs Trained Bias\nPearson r={pearson_r:.4f}, R²={r2:.4f}, RMSE={rmse:.6f}")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part3_bias_initial_vs_trained_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part3_bias_initial_vs_trained_scatter.png")

print("=" * 60)
print("Experiment 6 — Scatter Interpretation")
print("=" * 60)
if r2 > 0.999 and rmse < 1e-3:
    print("  -> Points lie almost exactly on y=x: trained bias is very close to initialization.")
elif r2 > 0.99:
    print("  -> Points lie close to y=x with small deviation: bias changed slightly.")
else:
    print("  -> Substantial systematic deviation: bias learned significantly during training.")
print("=" * 60)


# ============================================================
# Cell 13 — Experiment 7: Bias vs Distance (2 panels)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# LEFT: B_init vs distance
axes[0].scatter(d_vals[::50], b_init_vals[::50], s=3, alpha=0.3, color="steelblue")
d_line = np.linspace(d_vals.min(), d_vals.max(), 100)
axes[0].plot(d_line, beta_init - gamma_init * d_line, "r-", lw=2,
             label=f"gamma={gamma_init:.4f}, beta={beta_init:.4f}, R²={r2_init:.4f}")
axes[0].set_xlabel("Patch distance d(i,j)")
axes[0].set_ylabel("B(i,j)")
axes[0].set_title("Initial Bias vs Distance")
axes[0].legend()

# RIGHT: B_trained vs distance
axes[1].scatter(d_vals[::50], b_trained_vals[::50], s=3, alpha=0.3, color="darkorange")
axes[1].plot(d_line, beta_trained - gamma_trained * d_line, "r-", lw=2,
             label=f"gamma={gamma_trained:.4f}, beta={beta_trained:.4f}, R²={r2_trained:.4f}")
axes[1].set_xlabel("Patch distance d(i,j)")
axes[1].set_ylabel("B(i,j)")
axes[1].set_title("Trained Bias vs Distance")
axes[1].legend()

# Identical axis ranges
xmin = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part4_bias_vs_distance.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part4_bias_vs_distance.png")


# ============================================================
# Cell 14 — Experiment 8: Row/Column Structure Check
# ============================================================

# Verify the matrix really corresponds to the 2D patch grid.
# Show a representative spatial subset (e.g., first 14x14 patch block),
# explicitly separating CLS from the spatial grid.

# Patch-only matrices reshaped to (14, 14, 14, 14): [row_q, col_q, row_k, col_k]
B_init_4d = B_init_patch.reshape(GRID, GRID, GRID, GRID)
B_trained_4d = B_trained_patch.reshape(GRID, GRID, GRID, GRID)
delta_4d = B_trained_4d - B_init_4d

# For a fixed query patch (center), show the attention-bias over key patches
q_row, q_col = GRID // 2, GRID // 2   # center query patch

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

vmin_ab = min(B_init_4d.min(), B_trained_4d.min())
vmax_ab = max(B_init_4d.max(), B_trained_4d.max())
vmax_d = np.abs(delta_4d).max()

im0 = axes[0].imshow(B_init_4d[q_row, q_col], cmap="RdBu_r", vmin=vmin_ab, vmax=vmax_ab)
axes[0].set_title(f"Initial Bias from center patch ({q_row},{q_col})")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(B_trained_4d[q_row, q_col], cmap="RdBu_r", vmin=vmin_ab, vmax=vmax_ab)
axes[1].set_title("Trained Bias from center patch")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(delta_4d[q_row, q_col], cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
axes[2].set_title("Difference from center patch")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("Spatial Bias Structure (14x14 patch grid, center query)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part5_spatial_bias_structure.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part5_spatial_bias_structure.png")

print("=" * 60)
print("Experiment 8 — Structure Check")
print("=" * 60)
print(f"  CLS token (index 0) handled separately: row/col zero in both matrices")
print(f"  Patch-only matrices reshaped to 4D grid: {B_init_4d.shape}")
print(f"  Center query patch: ({q_row}, {q_col})")
print("=" * 60)


# ============================================================
# Cell 15 — Experiment 9: Parameter / Checkpoint Verification
# ============================================================

# 1) Is the bias in the checkpoint?
print("=" * 60)
print("Experiment 9 — Parameter / Checkpoint Verification")
print("=" * 60)
print(f"  Exact checkpoint key : {ckpt_key}")
print(f"  Key present          : {ckpt_key in ckpt_state}")
if ckpt_key in ckpt_state:
    ckpt_tensor = ckpt_state[ckpt_key]
    print(f"  Checkpoint shape     : {tuple(ckpt_tensor.shape)}")
    print(f"  Model param shape    : {tuple(bias_param.shape)}")
    print(f"  Shapes match         : {tuple(ckpt_tensor.shape) == tuple(bias_param.shape)}")
print(f"  requires_grad (arch) : {bias_param.requires_grad}")

# 2) Compare checkpoint bias vs a FRESH initialized model's bias.
#    This independently verifies we are not accidentally reconstructing
#    the trained tensor as "initialization".
fresh_model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.1,
    classifier_dropout=0.1,
)
fresh_bias = fresh_model.cross_attention.spatial_bias.bias_matrix.data.cpu().numpy()

# Compare fresh-init vs reconstructed B_init (should be identical)
init_match = np.allclose(fresh_bias, B_init, atol=1e-8)
print(f"  Fresh-init == reconstructed B_init : {init_match}")

# Compare fresh-init vs checkpoint trained bias (should differ if training changed it)
fresh_vs_trained_diff = np.abs(fresh_bias - B_trained).max()
print(f"  Max |fresh_init - trained|         : {fresh_vs_trained_diff:.6f}")
print(f"  Fresh-init differs from trained    : {fresh_vs_trained_diff > 1e-6}")
print("=" * 60)


# ============================================================
# Cell 16 — Experiment 10: Gradient / Trainability Sanity Check
# ============================================================

print("=" * 60)
print("Experiment 10 — Gradient / Trainability Sanity Check")
print("=" * 60)
print(f"  requires_grad : {bias_param.requires_grad}")
print(f"  is nn.Parameter: {isinstance(bias_param, nn.Parameter)}")

# Verify optimizer inclusion from the training code.
# models/dsca_vit.py get_parameter_groups() puts cross_attention (which
# contains spatial_bias.bias_matrix) into the "new" group.
# notebooks/train.py Stage 2 optimizer uses param_groups["new"] at lr 1e-4.
print()
print("  Optimizer inclusion check (from training code):")
print("    models/dsca_vit.py get_parameter_groups():")
print("      'new' = proj_h + proj_d + cross_attention + fusion + refinement + classifier")
print("      -> cross_attention.spatial_bias.bias_matrix IS in the 'new' group")
print("    notebooks/train.py Stage 2 optimizer:")
print("      optimizer = Adam([{encoder, lr=1e-5}, {new, lr=1e-4}])")
print("      -> the bias parameter IS included in the Stage 2 optimizer")
print()
print("  Conclusion: The spatial bias parameter WAS included in the Stage 2 optimizer.")
print("  Therefore an unchanged bias would reflect learning behavior, not an exclusion bug.")
print("=" * 60)


# ============================================================
# Cell 17 — Save Results (CSV)
# ============================================================

import csv

# --- spatial_bias_comparison.csv: full matrices ---
csv1 = os.path.join(OUT_DIR, "spatial_bias_comparison.csv")
with open(csv1, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "j", "B_init", "B_trained", "delta"])
    for i in range(NUM_TOKENS):
        for j in range(NUM_TOKENS):
            writer.writerow([i, j, f"{B_init[i,j]:.8f}", f"{B_trained[i,j]:.8f}", f"{delta[i,j]:.8f}"])
print(f"✅ Saved: {csv1}")

# --- spatial_bias_summary.csv: summary metrics ---
csv2 = os.path.join(OUT_DIR, "spatial_bias_summary.csv")
summary_rows = [
    ["metric", "initial", "trained", "change"],
    ["mean", f"{B_init.mean():.8f}", f"{B_trained.mean():.8f}", f"{delta.mean():.8f}"],
    ["std", f"{B_init.std():.8f}", f"{B_trained.std():.8f}", f"{delta.std():.8f}"],
    ["min", f"{B_init.min():.8f}", f"{B_trained.min():.8f}", f"{delta.min():.8f}"],
    ["max", f"{B_init.max():.8f}", f"{B_trained.max():.8f}", f"{delta.max():.8f}"],
    ["gamma", f"{gamma_init:.8f}", f"{gamma_trained:.8f}", f"{gamma_change:.8f}"],
    ["beta", f"{beta_init:.8f}", f"{beta_trained:.8f}", f"{beta_trained - beta_init:.8f}"],
    ["R2", f"{r2_init:.8f}", f"{r2_trained:.8f}", f"{r2_trained - r2_init:.8f}"],
    ["relative_frobenius_change", "", "", f"{np.linalg.norm(delta)/np.linalg.norm(B_init):.8f}"],
    ["rmse", "", "", f"{np.sqrt((delta**2).mean()):.8f}"],
    ["pearson_r", "", "", f"{pearson_r:.8f}"],
    ["mean_abs_change", "", "", f"{np.abs(delta).mean():.8f}"],
    ["max_abs_change", "", "", f"{np.abs(delta).max():.8f}"],
]
with open(csv2, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(summary_rows)
print(f"✅ Saved: {csv2}")


# ============================================================
# Cell 18 — Final Report
# ============================================================

print("=" * 60)
print("SPATIAL BIAS INITIALIZATION VS TRAINED — FINAL REPORT")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Bias parameter: {bias_path}")
print(f"Shape: {tuple(bias_param.shape)}")
print()
print(f"Initial gamma: {gamma_init:.4f}")
print(f"Trained gamma: {gamma_trained:.4f}")
print(f"Gamma change: {gamma_change:.4f}")
print(f"Relative gamma change: {rel_gamma_change*100:.2f}%")
print(f"Initial → trained correlation: {pearson_r:.4f}")
print(f"Relative Frobenius change: {np.linalg.norm(delta)/np.linalg.norm(B_init)*100:.4f}%")
print(f"RMSE: {np.sqrt((delta**2).mean()):.6f}")
print(f"R²: {r2:.4f}")
print(f"% entries with |ΔB| < 0.001: {float((abs_delta < 1e-3).mean()*100):.2f}%")
print(f"% entries with |ΔB| < 0.01: {float((abs_delta < 1e-2).mean()*100):.2f}%")
print(f"% entries with |ΔB| < 0.05: {float((abs_delta < 5e-2).mean()*100):.2f}%")
print(f"Optimizer inclusion: YES (verified from training code)")
print("=" * 60)

# --- Automatic interpretation (evidence-based) ---
print()
print("Interpretation (based on measured evidence):")
print()

# Evidence
rel_frob = np.linalg.norm(delta) / np.linalg.norm(B_init)
rmse_val = np.sqrt((delta**2).mean())
pct_lt_1e3 = float((abs_delta < 1e-3).mean() * 100)
pct_lt_1e2 = float((abs_delta < 1e-2).mean() * 100)

if pct_lt_1e3 > 99.9 and rmse_val < 1e-4:
    category = "A) BIAS ESSENTIALLY UNCHANGED"
    conclusion = ("The trained bias tensor is demonstrably very close to initialization "
                  "(>99.9% of entries differ by <1e-3, RMSE < 1e-4).")
elif pct_lt_1e2 > 95 and rmse_val < 1e-2:
    category = "A) BIAS ESSENTIALLY UNCHANGED"
    conclusion = ("The trained bias tensor is very close to initialization "
                  "(>95% of entries differ by <1e-2).")
elif pct_lt_1e2 > 50 and rel_frob < 0.1:
    category = "B) BIAS MODERATELY UPDATED"
    conclusion = ("The trained bias has measurable but not dramatic deviation from initialization "
                  "(relative Frobenius change < 10%).")
elif rel_frob >= 0.1:
    category = "C) BIAS SUBSTANTIALLY LEARNED"
    conclusion = ("The trained bias clearly differs from initialization "
                  "(relative Frobenius change >= 10%).")
else:
    category = "D) CANNOT DETERMINE"
    conclusion = "Implementation/checkpoint information is insufficient for a definitive conclusion."

print(f"Category: {category}")
print(f"Evidence: {conclusion}")
print()
print("Numerical evidence summary:")
print(f"  Relative Frobenius change : {rel_frob*100:.4f}%")
print(f"  RMSE                      : {rmse_val:.6f}")
print(f"  % |ΔB| < 1e-3             : {pct_lt_1e3:.2f}%")
print(f"  % |ΔB| < 1e-2             : {pct_lt_1e2:.2f}%")
print(f"  Pearson r (init vs trained): {pearson_r:.6f}")
print(f"  R²                        : {r2:.6f}")
print(f"  gamma_init                : {gamma_init:.6f}")
print(f"  gamma_trained             : {gamma_trained:.6f}")
print(f"  gamma_change              : {gamma_change:.6f}")
print("=" * 60)