# DSCA-ViT — Cross-Attention Analysis Notebook (02)
# ============================================================
# PURPOSE:
#   Inspect the learned cross-attention behaviour of a trained
#   DSCA-ViT model WITHOUT modifying or retraining it.
#
#   Questions answered:
#     - Is cross-attention actually active?
#     - Is attention focused or diffuse?
#     - Does the Beer-Lambert spatial bias influence attention?
#     - Does attention differ between HER2 classes?
#     - Does attention collapse during inference?
#     - Does the attention look biologically meaningful?
#
# HOW TO RUN:
#   Run all cells in order. No manual editing required.
#
# CHECKPOINT:
#   /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth
#
# OUTPUT:
#   .../DSCA_ViT/Results/Cross-Attention_analysis/   (figures + CSV)
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
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, spearmanr, mannwhitneyu

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
BATCH_SIZE      = 32
GRID            = 14                 # patch grid for cross-attention distance
N_PATCH         = GRID * GRID        # 196 patch tokens

CLASS_NAMES = ["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"]

CHECKPOINT_PATH = "/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth"

# Output folder (user-specified name): Cross-Attention_analysis
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
EXPERIMENT_DIR  = os.path.join(CHECKPOINT_ROOT, BACKBONE_NAME)
RESULTS_DIR     = os.path.join(EXPERIMENT_DIR, "Results")
OUT_DIR         = os.path.join(RESULTS_DIR, "Cross-Attention_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Cross-Attention Analysis Configuration")
print("=" * 60)
print(f"Model           : {BACKBONE_NAME}")
print(f"Checkpoint      : {CHECKPOINT_PATH}")
print(f"Output Dir      : {OUT_DIR}")
print("=" * 60)


# ============================================================
# Cell 5 — Dataset (downloads only if missing, then extracts)
# ============================================================

import zipfile

DATA_ROOT = Path("/content/HER2_Dataset")
DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), URL], check=True)
else:
    print("Dataset archive already exists.")

WSI_DIR = DATA_ROOT / "WSI-based-dataset"

if not WSI_DIR.exists():
    print("Extracting main archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Main archive extracted.")
else:
    print("Main archive already extracted.")

nested_archives = [
    WSI_DIR / "train_data_wsi.zip",
    WSI_DIR / "test_data_wsi.zip",
]

for archive in nested_archives:
    extract_folder = archive.parent / archive.stem.replace("_data_wsi", "")
    if extract_folder.exists():
        print(f"{extract_folder.name} already extracted.")
        continue
    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(extract_folder)

for archive in [ZIP_PATH] + nested_archives:
    if archive.exists():
        archive.unlink()

print("ZIP files removed.")

TEST_DIR = WSI_DIR / "test"
assert TEST_DIR.exists(), "Test directory not found."
print("\nDataset successfully prepared!")


# ============================================================
# Cell 6 — Build Model + Load Stage 2 Checkpoint
# ============================================================

from models import DSCAViT

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
# Cell 7 — Part 1: Locate Every Cross-Attention Block
# ============================================================

from models.cross_attention import CrossAttentionLayer

# Recursively find all CrossAttentionLayer modules
layers = []
def find_cross_attn(module, path=""):
    for name, child in module.named_children():
        child_path = f"{path}.{name}" if path else name
        if isinstance(child, CrossAttentionLayer):
            layers.append((child_path, child))
        find_cross_attn(child, child_path)

find_cross_attn(model)

print("=" * 60)
print("Part 1 — Cross-Attention Blocks Found")
print("=" * 60)
for path, layer in layers:
    print(f"  Layer            : {path}")
    print(f"    Heads          : {layer.num_heads}")
    print(f"    Embedding dim  : {layer.embed_dim}")
    print(f"    Head dim       : {layer.head_dim}")
    print()
print(f"  Total cross-attention layers: {len(layers)}")
print("=" * 60)

# The spatial bias matrix is SHARED between both directions (in BidirectionalCrossAttention)
bias_matrix = model.cross_attention.spatial_bias.bias_matrix.data.cpu().numpy()  # (197, 197)
print(f"  Shared spatial bias matrix shape: {bias_matrix.shape}")
print("=" * 60)


# ============================================================
# Cell 8 — Part 2: Capture Attention Weights/Logits via Forward Hooks
# ============================================================

# ------------------------------------------------------------------
# Read-only captures via forward wrapping (monkey-patch).
# We temporarily wrap each CrossAttentionLayer.forward to:
#   1. recompute the raw attention logits (QK^T/sqrt(d) + bias) using the
#      module's own learned weights (detached) — does NOT alter predictions;
#   2. read module.attn_weights (already stored by the module) + output;
#   3. delegate to the ORIGINAL forward, so outputs/predictions are unchanged.
#
# This approach works on ALL PyTorch versions (no hook kwargs support needed)
# and never modifies weights or retrains the model.
# ------------------------------------------------------------------

hook_state = {}   # layer_idx -> {"attn": [], "logits": [], "out": []}

def make_forward_wrapper(idx, layer):
    original_forward = layer.forward

    def wrapped_forward(*args, **kwargs):
        # Capture inputs (the module is called with keyword args)
        source = kwargs.get("source", args[0] if len(args) > 0 else None)
        context = kwargs.get("context", args[1] if len(args) > 1 else None)
        spatial_bias = kwargs.get("spatial_bias", args[2] if len(args) > 2 else None)

        # Recompute logits (detached, read-only)
        with torch.no_grad():
            B, Ns, C = source.shape                # Ns = source (query) tokens
            Nc = context.shape[1]                  # Nc = context (key/value) tokens
            ns = layer.norm_source(source)
            nc = layer.norm_context(context)
            q = layer.q_proj(ns).reshape(B, Ns, layer.num_heads, layer.head_dim).permute(0, 2, 1, 3)
            k = layer.k_proj(nc).reshape(B, Nc, layer.num_heads, layer.head_dim).permute(0, 2, 1, 3)
            logits = (q @ k.transpose(-2, -1)) / (layer.head_dim ** 0.5)
            if spatial_bias is not None:
                logits = logits + spatial_bias.unsqueeze(0).unsqueeze(0)
        hook_state[idx]["logits"].append(logits.detach().cpu())

        # Delegate to the original forward (predictions unchanged)
        output = original_forward(*args, **kwargs)

        # Capture attention weights + output embedding
        hook_state[idx]["attn"].append(layer.attn_weights.detach().cpu())
        hook_state[idx]["out"].append(output.detach().cpu())
        return output

    return wrapped_forward

# We store full attention for the selected sample images only (memory).
# For the full test set we compute online statistics instead.
for i, (path, layer) in enumerate(layers):
    hook_state[i] = {"attn": [], "logits": [], "out": []}
    layer.forward = make_forward_wrapper(i, layer)

print(f"✅ Wrapped forward of {len(layers)} cross-attention layers (read-only).")


# ============================================================
# Cell 9 — Inference: Collect Per-Image Statistics (online)
# ============================================================

from datasets import HER2Dataset, get_test_transform
from torch.utils.data import DataLoader

test_transform = get_test_transform(image_size=IMAGE_SIZE)
test_dataset = HER2Dataset(root_dir=str(TEST_DIR), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)

print(f"Test images : {len(test_dataset)}")

# Patch coordinates for distance computation (skip CLS)
rows = np.repeat(np.arange(GRID), GRID)          # (196,)
cols = np.tile(np.arange(GRID), GRID)
patch_coords = np.stack([rows, cols], axis=1)    # (196, 2)

# Precompute the 196x196 patch distance matrix
dist_matrix = np.zeros((N_PATCH, N_PATCH), dtype=np.float32)
for i in range(N_PATCH):
    for j in range(N_PATCH):
        dist_matrix[i, j] = np.sqrt(((patch_coords[i] - patch_coords[j]) ** 2).sum())

# Per-image summary records
records = []          # label, pred, conf, per-layer: mean_entropy, mean_max_attn,
                      # sparsity fractions, mean_distance
N_LAYERS = len(layers)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        B = images.size(0)

        # Collect hook captures for this batch (attn: (B, heads, 197, 197))
        batch_attns = {i: torch.cat(hook_state[i]["attn"], dim=0) for i in range(N_LAYERS)}
        batch_logits = {i: torch.cat(hook_state[i]["logits"], dim=0) for i in range(N_LAYERS)}
        # Clear hook state for next batch
        for i in range(N_LAYERS):
            hook_state[i] = {"attn": [], "logits": [], "out": []}

        for b in range(B):
            rec = {
                "filename": test_dataset.image_paths[len(records)].name,
                "label": labels[b].item(),
                "pred": preds[b].item(),
                "confidence": conf[b].item(),
                "layer_stats": [],
            }
            for i in range(N_LAYERS):
                attn = batch_attns[i][b]                 # (heads, 197, 197)
                # Patch-only attention (exclude CLS row/col)
                attn_patch = attn[:, 1:, 1:]             # (heads, 196, 196)
                ent = -(attn_patch * torch.log(attn_patch + 1e-12)).sum(dim=-1)  # (heads, 196)
                max_attn = attn_patch.max(dim=-1).values  # (heads, 196)
                # Sparsity fractions per head
                frac_05 = (attn_patch > 0.5).float().mean(dim=(1, 2))
                frac_025 = (attn_patch > 0.25).float().mean(dim=(1, 2))
                frac_01 = (attn_patch > 0.1).float().mean(dim=(1, 2))
                # Weighted distance: average key distance weighted by attention (per head, per query -> mean)
                attn_np = attn_patch.cpu().numpy()
                # expected distance per head = sum_ij a_ij * dist_ij / 196_queries
                exp_dist = (attn_np * dist_matrix[None]).sum(axis=(1, 2)) / N_PATCH
                rec["layer_stats"].append({
                    "mean_entropy": float(ent.mean()),
                    "mean_max_attn": float(max_attn.mean()),
                    "frac_gt_05": float(frac_05.mean()),
                    "frac_gt_025": float(frac_025.mean()),
                    "frac_gt_01": float(frac_01.mean()),
                    "mean_distance": float(exp_dist.mean()),
                })
            records.append(rec)

print(f"✅ Collected per-image cross-attention statistics for {len(records)} images.")


# ============================================================
# Cell 10 — Part 3: Attention Entropy
# ============================================================

all_entropy = np.array([r["layer_stats"][0]["mean_entropy"] for r in records])
all_entropy_d = np.array([r["layer_stats"][1]["mean_entropy"] for r in records])

# Pool both layers
entropy_pooled = np.concatenate([all_entropy, all_entropy_d])

print("=" * 60)
print("Part 3 — Attention Entropy")
print("=" * 60)
print(f"  Mean entropy   : {entropy_pooled.mean():.4f}")
print(f"  Std entropy    : {entropy_pooled.std():.4f}")
print(f"  Median entropy : {np.median(entropy_pooled):.4f}")
print(f"  Max possible   : {np.log(N_PATCH):.2f}  (uniform over 196 patches)")
print()

if entropy_pooled.mean() < 1.5:
    print("  -> Very low entropy: attention extremely concentrated")
elif entropy_pooled.mean() < 3.0:
    print("  -> Low-to-moderate entropy: healthy selective attention")
elif entropy_pooled.mean() < 4.0:
    print("  -> Moderate entropy: diffuse but selective-ish")
else:
    print("  -> High entropy: almost uniform attention (likely collapsed)")
print("=" * 60)

# Histogram of per-image entropy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(entropy_pooled, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Mean attention entropy (per image)")
axes[0].set_ylabel("Count")
axes[0].set_title("Histogram of Attention Entropy (both directions)")

# Boxplot per layer
axes[1].boxplot([all_entropy, all_entropy_d], labels=["H←DAB", "DAB←H"], patch_artist=True)
axes[1].set_ylabel("Mean attention entropy")
axes[1].set_title("Entropy by Cross-Attention Direction")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part3_attention_entropy.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part3_attention_entropy.png")


# ============================================================
# Cell 11 — Part 4: Maximum Attention Score
# ============================================================

all_max = np.array([r["layer_stats"][0]["mean_max_attn"] for r in records])
all_max_d = np.array([r["layer_stats"][1]["mean_max_attn"] for r in records])
max_pooled = np.concatenate([all_max, all_max_d])

print("=" * 60)
print("Part 4 — Maximum Attention Score")
print("=" * 60)
print(f"  Mean of per-image max attention : {max_pooled.mean():.4f}")
print(f"  Std  of per-image max attention : {max_pooled.std():.4f}")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(max_pooled, bins=40, color="darkorange", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Mean max attention")
axes[0].set_ylabel("Count")
axes[0].set_title("Histogram of Max Attention (per image)")

axes[1].boxplot([all_max, all_max_d], labels=["H←DAB", "DAB←H"], patch_artist=True)
axes[1].set_ylabel("Mean max attention")
axes[1].set_title("Max Attention by Direction")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part4_max_attention.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part4_max_attention.png")


# ============================================================
# Cell 12 — Part 5: Attention Sparsity
# ============================================================

fracs = {
    ">0.5":  np.array([r["layer_stats"][0]["frac_gt_05"] for r in records]),
    ">0.25": np.array([r["layer_stats"][0]["frac_gt_025"] for r in records]),
    ">0.1":  np.array([r["layer_stats"][0]["frac_gt_01"] for r in records]),
}

print("=" * 60)
print("Part 5 — Attention Sparsity (H←DAB direction)")
print("=" * 60)
for label, vals in fracs.items():
    print(f"  Fraction of attention weights {label}: mean={vals.mean():.4f}, std={vals.std():.4f}")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (label, vals) in zip(axes, fracs.items()):
    ax.hist(vals, bins=40, color="seagreen", edgecolor="white", alpha=0.8)
    ax.set_xlabel(f"Fraction of values {label}")
    ax.set_ylabel("Count")
    ax.set_title(f"Sparsity: fraction {label}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part5_sparsity.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part5_sparsity.png")


# ============================================================
# Cell 13 — Part 6: Attention Distance (Query-Key Patch Distance)
# ============================================================

all_dist = np.array([r["layer_stats"][0]["mean_distance"] for r in records])
all_dist_d = np.array([r["layer_stats"][1]["mean_distance"] for r in records])
dist_pooled = np.concatenate([all_dist, all_dist_d])

print("=" * 60)
print("Part 6 — Attention Spatial Distance")
print("=" * 60)
print(f"  Mean attention distance : {dist_pooled.mean():.3f} patches")
print(f"  Std  attention distance : {dist_pooled.std():.3f} patches")
print(f"  (max on 14x14 grid = {np.sqrt(2*(GRID-1)**2):.2f} patches)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(dist_pooled, bins=40, color="mediumpurple", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Mean attention distance (patches)")
axes[0].set_ylabel("Count")
axes[0].set_title("Histogram of Attention Distance")

axes[1].boxplot([all_dist, all_dist_d], labels=["H←DAB", "DAB←H"], patch_artist=True)
axes[1].set_ylabel("Mean attention distance")
axes[1].set_title("Attention Distance by Direction")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part6_attention_distance.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part6_attention_distance.png")


# ============================================================
# Cell 14 — Part 6b: Attention Direction Comparison (H→DAB vs DAB→H)
# ============================================================
# DSCA-ViT uses BIDIRECTIONAL cross-attention between two stain streams.
# This experiment checks whether the two directions actually behave differently.
# We compare, per image (paired):
#   - Entropy(H→DAB)  vs  Entropy(DAB→H)
#   - MaxAttention(H→DAB) vs MaxAttention(DAB→H)
#   - Distance(H→DAB) vs Distance(DAB→H)
# using paired statistical tests (Wilcoxon signed-rank) + distribution plots.

from scipy.stats import wilcoxon

# Per-image paired arrays (already computed in Cell 13)
#   all_entropy   = H→DAB direction (layer 0)
#   all_entropy_d = DAB→H direction (layer 1)
#   all_max       = H→DAB
#   all_max_d     = DAB→H
#   all_dist      = H→DAB
#   all_dist_d    = DAB→H

print("=" * 60)
print("Part 6b — Attention Direction Comparison (H→DAB vs DAB→H)")
print("=" * 60)

# --- Paired statistics ---
print("  Per-image means (H→DAB vs DAB→H):")
print(f"    Entropy      : {all_entropy.mean():.4f}  vs  {all_entropy_d.mean():.4f}")
print(f"    Max attention: {all_max.mean():.4f}  vs  {all_max_d.mean():.4f}")
print(f"    Distance     : {all_dist.mean():.3f}  vs  {all_dist_d.mean():.3f}")
print()

# --- Paired Wilcoxon signed-rank tests ---
# (paired per image, so Wilcoxon is appropriate)
try:
    stat_ent, p_ent_dir = wilcoxon(all_entropy, all_entropy_d)
    stat_max, p_max_dir = wilcoxon(all_max, all_max_d)
    stat_dist, p_dist_dir = wilcoxon(all_dist, all_dist_d)
    print("  Paired Wilcoxon signed-rank tests (H→DAB vs DAB→H):")
    print(f"    Entropy      : p = {p_ent_dir:.4e}")
    print(f"    Max attention: p = {p_max_dir:.4e}")
    print(f"    Distance     : p = {p_dist_dir:.4e}")
    print()
    print("  Interpretation:")
    if p_ent_dir < 0.05:
        print("    ✓ Entropy differs significantly between directions -> directions behave differently")
    else:
        print("    ⚠ Entropy not significantly different -> directions may be symmetric")
    if p_max_dir < 0.05:
        print("    ✓ Max attention differs significantly between directions")
    else:
        print("    ⚠ Max attention not significantly different")
    if p_dist_dir < 0.05:
        print("    ✓ Attention distance differs significantly -> directions use different spatial ranges")
    else:
        print("    ⚠ Attention distance not significantly different")
except Exception as e:
    print(f"  Wilcoxon test failed (possibly identical arrays): {e}")
    p_ent_dir = p_max_dir = p_dist_dir = 1.0
print("=" * 60)

# --- Paired boxplots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].boxplot([all_entropy, all_entropy_d], labels=["H→DAB", "DAB→H"], patch_artist=True)
axes[0].set_title("Attention Entropy by Direction")
axes[0].set_ylabel("Entropy")

axes[1].boxplot([all_max, all_max_d], labels=["H→DAB", "DAB→H"], patch_artist=True)
axes[1].set_title("Max Attention by Direction")
axes[1].set_ylabel("Max attention")

axes[2].boxplot([all_dist, all_dist_d], labels=["H→DAB", "DAB→H"], patch_artist=True)
axes[2].set_title("Attention Distance by Direction")
axes[2].set_ylabel("Distance (patches)")

plt.suptitle("Part 6b — Attention Direction Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part6b_direction_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part6b_direction_comparison.png")

# --- Paired scatter (per-image H→DAB vs DAB→H) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(all_entropy, all_entropy_d, alpha=0.3, s=8)
lim0 = [min(all_entropy.min(), all_entropy_d.min()), max(all_entropy.max(), all_entropy_d.max())]
axes[0].plot(lim0, lim0, "r--")
axes[0].set_xlabel("Entropy H→DAB")
axes[0].set_ylabel("Entropy DAB→H")
axes[0].set_title("Entropy: H→DAB vs DAB→H (per image)")

axes[1].scatter(all_max, all_max_d, alpha=0.3, s=8)
lim1 = [min(all_max.min(), all_max_d.min()), max(all_max.max(), all_max_d.max())]
axes[1].plot(lim1, lim1, "r--")
axes[1].set_xlabel("Max attention H→DAB")
axes[1].set_ylabel("Max attention DAB→H")
axes[1].set_title("Max Attention: H→DAB vs DAB→H")

axes[2].scatter(all_dist, all_dist_d, alpha=0.3, s=8)
lim2 = [min(all_dist.min(), all_dist_d.min()), max(all_dist.max(), all_dist_d.max())]
axes[2].plot(lim2, lim2, "r--")
axes[2].set_xlabel("Distance H→DAB")
axes[2].set_ylabel("Distance DAB→H")
axes[2].set_title("Distance: H→DAB vs DAB→H")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part6b_direction_scatter.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part6b_direction_scatter.png")


# ============================================================
# Cell 15 — Part 7: Spatial Bias Contribution (learned bias matrix)
# ============================================================

# The spatial bias matrix is SHARED between the two directions
bias_matrix = model.cross_attention.spatial_bias.bias_matrix.data.cpu().numpy()  # (197, 197)
bias_patch = bias_matrix[1:, 1:]   # (196, 196) patch-only

# Estimate effective gamma by fitting: B[i,j] ≈ -gamma_eff * dist(i,j)
# (excluding diagonal where beta is applied)
mask = dist_matrix > 0
d_vals = dist_matrix[mask]
b_vals = bias_patch[mask]

# slope of B vs -d  -> gamma_eff
gamma_eff = float(np.polyfit(d_vals, -b_vals, 1)[0])   # gamma_eff = -dB/dd
beta_diag = float(np.diag(bias_patch).mean())

print("=" * 60)
print("Part 7 — Spatial Bias Contribution")
print("=" * 60)
print(f"  Initialization gamma : 0.1 (constructor)")
print(f"  Learned beta (diagonal) : {beta_diag:.4f}")
print(f"  Effective gamma (fitted): {gamma_eff:.4f}")
print()
if gamma_eff < 0.05:
    print("  -> Gamma ≈ 0: bias was effectively ignored by training")
elif gamma_eff < 0.2:
    print("  -> Gamma moderate: bias retained as a soft prior")
else:
    print("  -> Gamma high: bias strongly dominates attention locality")
print("=" * 60)

# Visualize bias matrix vs distance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(bias_patch[:50, :50], cmap="RdBu_r")
axes[0].set_title("Learned Spatial Bias Matrix (first 50 patches)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

axes[1].scatter(d_vals[::50], b_vals[::50], s=2, alpha=0.3)
axes[1].set_xlabel("Patch distance d(i,j)")
axes[1].set_ylabel("Learned bias B(i,j)")
axes[1].set_title(f"Bias vs Distance (fitted gamma={gamma_eff:.3f})")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part7_spatial_bias.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part7_spatial_bias.png")


# ============================================================
# Cell 16 — Part 8: Attention Maps (5 per class)
# ============================================================

from models.color_deconv import deconvolve_numpy
from PIL import Image

N_PER_CLASS = 5
# Choose 5 random images per class
random.seed(SEED)
selected_idx = []
for c in range(NUM_CLASSES):
    idxs = [i for i, r in enumerate(records) if r["label"] == c]
    selected_idx.extend(random.sample(idxs, min(N_PER_CLASS, len(idxs))))

def smooth_upsample(map_14, size=224):
    arr = (map_14 * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return np.array(img) / 255.0

# We need full attention maps for these images -> re-run on those specific images.
# Build a map from dataset index -> record (records are in dataset order)
# Re-infer selected images individually to capture full attention tensors.
attn_maps_full = {}   # dataset_idx -> {layer: attn (heads,197,197)}

with torch.no_grad():
    for idx in selected_idx:
        img_t, lab = test_dataset[idx]
        input_t = img_t.unsqueeze(0).to(device)
        hook_state.clear()
        for i, (path, layer) in enumerate(layers):
            hook_state[i] = {"attn": [], "logits": [], "out": []}
        _ = model(input_t)
        attn_maps_full[idx] = {}
        for i in range(N_LAYERS):
            attn_maps_full[idx][i] = torch.cat(hook_state[i]["attn"], dim=0)[0].cpu().numpy()  # (heads,197,197)

# Plot: RGB | H | DAB | H←DAB average attention heatmap | overlay
n = len(selected_idx)
fig, axes = plt.subplots(n, 5, figsize=(19, 5 * n))

for row, idx in enumerate(selected_idx):
    rec = records[idx]
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    h_ch, dab_ch = deconvolve_numpy(img_rgb)

    # Average attention over heads for H←DAB layer, patch->patch, reshape to 14x14
    attn = attn_maps_full[idx][0]                      # (heads, 197, 197)
    attn_avg = attn.mean(axis=0)[1:, 1:]               # (196, 196) query->key
    # For each query patch, the attended key location = argmax; build a 14x14 map of max-attention source
    attn_query_map = attn_avg.max(axis=1).reshape(GRID, GRID)  # max attention received by each query patch
    attn_upsampled = smooth_upsample(attn_query_map)

    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f"RGB\nGT={CLASS_NAMES[rec['label']]} | Pred={CLASS_NAMES[rec['pred']]} | conf={rec['confidence']:.2f}", fontsize=9)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(h_ch, cmap="Blues_r")
    axes[row, 1].set_title("Hematoxylin", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(dab_ch, cmap="YlOrBr")
    axes[row, 2].set_title("DAB", fontsize=9)
    axes[row, 2].axis("off")

    im = axes[row, 3].imshow(attn_query_map, cmap="hot")
    axes[row, 3].set_title("Attention Map (14x14)", fontsize=9)
    axes[row, 3].axis("off")
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    axes[row, 4].imshow(img_rgb)
    axes[row, 4].imshow(attn_upsampled, cmap="hot", alpha=0.5)
    axes[row, 4].set_title("Attention Overlay", fontsize=9)
    axes[row, 4].axis("off")

plt.suptitle("Part 8 — Cross-Attention Maps (5 per class)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part8_attention_maps.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part8_attention_maps.png")


# ============================================================
# Cell 17 — Part 9: Average Attention Map per Class
# ============================================================

# Average attention map per class, using the full-inference maps of sampled images
class_avg_maps = {}
for c in range(NUM_CLASSES):
    idxs = [idx for idx in selected_idx if records[idx]["label"] == c]
    if not idxs:
        continue
    # Average the per-query max-attention maps across sampled images
    maps_simple = np.stack([
        attn_maps_full[idx][0].mean(axis=0)[1:, 1:].max(axis=1).reshape(GRID, GRID)
        for idx in idxs
    ])
    class_avg_maps[c] = maps_simple.mean(axis=0)

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(18, 4.5))
for c in range(NUM_CLASSES):
    ax = axes[c]
    if c in class_avg_maps:
        im = ax.imshow(class_avg_maps[c], cmap="hot")
        ax.set_title(f"{CLASS_NAMES[c]} (n={len([i for i in selected_idx if records[i]['label']==c])})", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

plt.suptitle("Part 9 — Average Attention Map by Class", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part9_average_attention_per_class.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part9_average_attention_per_class.png")


# ============================================================
# Cell 18 — Part 10: Head Diversity (cosine similarity between heads)
# ============================================================

# Use the sampled full attention maps (per-image per-head, averaged over queries)
# Head signature = mean attention vector over queries (197 dims, excluding CLS)
head_signatures = []   # list of (heads, 196)
for idx in selected_idx:
    attn = attn_maps_full[idx][0]                  # (heads, 197, 197)
    sig = attn[:, 1:, 1:].mean(axis=0)             # (heads, 196)
    head_signatures.append(sig)

# Mean head signature across images
head_sig_mean = np.stack(head_signatures).mean(axis=0)   # (heads, 196)
# Normalize
head_sig_norm = head_sig_mean / (np.linalg.norm(head_sig_mean, axis=1, keepdims=True) + 1e-12)
sim_matrix = head_sig_norm @ head_sig_norm.T       # (heads, heads)
mean_sim = float(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)].mean())

print("=" * 60)
print("Part 10 — Head Diversity")
print("=" * 60)
print(f"  Average pairwise head cosine similarity: {mean_sim:.4f}")
if mean_sim > 0.9:
    print("  -> Heads heavily redundant (similarity near 1)")
elif mean_sim > 0.7:
    print("  -> Heads partially redundant")
else:
    print("  -> Heads specialize (low similarity)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

im0 = axes[0].imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
axes[0].set_title("Head Cosine Similarity Matrix")
axes[0].set_xlabel("Head")
axes[0].set_ylabel("Head")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Clustered heatmap
g = sns.clustermap(sim_matrix, cmap="viridis", vmin=0, vmax=1, figsize=(6, 5))
g.fig.suptitle("Clustered Head Similarity", y=1.02)
g.savefig(os.path.join(OUT_DIR, "part10_head_similarity_clustered.png"), dpi=300, bbox_inches="tight")
plt.close(g.fig)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part10_head_similarity.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part10_head_similarity.png")


# ============================================================
# Cell 19 — Part 11: Correlation with Confidence
# ============================================================

confs = np.array([r["confidence"] for r in records])
entropy_img = np.array([r["layer_stats"][0]["mean_entropy"] for r in records])
max_img = np.array([r["layer_stats"][0]["mean_max_attn"] for r in records])

pearson_ent, _ = pearsonr(confs, entropy_img)
spearman_ent, _ = spearmanr(confs, entropy_img)
pearson_max, _ = pearsonr(confs, max_img)
spearman_max, _ = spearmanr(confs, max_img)

print("=" * 60)
print("Part 11 — Correlation with Confidence")
print("=" * 60)
print(f"  Confidence vs Attention Entropy:")
print(f"    Pearson  : {pearson_ent:.4f}")
print(f"    Spearman : {spearman_ent:.4f}")
print(f"  Confidence vs Max Attention:")
print(f"    Pearson  : {pearson_max:.4f}")
print(f"    Spearman : {spearman_max:.4f}")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(entropy_img, confs, alpha=0.3, s=10)
axes[0].set_xlabel("Mean attention entropy")
axes[0].set_ylabel("Confidence")
axes[0].set_title(f"Entropy vs Confidence (r={pearson_ent:.3f})")
axes[1].scatter(max_img, confs, alpha=0.3, s=10)
axes[1].set_xlabel("Mean max attention")
axes[1].set_ylabel("Confidence")
axes[1].set_title(f"Max Attention vs Confidence (r={pearson_max:.3f})")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part11_confidence_correlation.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part11_confidence_correlation.png")


# ============================================================
# Cell 20 — Part 12: Correct vs Incorrect Predictions
# ============================================================

correct_mask = np.array([r["pred"] == r["label"] for r in records])
entropy_correct = entropy_img[correct_mask]
entropy_wrong = entropy_img[~correct_mask]
max_correct = max_img[correct_mask]
max_wrong = max_img[~correct_mask]
dist_correct = all_dist[correct_mask]
dist_wrong = all_dist[~correct_mask]

print("=" * 60)
print("Part 12 — Correct vs Incorrect Predictions")
print("=" * 60)
print(f"  Correct: {correct_mask.sum()}, Incorrect: {(~correct_mask).sum()}")

# Mann-Whitney U tests
stat_ent, p_ent = mannwhitneyu(entropy_correct, entropy_wrong, alternative="two-sided")
stat_max, p_max = mannwhitneyu(max_correct, max_wrong, alternative="two-sided")
stat_dist, p_dist = mannwhitneyu(dist_correct, dist_wrong, alternative="two-sided")

print(f"  Attention entropy : correct={entropy_correct.mean():.4f} vs wrong={entropy_wrong.mean():.4f}  (Mann-Whitney p={p_ent:.4e})")
print(f"  Max attention     : correct={max_correct.mean():.4f} vs wrong={max_wrong.mean():.4f}  (Mann-Whitney p={p_max:.4e})")
print(f"  Attention distance: correct={dist_correct.mean():.3f} vs wrong={dist_wrong.mean():.3f}  (Mann-Whitney p={p_dist:.4e})")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].boxplot([entropy_correct, entropy_wrong], labels=["Correct", "Incorrect"], patch_artist=True)
axes[0].set_title("Attention Entropy")
axes[0].set_ylabel("Entropy")
axes[1].boxplot([max_correct, max_wrong], labels=["Correct", "Incorrect"], patch_artist=True)
axes[1].set_title("Max Attention")
axes[1].set_ylabel("Max attention")
axes[2].boxplot([dist_correct, dist_wrong], labels=["Correct", "Incorrect"], patch_artist=True)
axes[2].set_title("Attention Distance")
axes[2].set_ylabel("Distance (patches)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part12_correct_vs_incorrect.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: part12_correct_vs_incorrect.png")


# ============================================================
# Cell 21 — Part 13: Save Results
# ============================================================

import csv

csv_path = os.path.join(OUT_DIR, "attention_statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id", "true_label", "prediction", "confidence",
        "entropy", "max_attention", "distance", "gamma"
    ])
    for r in records:
        writer.writerow([
            r["filename"],
            r["label"],
            r["pred"],
            round(r["confidence"], 6),
            round(r["layer_stats"][0]["mean_entropy"], 6),
            round(r["layer_stats"][0]["mean_max_attn"], 6),
            round(r["layer_stats"][0]["mean_distance"], 6),
            round(gamma_eff, 6),
        ])

print(f"✅ CSV saved: {csv_path}")
print(f"   Rows: {len(records)}")
print(f"   All figures saved to: {OUT_DIR}")


# ============================================================
# Cell 22 — Final Report
# ============================================================

print("=" * 60)
print("Cross-Attention Diagnostic Report")
print("=" * 60)
print(f"Mean entropy: {entropy_pooled.mean():.3f}")
print(f"Entropy std: {entropy_pooled.std():.3f}")
print(f"Average gamma: {gamma_eff:.3f}")
print(f"Average attention distance: {dist_pooled.mean():.3f}")
print(f"Head diversity: {mean_sim:.3f}")
print(f"Correct prediction entropy: {entropy_correct.mean():.3f}")
print(f"Incorrect prediction entropy: {entropy_wrong.mean():.3f}")
print()

conclusions = []
if entropy_pooled.mean() < 3.0:
    conclusions.append("✓ Attention is focused")
else:
    conclusions.append("⚠ Attention nearly uniform")
if gamma_eff >= 0.05:
    conclusions.append("✓ Spatial bias retained")
else:
    conclusions.append("⚠ Gamma collapsed")
if mean_sim < 0.9:
    conclusions.append("✓ Heads are specialized")
else:
    conclusions.append("⚠ Heads redundant")
if p_ent > 0.05:
    conclusions.append("⚠ No significant entropy difference between correct/incorrect")

print("Overall conclusion:")
for c in conclusions:
    print(f"  {c}")
print()

if any("⚠" in c for c in conclusions):
    print("⚠ Cross-attention may not be contributing optimally.")
else:
    print("✓ Cross-attention appears healthy and focused.")
print("=" * 60)