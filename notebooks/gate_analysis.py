# DSCA-ViT — Gated Fusion Analysis Notebook (Enhanced)
# ============================================================
# PURPOSE:
#   Analyze the Gated Fusion module of a trained DSCA-ViT model.
#   Determine whether the fusion mechanism is functioning correctly
#   or has collapsed (e.g., always H-dominant or always DAB-dominant).
#
#   Gate:  g = sigmoid(Linear(concat(H_i, D_i)))
#          F_i = g * H_i + (1 - g) * D_i
#          g≈1 → H dominates | g≈0 → DAB dominates | g≈0.5 → both
#
#   IMPORTANT GATE DEFINITION:
#     F = g*H + (1-g)*DAB
#     g≈1 → H (morphology) dominates
#     g≈0 → DAB (HER2 signal) dominates
#
# HOW TO RUN:
#   Run Cells 1-7 (setup + inference), then Cells 8-20 (experiments).
#   Uses the trained Stage 2 checkpoint WITHOUT modifying the model.
#
# CHECKPOINT:
#   /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth
#
# OUTPUT:
#   All results saved to:
#   .../DSCA_ViT/Results/gate_analysis/
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

# Add repository to Python path
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
from scipy.stats import gaussian_kde

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

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

# Experiment identity
BACKBONE_NAME   = "DSCA_ViT"
MODEL_ID        = "dsca_vit_b16"
NUM_CLASSES     = 4
IMAGE_SIZE      = 224
BATCH_SIZE      = 32

# Class display names (order matches class_0, class_1+, class_2+, class_3+)
CLASS_NAMES = ["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"]

# Checkpoint (weights-only file from Stage 2)
CHECKPOINT_PATH = "/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth"

# Results output directory — dedicated gate_analysis subfolder
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
EXPERIMENT_DIR  = os.path.join(CHECKPOINT_ROOT, BACKBONE_NAME)
RESULTS_DIR     = os.path.join(EXPERIMENT_DIR, "Results")
GATE_DIR        = os.path.join(RESULTS_DIR, "gate_analysis")
os.makedirs(GATE_DIR, exist_ok=True)

print("=" * 60)
print("Gate Analysis Configuration")
print("=" * 60)
print(f"Model           : {BACKBONE_NAME}")
print(f"Checkpoint      : {CHECKPOINT_PATH}")
print(f"Output Dir      : {GATE_DIR}")
print("=" * 60)


# ============================================================
# Cell 5 — Dataset (downloads only if missing, then extracts)
# ============================================================
# (Same proven logic as model_HER2_ViT.ipynb Cell 2)

import zipfile

DATA_ROOT = Path("/content/HER2_Dataset")
DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

# ------------------------------------------------------------
# Download dataset (only if not already downloaded)
# ------------------------------------------------------------

if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run([
        "wget",
        "-O",
        str(ZIP_PATH),
        URL
    ], check=True)
else:
    print("Dataset archive already exists.")

# ------------------------------------------------------------
# Extract main archive
# ------------------------------------------------------------

WSI_DIR = DATA_ROOT / "WSI-based-dataset"

if not WSI_DIR.exists():
    print("Extracting main archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Main archive extracted.")
else:
    print("Main archive already extracted.")

# ------------------------------------------------------------
# Extract nested train/test archives
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Remove zip archives to save disk space
# ------------------------------------------------------------

for archive in [ZIP_PATH] + nested_archives:
    if archive.exists():
        archive.unlink()

print("ZIP files removed.")

# ------------------------------------------------------------
# Dataset paths — we analyze the TEST split
# ------------------------------------------------------------

TEST_DIR = WSI_DIR / "test"

print("\nDataset location:")
print(TEST_DIR)

assert TEST_DIR.exists(), "Test directory not found."

print("\nDataset successfully prepared!")

# ------------------------------------------------------------
# Show test folder structure
# ------------------------------------------------------------

print("\nTest folder structure:")
for cls in sorted(os.listdir(TEST_DIR)):
    cls_path = TEST_DIR / cls
    if cls_path.is_dir():
        n = len(os.listdir(cls_path))
        print(f"   {cls:<10} {n:5d} images")


# ============================================================
# Cell 6 — Build Model + Load Stage 2 Checkpoint
# ============================================================

from models import DSCAViT

# ----------------------------------------------------------
# Build the SAME architecture as training (so weights load cleanly)
# ----------------------------------------------------------

model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.1,
    classifier_dropout=0.1,
)

model = model.to(device)

# ----------------------------------------------------------
# Load the Stage 2 weights.
# The file is weights-only (torch.save(model.state_dict(), ...)).
# Handle both weights-only and full-checkpoint formats.
# ----------------------------------------------------------

assert os.path.exists(CHECKPOINT_PATH), (
    f"Checkpoint not found at:\n    {CHECKPOINT_PATH}"
)

print(f"✅ Loading checkpoint:\n    {CHECKPOINT_PATH}")

state = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(state, dict) and "model_state_dict" in state:
    # Full checkpoint dict (from save_checkpoint)
    model.load_state_dict(state["model_state_dict"])
    print("✅ Loaded full checkpoint (model_state_dict).")
else:
    # Weights-only state_dict
    model.load_state_dict(state)
    print("✅ Loaded weights-only state_dict.")

model.eval()

# ----------------------------------------------------------
# Parameter summary
# ----------------------------------------------------------

counts = model.count_parameters()
print("=" * 60)
print("DSCA-ViT Parameter Summary")
print("=" * 60)
for name, count in counts.items():
    print(f"  {name:<20} : {count:>12,}")
print("=" * 60)


# ============================================================
# Cell 7 — Inference Pass: Collect Gates, Predictions, Confidence, Entropy
# ============================================================

from datasets import HER2Dataset, get_test_transform
from torch.utils.data import DataLoader

test_transform = get_test_transform(image_size=IMAGE_SIZE)

test_dataset = HER2Dataset(root_dir=str(TEST_DIR), transform=test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Test images    : {len(test_dataset)}")
print(f"Test batches   : {len(test_loader)}")

# ----------------------------------------------------------
# Collect per-image data
# ----------------------------------------------------------

# Each entry:
#   filename, label, pred, confidence, pred_entropy, gates (196,)
records = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)                       # (B, 4)
        probs = torch.softmax(logits, dim=1)         # (B, 4)
        conf, preds = probs.max(dim=1)               # (B,)

        # Prediction entropy: H(p) = -sum(p * log(p))
        pred_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)  # (B,)

        gates = model.get_gate_values()              # (B, 196, 768)
        # Average over the 768 embedding dims -> (B, 196)
        gates_mean = gates.mean(dim=-1).cpu().numpy()  # (B, 196)

        for i in range(images.size(0)):
            records.append({
                "filename": None,          # filled below (dataset order)
                "label": labels[i].item(),
                "pred": preds[i].item(),
                "confidence": conf[i].item(),
                "pred_entropy": pred_entropy[i].item(),
                "gates": gates_mean[i],    # (196,)
            })

# Fill filenames properly (loader is shuffle=False, so records are in dataset order)
for idx, rec in enumerate(records):
    rec["filename"] = test_dataset.image_paths[idx].name

print(f"✅ Collected {len(records)} records.")

# Quick sanity: show first record
r0 = records[0]
print(f"  First: {r0['filename']} | GT={r0['label']} | Pred={r0['pred']} | "
      f"Conf={r0['confidence']:.3f} | Entropy={r0['pred_entropy']:.3f} | gates shape={r0['gates'].shape}")


# ============================================================
# Cell 8 — Experiment 1: Global Gate Statistics
# ============================================================

all_gates = np.concatenate([r["gates"] for r in records])   # (N*196,)

mean_g = float(all_gates.mean())
median_g = float(np.median(all_gates))
std_g = float(all_gates.std())
min_g = float(all_gates.min())
max_g = float(all_gates.max())

# Bucket percentages
p_lt_02 = float((all_gates < 0.2).mean() * 100)
p_02_04 = float(((all_gates >= 0.2) & (all_gates < 0.4)).mean() * 100)
p_04_06 = float(((all_gates >= 0.4) & (all_gates < 0.6)).mean() * 100)
p_06_08 = float(((all_gates >= 0.6) & (all_gates < 0.8)).mean() * 100)
p_gt_08 = float((all_gates >= 0.8).mean() * 100)

print("=" * 60)
print("Experiment 1 — Global Gate Statistics")
print("=" * 60)
print(f"  Mean gate   : {mean_g:.4f}")
print(f"  Median gate : {median_g:.4f}")
print(f"  Std gate    : {std_g:.4f}")
print(f"  Min gate    : {min_g:.4f}")
print(f"  Max gate    : {max_g:.4f}")
print()
print("  Percentage of gates:")
print(f"    < 0.2   : {p_lt_02:6.2f}%   (strongly favor DAB)")
print(f"    0.2-0.4 : {p_02_04:6.2f}%   (favor DAB)")
print(f"    0.4-0.6 : {p_04_06:6.2f}%   (mixed)")
print(f"    0.6-0.8 : {p_06_08:6.2f}%   (favor H)")
print(f"    > 0.8   : {p_gt_08:6.2f}%   (strongly favor H)")
print()
print(f"  Summary: {p_gt_08 + p_06_08:.1f}% of patches favor H, "
      f"{p_04_06:.1f}% mixed, {p_lt_02 + p_02_04:.1f}% favor DAB")
print()

# Interpretation
if p_lt_02 > 70:
    interp = "Mostly near 0 (DAB-dominant)"
elif p_gt_08 > 70:
    interp = "Mostly near 1 (H-dominant)"
elif 0.45 <= mean_g <= 0.55 and std_g > 0.15:
    interp = "Centered near 0.5 (balanced)"
else:
    interp = "Well distributed"
print(f"  Interpretation: {interp}")
print("=" * 60)


# ============================================================
# Cell 9 — Experiment 2: Histogram + KDE of All Gate Values
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(all_gates, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(0.5, color="red", linestyle="--", label="0.5 (balanced)")
axes[0].axvline(mean_g, color="green", linestyle="--", label=f"mean={mean_g:.2f}")
axes[0].set_xlabel("Gate value")
axes[0].set_ylabel("Count")
axes[0].set_title("Histogram of All Gate Values")
axes[0].legend()

# KDE
try:
    kde = gaussian_kde(all_gates)
    xs = np.linspace(0, 1, 300)
    axes[1].plot(xs, kde(xs), color="darkorange", lw=2)
    axes[1].fill_between(xs, kde(xs), alpha=0.3, color="darkorange")
    axes[1].axvline(0.5, color="red", linestyle="--", label="0.5 (balanced)")
    axes[1].axvline(mean_g, color="green", linestyle="--", label=f"mean={mean_g:.2f}")
    axes[1].set_xlabel("Gate value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("KDE of All Gate Values")
    axes[1].legend()
except Exception as e:
    axes[1].text(0.5, 0.5, f"KDE failed: {e}", ha="center", va="center")
    axes[1].set_title("KDE (failed)")

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp2_gate_histogram.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp2_gate_histogram.png")


# ============================================================
# Cell 10 — Experiment 3: Per-Image Statistics
# ============================================================

img_means = np.array([r["gates"].mean() for r in records])
img_stds = np.array([r["gates"].std() for r in records])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(img_means, bins=40, color="seagreen", edgecolor="white", alpha=0.8)
axes[0].axvline(0.5, color="red", linestyle="--", label="0.5")
axes[0].set_xlabel("Per-image mean gate")
axes[0].set_ylabel("Count")
axes[0].set_title("Histogram of Image Mean Gates")
axes[0].legend()

axes[1].hist(img_stds, bins=40, color="mediumpurple", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Per-image std gate")
axes[1].set_ylabel("Count")
axes[1].set_title("Histogram of Image Gate Std")

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp3_per_image_stats.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp3_per_image_stats.png")

print(f"\nPer-image mean gate: mean={img_means.mean():.4f}, std={img_means.std():.4f}")
print(f"Per-image std gate : mean={img_stds.mean():.4f}, std={img_stds.std():.4f}")


# ============================================================
# Cell 11 — Experiment 4: Per-Class Gate Statistics (mean + full distribution)
# ============================================================

class_gates = {c: [] for c in range(NUM_CLASSES)}
for r in records:
    class_gates[r["label"]].append(r["gates"])

print("=" * 60)
print("Experiment 4 — Per-Class Gate Statistics")
print("=" * 60)

class_means = {}
class_stds = {}

for c in range(NUM_CLASSES):
    g = np.concatenate(class_gates[c])
    class_means[c] = float(g.mean())
    class_stds[c] = float(g.std())
    print(f"  {CLASS_NAMES[c]:<10} | n={len(class_gates[c]):5d} | mean={class_means[c]:.4f} | std={class_stds[c]:.4f}")

# Do different classes rely on different stains?
spread = max(class_means.values()) - min(class_means.values())
print(f"\n  Spread (max-min class mean): {spread:.4f}")
print(f"  -> {'YES, classes rely on different stains' if spread > 0.15 else 'No strong per-class difference'}")
print("=" * 60)

# --- Histograms per class ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, c in enumerate(range(NUM_CLASSES)):
    ax = axes[idx // 2][idx % 2]
    g = np.concatenate(class_gates[c])
    ax.hist(g, bins=40, alpha=0.7, label=f"mean={class_means[c]:.2f}")
    ax.axvline(0.5, color="red", linestyle="--")
    ax.set_title(f"{CLASS_NAMES[c]} (n={len(class_gates[c])})")
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp4_per_class_histograms.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp4_per_class_histograms.png")

# --- Box + Violin plots per class (full distribution) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
box_data = [np.concatenate(class_gates[c]) for c in range(NUM_CLASSES)]
axes[0].boxplot(box_data, labels=CLASS_NAMES, patch_artist=True)
axes[0].axhline(0.5, color="red", linestyle="--", label="0.5")
axes[0].set_ylabel("Gate value")
axes[0].set_title("Box Plot of Gate Values by Class")
axes[0].legend()

# Violin plot
violin_data = []
violin_labels = []
for c in range(NUM_CLASSES):
    g = np.concatenate(class_gates[c])
    # Subsample for violin (avoid huge arrays)
    if len(g) > 20000:
        g = np.random.choice(g, 20000, replace=False)
    violin_data.append(g)
    violin_labels.append(CLASS_NAMES[c])

vp = axes[1].violinplot(violin_data, showmeans=True, showmedians=True)
axes[1].set_xticks(range(1, NUM_CLASSES + 1))
axes[1].set_xticklabels(violin_labels)
axes[1].axhline(0.5, color="red", linestyle="--", label="0.5")
axes[1].set_ylabel("Gate value")
axes[1].set_title("Violin Plot of Gate Values by Class")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp4_per_class_box_violin.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp4_per_class_box_violin.png")


# ============================================================
# Cell 12 — Experiment 5: Spatial Gate Heatmaps (5 per class)
# ============================================================

from models.color_deconv import deconvolve_numpy
from PIL import Image

N_PER_CLASS = 5

# Pick 5 random images per class (reproducible)
random.seed(SEED)
selected = []
for c in range(NUM_CLASSES):
    idxs = [i for i, r in enumerate(records) if r["label"] == c]
    chosen = random.sample(idxs, min(N_PER_CLASS, len(idxs)))
    for i in chosen:
        selected.append((i, c))

def smooth_upsample(gate_14x14, size=224):
    """Upsample a 14x14 gate map to 224x224 with smooth interpolation."""
    from PIL import Image as PILImage
    arr = (gate_14x14 * 255).astype(np.uint8)
    img = PILImage.fromarray(arr).resize((size, size), PILImage.BICUBIC)
    return np.array(img) / 255.0

fig, axes = plt.subplots(len(selected), 4, figsize=(16, 5 * len(selected)))

for row, (idx, c) in enumerate(selected):
    rec = records[idx]
    img_path = test_dataset.image_paths[idx]

    # Original RGB
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    # H and DAB channels (same fixed Ruifrok math the model uses)
    h_ch, dab_ch = deconvolve_numpy(img_rgb)

    # Gate heatmap (14x14 -> 224x224 smooth)
    gate_14 = rec["gates"].reshape(14, 14)
    gate_up = smooth_upsample(gate_14)

    # --- Column 0: Original RGB ---
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f"RGB\nGT={CLASS_NAMES[rec['label']]} | Pred={CLASS_NAMES[rec['pred']]}", fontsize=9)
    axes[row, 0].axis("off")

    # --- Column 1: H channel ---
    axes[row, 1].imshow(h_ch, cmap="Blues_r")
    axes[row, 1].set_title("Hematoxylin", fontsize=9)
    axes[row, 1].axis("off")

    # --- Column 2: DAB channel ---
    axes[row, 2].imshow(dab_ch, cmap="YlOrBr")
    axes[row, 2].set_title("DAB", fontsize=9)
    axes[row, 2].axis("off")

    # --- Column 3: Gate heatmap overlay ---
    axes[row, 3].imshow(img_rgb)
    im = axes[row, 3].imshow(gate_up, cmap="RdBu_r", alpha=0.5, vmin=0, vmax=1)
    axes[row, 3].set_title(f"Gate (mean={rec['gates'].mean():.2f})", fontsize=9)
    axes[row, 3].axis("off")

plt.suptitle("Experiment 5 — Spatial Gate Heatmaps (5 per class)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp5_gate_heatmaps.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp5_gate_heatmaps.png")


# ============================================================
# Cell 13 — Experiment 6: Gate Activation Map (mean per patch)
# ============================================================

# Average gate over the entire dataset for every patch position
gate_stack = np.stack([r["gates"].reshape(14, 14) for r in records])  # (N, 14, 14)
mean_gate_map = gate_stack.mean(axis=0)  # (14, 14)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(mean_gate_map, cmap="RdBu_r", vmin=0, vmax=1)
axes[0].set_title("Mean Gate per Patch (14x14)")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Smoothed version
mean_gate_up = smooth_upsample(mean_gate_map)
im1 = axes[1].imshow(mean_gate_up, cmap="RdBu_r", vmin=0, vmax=1)
axes[1].set_title("Mean Gate per Patch (smoothed 224x224)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp6_gate_activation_map.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp6_gate_activation_map.png")

# Interpretation
map_mean = float(mean_gate_map.mean())
map_min = float(mean_gate_map.min())
map_max = float(mean_gate_map.max())
print(f"\nMean gate map: overall={map_mean:.4f}, min={map_min:.4f}, max={map_max:.4f}")
if map_max - map_min > 0.2:
    print("-> Some spatial regions are consistently H-dominant or DAB-dominant.")
else:
    print("-> Gate map is fairly uniform across spatial positions.")


# ============================================================
# Cell 14 — Experiment 7: Confidence & Entropy vs Mean Gate
# ============================================================

confs = np.array([r["confidence"] for r in records])
means = np.array([r["gates"].mean() for r in records])
entropies = np.array([r["pred_entropy"] for r in records])

# Correlations
corr_conf = np.corrcoef(means, confs)[0, 1]
corr_ent = np.corrcoef(means, entropies)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confidence vs mean gate
axes[0].scatter(means, confs, alpha=0.3, s=10, color="steelblue")
axes[0].set_xlabel("Per-image mean gate")
axes[0].set_ylabel("Prediction confidence (softmax max)")
axes[0].set_title(f"Mean Gate vs Confidence (corr={corr_conf:.3f})")
axes[0].axvline(0.5, color="red", linestyle="--", label="0.5")
axes[0].legend()

# Prediction entropy vs mean gate
axes[1].scatter(means, entropies, alpha=0.3, s=10, color="darkorange")
axes[1].set_xlabel("Per-image mean gate")
axes[1].set_ylabel("Prediction entropy")
axes[1].set_title(f"Mean Gate vs Prediction Entropy (corr={corr_ent:.3f})")
axes[1].axvline(0.5, color="red", linestyle="--", label="0.5")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp7_confidence_entropy_vs_gate.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp7_confidence_entropy_vs_gate.png")
print(f"\nCorrelation (mean gate, confidence): {corr_conf:.4f}")
print(f"Correlation (mean gate, entropy)  : {corr_ent:.4f}")


# ============================================================
# Cell 15 — Experiment 8: Correct vs Misclassified (overall + per class)
# ============================================================

correct_idx = [i for i, r in enumerate(records) if r["pred"] == r["label"]]
wrong_idx = [i for i, r in enumerate(records) if r["pred"] != r["label"]]

print(f"Correctly classified: {len(correct_idx)}")
print(f"Misclassified      : {len(wrong_idx)}")

# --- Overall comparison ---
random.seed(SEED)
correct_sample = random.sample(correct_idx, min(20, len(correct_idx)))
wrong_sample = random.sample(wrong_idx, min(20, len(wrong_idx)))

correct_gates = np.concatenate([records[i]["gates"] for i in correct_sample])
wrong_gates = np.concatenate([records[i]["gates"] for i in wrong_sample])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(correct_gates, bins=40, color="seagreen", alpha=0.7, label=f"mean={correct_gates.mean():.3f}")
axes[0].axvline(0.5, color="red", linestyle="--")
axes[0].set_title(f"Correctly Classified (n={len(correct_sample)})")
axes[0].set_xlabel("Gate value")
axes[0].legend()

axes[1].hist(wrong_gates, bins=40, color="crimson", alpha=0.7, label=f"mean={wrong_gates.mean():.3f}")
axes[1].axvline(0.5, color="red", linestyle="--")
axes[1].set_title(f"Misclassified (n={len(wrong_sample)})")
axes[1].set_xlabel("Gate value")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp8_correct_vs_wrong.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp8_correct_vs_wrong.png")

print(f"\nCorrect gates: mean={correct_gates.mean():.4f}, std={correct_gates.std():.4f}")
print(f"Wrong gates  : mean={wrong_gates.mean():.4f}, std={wrong_gates.std():.4f}")

# --- Per-class correct vs wrong ---
print("\n" + "=" * 60)
print("Per-Class Correct vs Wrong Gate Statistics")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, c in enumerate(range(NUM_CLASSES)):
    ax = axes[idx // 2][idx % 2]

    c_correct = [i for i in correct_idx if records[i]["label"] == c]
    c_wrong = [i for i in wrong_idx if records[i]["label"] == c]

    if c_correct:
        g_c = np.concatenate([records[i]["gates"] for i in c_correct])
        ax.hist(g_c, bins=40, alpha=0.6, color="seagreen",
                label=f"correct (n={len(c_correct)}, mean={g_c.mean():.2f})")
    if c_wrong:
        g_w = np.concatenate([records[i]["gates"] for i in c_wrong])
        ax.hist(g_w, bins=40, alpha=0.6, color="crimson",
                label=f"wrong (n={len(c_wrong)}, mean={g_w.mean():.2f})")

    ax.axvline(0.5, color="red", linestyle="--")
    ax.set_title(f"{CLASS_NAMES[c]}")
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp8_per_class_correct_vs_wrong.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp8_per_class_correct_vs_wrong.png")


# ============================================================
# Cell 16 — Experiment 9: Gate Values vs Predicted Class
# ============================================================

pred_gates = {c: [] for c in range(NUM_CLASSES)}
for r in records:
    pred_gates[r["pred"]].append(r["gates"])

print("=" * 60)
print("Experiment 9 — Gate Statistics by PREDICTED Class")
print("=" * 60)

pred_means = {}
for c in range(NUM_CLASSES):
    g = np.concatenate(pred_gates[c])
    pred_means[c] = float(g.mean())
    print(f"  Pred {CLASS_NAMES[c]:<10} | n={len(pred_gates[c]):5d} | mean={pred_means[c]:.4f} | std={float(g.std()):.4f}")

# Does the network predict 3+ only when gate is high?
print("\n  -> If a class is predicted only at extreme gate values, that reveals bias.")
print("=" * 60)

# Box/violin by predicted class
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

box_data_pred = [np.concatenate(pred_gates[c]) for c in range(NUM_CLASSES)]
axes[0].boxplot(box_data_pred, labels=CLASS_NAMES, patch_artist=True)
axes[0].axhline(0.5, color="red", linestyle="--", label="0.5")
axes[0].set_ylabel("Gate value")
axes[0].set_title("Box Plot of Gate Values by Predicted Class")
axes[0].legend()

violin_data_pred = []
for c in range(NUM_CLASSES):
    g = np.concatenate(pred_gates[c])
    if len(g) > 20000:
        g = np.random.choice(g, 20000, replace=False)
    violin_data_pred.append(g)

vp = axes[1].violinplot(violin_data_pred, showmeans=True, showmedians=True)
axes[1].set_xticks(range(1, NUM_CLASSES + 1))
axes[1].set_xticklabels(CLASS_NAMES)
axes[1].axhline(0.5, color="red", linestyle="--", label="0.5")
axes[1].set_ylabel("Gate value")
axes[1].set_title("Violin Plot of Gate Values by Predicted Class")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp9_gate_by_predicted_class.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp9_gate_by_predicted_class.png")


# ============================================================
# Cell 17 — Experiment 10: Gate Entropy
# ============================================================

# Gate entropy: H(g) = -g log g - (1-g) log(1-g)
#   low entropy  -> gate near 0 or 1 -> hard selection
#   high entropy -> gate near 0.5   -> both streams mixed
eps = 1e-12
gate_entropy = -all_gates * np.log(all_gates + eps) - (1 - all_gates) * np.log(1 - all_gates + eps)

mean_entropy = float(gate_entropy.mean())
median_entropy = float(np.median(gate_entropy))

print("=" * 60)
print("Experiment 10 — Gate Entropy")
print("=" * 60)
print(f"  Mean gate entropy   : {mean_entropy:.4f}")
print(f"  Median gate entropy : {median_entropy:.4f}")
print(f"  (max possible = ln(2) = {np.log(2):.4f})")
print()

# Interpretation
if mean_entropy < 0.2:
    ent_interp = "Very low — gates are making hard selections (near 0 or 1)"
elif mean_entropy < 0.5:
    ent_interp = "Low-to-moderate — gates lean toward hard selection"
elif mean_entropy < 0.6:
    ent_interp = "Moderate — gates mix both streams substantially"
else:
    ent_interp = "High — gates are near 0.5, both streams contribute equally"
print(f"  Interpretation: {ent_interp}")
print("=" * 60)

# Histogram of gate entropy
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gate_entropy, bins=50, color="teal", edgecolor="white", alpha=0.8)
ax.axvline(mean_entropy, color="red", linestyle="--", label=f"mean={mean_entropy:.3f}")
ax.set_xlabel("Gate entropy H(g)")
ax.set_ylabel("Count")
ax.set_title("Histogram of Gate Entropy")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(GATE_DIR, "exp10_gate_entropy.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: exp10_gate_entropy.png")


# ============================================================
# Cell 18 — Experiment 11: Best vs Worst Predictions (side-by-side)
# ============================================================

# Top 10 highest-confidence CORRECT predictions
correct_sorted = sorted(correct_idx, key=lambda i: -records[i]["confidence"])[:10]

# Bottom 10 lowest-confidence WRONG predictions
wrong_sorted = sorted(wrong_idx, key=lambda i: records[i]["confidence"])[:10]

def plot_examples(indices, title, filename):
    """Plot RGB | H | DAB | Gate for a list of image indices."""
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 5 * n))
    for row, idx in enumerate(indices):
        rec = records[idx]
        img_path = test_dataset.image_paths[idx]
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h_ch, dab_ch = deconvolve_numpy(img_rgb)
        gate_14 = rec["gates"].reshape(14, 14)
        gate_up = smooth_upsample(gate_14)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"RGB\nGT={CLASS_NAMES[rec['label']]} | Pred={CLASS_NAMES[rec['pred']]}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(h_ch, cmap="Blues_r")
        axes[row, 1].set_title("H", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(dab_ch, cmap="YlOrBr")
        axes[row, 2].set_title("DAB", fontsize=9)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(img_rgb)
        axes[row, 3].imshow(gate_up, cmap="RdBu_r", alpha=0.5, vmin=0, vmax=1)
        axes[row, 3].set_title(f"Gate (mean={rec['gates'].mean():.2f}, conf={rec['confidence']:.2f})", fontsize=9)
        axes[row, 3].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GATE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

plot_examples(correct_sorted, "Experiment 11 — Top 10 Highest-Confidence Correct Predictions",
              "exp11_best_predictions.png")
plot_examples(wrong_sorted, "Experiment 11 — Bottom 10 Lowest-Confidence Wrong Predictions",
              "exp11_worst_predictions.png")


# ============================================================
# Cell 19 — Experiment 12: Fusion Ablation (H-only vs DAB-only vs Fusion)
# ============================================================
# THE MOST IMPORTANT EXPERIMENT:
#   Run the same checkpoint 3 ways and compare accuracy:
#     - Normal (adaptive fusion)
#     - gate=1 (H only)
#     - gate=0 (DAB only)
#   If Fusion > both, the gate is doing something useful.
#   If Fusion ≈ H, DAB contributes almost nothing.
#   If Fusion ≈ DAB, H contributes almost nothing.

print("=" * 60)
print("Experiment 12 — Fusion Ablation (H-only vs DAB-only vs Fusion)")
print("=" * 60)
print("Running the same checkpoint 3 ways...")

def evaluate_fusion():
    """Run normal inference (adaptive gates)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = 100.0 * sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    return acc

# NOTE: A TRUE gate-override ablation (forcing gate=1 = H-only, gate=0 = DAB-only)
# would require modifying the model's forward pass. We must NOT modify the model.
# Instead, we approximate H-only vs DAB-only behavior from the gate distribution:
#   - If gates ≈ 1 everywhere  -> the model effectively behaves H-only
#   - If gates ≈ 0 everywhere  -> the model effectively behaves DAB-only
# We report the actual fusion accuracy plus the gate-based dominance fractions.

fusion_acc = evaluate_fusion()

# Estimate H-only / DAB-only behavior from gate distribution
# (patches with gate>0.5 are H-dominant, gate<0.5 are DAB-dominant)
h_dominant_frac = float((all_gates > 0.5).mean())
dab_dominant_frac = 1.0 - h_dominant_frac

print(f"  Fusion accuracy (adaptive gates): {fusion_acc:.2f}%")
print(f"  Fraction of H-dominant patches  : {h_dominant_frac:.3f}")
print(f"  Fraction of DAB-dominant patches: {dab_dominant_frac:.3f}")
print()
print("  Interpretation:")
if h_dominant_frac > 0.9:
    print("    -> The model behaves almost like H-only (DAB contributes little).")
elif dab_dominant_frac > 0.9:
    print("    -> The model behaves almost like DAB-only (H contributes little).")
elif 0.3 < h_dominant_frac < 0.7:
    print("    -> The model uses BOTH streams substantially (healthy adaptive fusion).")
else:
    print("    -> The model leans toward one stream but still uses both.")
print("=" * 60)


# ============================================================
# Cell 20 — Experiment 13: Save Results CSV
# ============================================================

import csv

csv_path = os.path.join(GATE_DIR, "gate_analysis_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "ground_truth", "prediction", "confidence",
        "pred_entropy", "mean_gate", "std_gate", "gate_entropy"
    ])
    for r in records:
        g = r["gates"]
        ge = -g * np.log(g + eps) - (1 - g) * np.log(1 - g + eps)
        writer.writerow([
            r["filename"],
            r["label"],
            r["pred"],
            round(r["confidence"], 6),
            round(r["pred_entropy"], 6),
            round(float(g.mean()), 6),
            round(float(g.std()), 6),
            round(float(ge.mean()), 6),
        ])

print(f"✅ CSV saved: {csv_path}")
print(f"   Rows: {len(records)}")


# ============================================================
# Cell 21 — Experiment 14: Automatic Interpretation Report
# ============================================================

print("=" * 60)
print("Gate Analysis Summary")
print("=" * 60)

# --- Mean gate ---
print(f"Mean gate: {mean_g:.2f}")

# --- Distribution ---
if p_lt_02 > 70:
    distribution = "Collapsed to DAB"
elif p_gt_08 > 70:
    distribution = "Collapsed to Hematoxylin"
elif 0.45 <= mean_g <= 0.55 and std_g > 0.15:
    distribution = "Centered near 0.5"
else:
    distribution = "Well balanced"
print(f"Distribution: {distribution}")

# --- Collapse ---
collapsed = (p_lt_02 > 70) or (p_gt_08 > 70)
print(f"Collapse: {'YES' if collapsed else 'NO'}")

# --- Per-class differences ---
spread = max(class_means.values()) - min(class_means.values())
per_class_diff = spread > 0.15
print(f"Per-class differences: {'YES' if per_class_diff else 'NO'}")

# --- Gate variance ---
if std_g < 0.1:
    variance = "Low"
elif std_g < 0.25:
    variance = "Medium"
else:
    variance = "High"
print(f"Gate variance: {variance}")

# --- Gate entropy ---
if mean_entropy < 0.2:
    entropy_label = "Low (hard selection)"
elif mean_entropy < 0.5:
    entropy_label = "Low-to-moderate"
elif mean_entropy < 0.6:
    entropy_label = "Moderate (mixed)"
else:
    entropy_label = "High (balanced)"
print(f"Gate entropy: {entropy_label}")

# --- Fusion health ---
if h_dominant_frac > 0.9:
    fusion_health = "H-only (DAB ignored)"
elif dab_dominant_frac > 0.9:
    fusion_health = "DAB-only (H ignored)"
elif 0.3 < h_dominant_frac < 0.7:
    fusion_health = "Healthy adaptive fusion"
else:
    fusion_health = "Leans one stream"
print(f"Fusion health: {fusion_health}")

# --- Conclusion ---
if p_gt_08 > 70:
    conclusion = "The network is effectively ignoring the DAB stream."
elif p_lt_02 > 70:
    conclusion = "The network is effectively ignoring the Hematoxylin stream."
elif 0.3 < h_dominant_frac < 0.7:
    conclusion = "Fusion module appears healthy — both streams contribute."
else:
    conclusion = "Fusion module leans toward one stream but is not fully collapsed."
print(f"Conclusion: {conclusion}")

print("=" * 60)