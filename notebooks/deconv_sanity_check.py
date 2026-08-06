# ============================================================
# DSCA-ViT — Color Deconvolution Sanity Check
# ============================================================
#
# PURPOSE:
#   Verify that the Ruifrok & Johnston color deconvolution
#   correctly separates HER2 IHC images into:
#     - Hematoxylin (H) : blue/purple nuclei morphology
#     - DAB             : brown HER2 membrane staining
#
# WHAT TO LOOK FOR:
#   ✅ DAB channel shows brown membrane signal (cell borders)
#   ✅ H channel shows blue nuclei (cell centers)
#   ✅ Channels are visually "clean" — not mixed
#   ❌ If DAB looks like a grayscale copy of RGB → vectors are wrong
#   ❌ If H contains brown signal → channels are swapped or mixed
#
# This is the MOST IMPORTANT sanity check before training.
# If the deconvolution is wrong, the entire dual-stream
# architecture loses its biological meaning.
#
# ============================================================

# ============================================================
# Cell 1 — Setup
# ============================================================

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Mount Google Drive (for dataset access if needed)
# from google.colab import drive
# drive.mount("/content/drive")

# Add repo to path
REPO_DIR = "/content/DSCA-ViT"
sys.path.insert(0, REPO_DIR)

from models.color_deconv import deconvolve_numpy

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("✅ Setup complete.")


# ============================================================
# Cell 2 — Collect Image Paths from Dataset
# ============================================================

DATA_ROOT = Path("/content/HER2_Dataset/WSI-based-dataset")

# Use both train and test for broader coverage
SPLITS = ["train", "test"]
CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

all_image_paths = []
all_image_labels = []

for split in SPLITS:
    for cls in CLASSES:
        cls_dir = DATA_ROOT / split / cls
        if not cls_dir.exists():
            print(f"⚠️  Missing: {cls_dir}")
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(img_path)
                all_image_labels.append(f"{split}/{cls}")

print(f"Total images found: {len(all_image_paths)}")
print(f"Classes: {CLASSES}")


# ============================================================
# Cell 3 — Sample 20 Random Patches (5 per class)
# ============================================================

NUM_PER_CLASS = 5
sampled_paths = []
sampled_labels = []

for cls in CLASSES:
    # Collect all images for this class (from any split)
    cls_paths = [
        (p, l) for p, l in zip(all_image_paths, all_image_labels)
        if cls in l
    ]

    if len(cls_paths) == 0:
        print(f"⚠️  No images found for {cls}")
        continue

    # Sample up to NUM_PER_CLASS
    k = min(NUM_PER_CLASS, len(cls_paths))
    chosen = random.sample(cls_paths, k)

    for p, l in chosen:
        sampled_paths.append(p)
        sampled_labels.append(cls)

print(f"\nSampled {len(sampled_paths)} patches:")
for cls in CLASSES:
    n = sampled_labels.count(cls)
    print(f"  {cls:<10}: {n} patches")


# ============================================================
# Cell 4 — Run Deconvolution & Plot (20 patches, 3 columns each)
# ============================================================

def normalize_for_display(channel: np.ndarray) -> np.ndarray:
    """Normalize a single-channel image to [0, 1] for display."""
    vmin, vmax = channel.min(), channel.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(channel)
    return (channel - vmin) / (vmax - vmin)


N = len(sampled_paths)
fig, axes = plt.subplots(N, 3, figsize=(12, 3.5 * N))

# Ensure axes is 2D even if N=1
if N == 1:
    axes = axes[np.newaxis, :]

for i, (img_path, label) in enumerate(zip(sampled_paths, sampled_labels)):

    # Load image
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    # Deconvolve
    h_channel, dab_channel = deconvolve_numpy(img_rgb)

    # --- Column 1: Original RGB ---
    axes[i, 0].imshow(img_rgb)
    axes[i, 0].set_title(f"RGB — {label}", fontsize=11, fontweight="bold")
    axes[i, 0].axis("off")

    # --- Column 2: Hematoxylin ---
    axes[i, 1].imshow(normalize_for_display(h_channel), cmap="Blues_r")
    axes[i, 1].set_title("Hematoxylin (nuclei)", fontsize=11)
    axes[i, 1].axis("off")

    # --- Column 3: DAB ---
    axes[i, 2].imshow(normalize_for_display(dab_channel), cmap="YlOrBr")
    axes[i, 2].set_title("DAB (HER2 membrane)", fontsize=11)
    axes[i, 2].axis("off")

# Add column headers at top
fig.text(0.22, 0.995, "Original RGB", ha="center", fontsize=14, fontweight="bold")
fig.text(0.52, 0.995, "Hematoxylin (H)", ha="center", fontsize=14, fontweight="bold", color="#2166ac")
fig.text(0.82, 0.995, "DAB (HER2)", ha="center", fontsize=14, fontweight="bold", color="#8c510a")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("/content/deconv_sanity_check_20patches.png", dpi=120, bbox_inches="tight")
plt.show()

print("\n✅ Saved: /content/deconv_sanity_check_20patches.png")


# ============================================================
# Cell 5 — Per-Class Summary Statistics
# ============================================================

print("\n" + "=" * 60)
print("Color Deconvolution Statistics (per class)")
print("=" * 60)

for cls in CLASSES:

    cls_indices = [i for i, l in enumerate(sampled_labels) if l == cls]

    if not cls_indices:
        continue

    h_means = []
    dab_means = []
    h_maxes = []
    dab_maxes = []

    for idx in cls_indices:
        img_rgb = np.array(Image.open(sampled_paths[idx]).convert("RGB"))
        h_ch, dab_ch = deconvolve_numpy(img_rgb)
        h_means.append(h_ch.mean())
        dab_means.append(dab_ch.mean())
        h_maxes.append(h_ch.max())
        dab_maxes.append(dab_ch.max())

    print(f"\n  {cls}")
    print(f"  {'─' * 40}")
    print(f"  H   mean: {np.mean(h_means):.4f}  |  max: {np.mean(h_maxes):.4f}")
    print(f"  DAB mean: {np.mean(dab_means):.4f}  |  max: {np.mean(dab_maxes):.4f}")
    print(f"  DAB/H ratio: {np.mean(dab_means) / (np.mean(h_means) + 1e-8):.4f}")

print(f"\n{'=' * 60}")
print("INTERPRETATION GUIDE:")
print("=" * 60)
print("""
  class_0  (HER2 negative):
    → DAB should be VERY LOW (little to no brown staining)
    → H should be moderate (nuclei always present)
    → DAB/H ratio should be LOW (< 0.5)

  class_1+ (weak positive):
    → DAB slightly higher than class_0
    → Faint, incomplete membrane staining

  class_2+ (equivocal):
    → DAB moderate
    → Weak to moderate complete membrane staining

  class_3+ (strong positive):
    → DAB should be HIGH (intense brown membrane)
    → DAB/H ratio should be HIGH (> 1.0)
    → Strong, complete membrane staining

  If DAB does NOT increase from class_0 → class_3+,
  the deconvolution is NOT separating stains correctly.
  ❌ DO NOT proceed with training.
""")


# ============================================================
# Cell 6 — Zoom-in: 4 Patches Side-by-Side (One Per Class)
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(14, 16))

for row, cls in enumerate(CLASSES):

    cls_indices = [i for i, l in enumerate(sampled_labels) if l == cls]
    if not cls_indices:
        for col in range(3):
            axes[row, col].axis("off")
        continue

    idx = cls_indices[0]  # Take first sample of each class
    img_rgb = np.array(Image.open(sampled_paths[idx]).convert("RGB"))
    h_ch, dab_ch = deconvolve_numpy(img_rgb)

    # RGB
    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_ylabel(cls, fontsize=14, fontweight="bold", rotation=0, labelpad=60)
    axes[row, 0].set_title("RGB" if row == 0 else "", fontsize=13)
    axes[row, 0].axis("off")

    # H
    axes[row, 1].imshow(normalize_for_display(h_ch), cmap="Blues_r")
    axes[row, 1].set_title("Hematoxylin" if row == 0 else "", fontsize=13)
    axes[row, 1].axis("off")

    # DAB
    im = axes[row, 2].imshow(normalize_for_display(dab_ch), cmap="YlOrBr")
    axes[row, 2].set_title("DAB (HER2)" if row == 0 else "", fontsize=13)
    axes[row, 2].axis("off")

plt.suptitle(
    "Color Deconvolution — One Patch Per HER2 Score\n"
    "DAB intensity should increase from 0 → 3+",
    fontsize=16, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.savefig("/content/deconv_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Saved: /content/deconv_per_class.png")
print("\n🔍 Look at the DAB column from top to bottom.")
print("   The brown signal should clearly INCREASE from class_0 to class_3+.")
print("   If it doesn't → the deconvolution vectors need recalibration.")
