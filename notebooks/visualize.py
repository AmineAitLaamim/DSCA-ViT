# DSCA-ViT — Visualization Notebook
# Visualize color deconvolution outputs, gate values, and
# cross-attention maps for interpretability analysis.

# ============================================================
# Cell 1 — Clone / Pull Repository
# ============================================================

import subprocess
import os

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
# Cell 2 — Imports & Setup
# ============================================================

import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path

from models.color_deconv import deconvolve_numpy
from models import DSCAViT
from datasets import get_test_transform, HER2Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helpers
# ============================================================

def load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def normalize_map(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] for display."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# ============================================================
# Visualization 1 — Color Deconvolution
# ============================================================

def visualize_deconvolution(image_path: str) -> None:
    """
    Show original RGB, Hematoxylin channel, and DAB channel
    side by side for a single image.
    """
    img_rgb = load_image(image_path)
    h_ch, dab_ch = deconvolve_numpy(img_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(normalize_map(h_ch), cmap="Blues_r")
    axes[1].set_title("Hematoxylin (Morphology)", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(normalize_map(dab_ch), cmap="YlOrBr")
    axes[2].set_title("DAB (HER2 Expression)", fontsize=14)
    axes[2].axis("off")

    plt.suptitle("Color Deconvolution — Ruifrok & Johnston", fontsize=16)
    plt.tight_layout()
    plt.savefig("/content/deconvolution_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: /content/deconvolution_visualization.png")


# ============================================================
# Visualization 1b — Batch Color Deconvolution Sanity Check
# ============================================================

def visualize_deconvolution_batch(
    root_dir: str,
    num_samples: int = 20,
    seed: int = 42,
) -> None:
    """
    CRITICAL SANITY CHECK: Visualize color deconvolution on random patches.

    This is the highest-priority validation for the DSCA-ViT architecture.
    If the deconvolution is poor, the H stream isn't learning morphology
    and the DAB stream isn't learning HER2 staining — the dual-stream
    idea becomes meaningless.

    For each patch, shows: Original RGB | Hematoxylin | DAB

    What to look for:
      - H channel (blue colormap): should show nuclei / morphology clearly
      - DAB channel (brown colormap): should show brown HER2 membrane signal
      - If DAB looks gray/noisy or H looks brown, the stain vectors are
        miscalibrated for this dataset.

    Args:
        root_dir: Root directory containing class_0/, class_1+/,
                  class_2+/, class_3+/ subdirectories.
        num_samples: Number of random patches to visualize.
        seed: Random seed for reproducibility.
    """
    # Collect all image paths
    dataset = HER2Dataset(root_dir=root_dir, transform=None)

    # Sample random indices (reproducible)
    random.seed(seed)
    n_available = len(dataset)
    n_samples = min(num_samples, n_available)
    indices = random.sample(range(n_available), n_samples)

    # Layout: 5 images per row, each image has 3 panels (RGB, H, DAB)
    n_cols = 5
    n_rows = math.ceil(n_samples / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols * 3,
        figsize=(n_cols * 3, n_rows * 3),
    )

    for idx, sample_idx in enumerate(indices):
        img_path = dataset.image_paths[sample_idx]
        class_name = dataset.classes[dataset.labels[sample_idx]]

        img_rgb = load_image(str(img_path))
        h_ch, dab_ch = deconvolve_numpy(img_rgb)

        row = idx // n_cols
        col = (idx % n_cols) * 3

        # Original RGB
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f"RGB\n{class_name}", fontsize=9)
        axes[row, col].axis("off")

        # Hematoxylin
        axes[row, col + 1].imshow(normalize_map(h_ch), cmap="Blues_r")
        axes[row, col + 1].set_title("H", fontsize=9)
        axes[row, col + 1].axis("off")

        # DAB
        axes[row, col + 2].imshow(normalize_map(dab_ch), cmap="YlOrBr")
        axes[row, col + 2].set_title("DAB", fontsize=9)
        axes[row, col + 2].axis("off")

    # Hide any unused subplots (if fewer than n_rows * n_cols images)
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 3
        for c in range(3):
            axes[row, col + c].axis("off")

    plt.suptitle(
        f"Color Deconvolution Sanity Check — {n_samples} Random Patches\n"
        f"H should show nuclei/morphology | DAB should show brown HER2 signal",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig("/content/deconvolution_batch_check.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: /content/deconvolution_batch_check.png")


# ============================================================
# Visualization 2 — Gate Value Heatmap
# ============================================================

def visualize_gate_values(model: DSCAViT, image_path: str) -> None:
    """
    Visualize per-patch gate values from the gated fusion.

    High gate values (near 1.0) mean the H (morphology) stream
    dominates at that patch. Low gate values (near 0.0) mean
    the DAB (HER2 signal) stream dominates.
    """
    model.eval()
    transform = get_test_transform(image_size=224)

    img_pil  = Image.open(image_path).convert("RGB")
    img_rgb  = np.array(img_pil)
    x_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x_tensor)

    gates = model.get_gate_values()  # (1, 196, 768)
    gate_map = gates[0].mean(dim=-1)  # Average over embed dim -> (196,)
    gate_map = gate_map.cpu().numpy().reshape(14, 14)  # 14x14 patch grid

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    im = axes[1].imshow(gate_map, cmap="RdBu_r", vmin=0.0, vmax=1.0)
    axes[1].set_title("Gate Values\n(Blue=DAB dominant, Red=H dominant)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay on original
    gate_upsampled = np.array(
        Image.fromarray((gate_map * 255).astype(np.uint8)).resize(
            (224, 224), Image.NEAREST
        )
    )
    axes[2].imshow(img_rgb)
    axes[2].imshow(gate_upsampled, cmap="RdBu_r", alpha=0.5)
    axes[2].set_title("Gate Overlay", fontsize=13)
    axes[2].axis("off")

    plt.suptitle("Gated Fusion — Which Stain Dominates at Each Patch?", fontsize=15)
    plt.tight_layout()
    plt.savefig("/content/gate_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: /content/gate_visualization.png")


# ============================================================
# Visualization 3 — Cross-Attention Maps
# ============================================================

def visualize_cross_attention(model: DSCAViT, image_path: str) -> None:
    """
    Visualize the cross-attention weights for a selected patch.

    Shows: for patch at center of image, which DAB patches does
    the H stream attend to (and vice versa)?
    """
    model.eval()
    transform = get_test_transform(image_size=224)

    img_pil  = Image.open(image_path).convert("RGB")
    img_rgb  = np.array(img_pil)
    x_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x_tensor)

    # Get attention weights from cross-attention module
    bca = model.cross_attention
    attn_hd = bca.cross_attn_h.attn_weights  # (1, 12, 197, 197)
    attn_dh = bca.cross_attn_d.attn_weights  # (1, 12, 197, 197)

    if attn_hd is None:
        print("Attention weights not stored. Ensure model stores them.")
        return

    # Average over heads, exclude CLS
    attn_hd_avg = attn_hd[0].mean(dim=0)  # (197, 197)
    attn_dh_avg = attn_dh[0].mean(dim=0)

    # Select center patch (index 105 = row 7, col 7 on 14x14 grid)
    # 7 * 14 + 7 = 105, +1 because token 0 is CLS
    center_idx = 105 + 1  # = 106

    attn_from_h = attn_hd_avg[center_idx, 1:].cpu().numpy().reshape(14, 14)
    attn_from_d = attn_dh_avg[center_idx, 1:].cpu().numpy().reshape(14, 14)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    # Mark center patch
    rect_row, rect_col = 7 * 16, 7 * 16
    rect = plt.Rectangle(
        (rect_col, rect_row), 16, 16,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Selected Patch (red box)", fontsize=13)
    axes[0].axis("off")

    im1 = axes[1].imshow(normalize_map(attn_from_h), cmap="hot")
    axes[1].set_title("H→DAB Attention\n(Where does H patch look in DAB?)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(normalize_map(attn_from_d), cmap="hot")
    axes[2].set_title("DAB→H Attention\n(Where does DAB patch look in H?)", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Spatially-Biased Cross-Attention Maps", fontsize=15)
    plt.tight_layout()
    plt.savefig("/content/attention_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: /content/attention_visualization.png")


# ============================================================
# Usage example
# ============================================================

# Replace with a real image path from your dataset
SAMPLE_IMAGE = "/content/HER2_Dataset/WSI-based-dataset/test/class_3+/some_image.png"

# Visualization 1: Color deconvolution (no model needed)
visualize_deconvolution(SAMPLE_IMAGE)

# Visualization 1b: CRITICAL — Batch deconvolution sanity check (20 random patches)
# Run this FIRST before training. Inspect the output figure:
#   - H channel should show nuclei/morphology (blue)
#   - DAB channel should show brown HER2 membrane signal
# If the separation looks wrong, the fixed Ruifrok stain vectors are
# miscalibrated for this dataset and the dual-stream premise is invalid.
TRAIN_ROOT = "/content/HER2_Dataset/WSI-based-dataset/train"
visualize_deconvolution_batch(root_dir=TRAIN_ROOT, num_samples=20, seed=42)

# Load trained model for visualizations 2 and 3
# model = DSCAViT(num_classes=4, pretrained=False).to(device)
# load_checkpoint(path="...", model=model, device=device)
# visualize_gate_values(model, SAMPLE_IMAGE)
# visualize_cross_attention(model, SAMPLE_IMAGE)
