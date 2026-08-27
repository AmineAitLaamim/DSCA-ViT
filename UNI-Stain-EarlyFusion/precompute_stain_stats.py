# ============================================================
# UNI-Stain-EarlyFusion — Precompute Global Stain Statistics
# ============================================================
#
# Computes the global mean/std of the H and DAB channels over the
# TRAINING split only (not validation). Uses the WSI-aware split
# from the proper ViT-B16 baseline (split_indices_wsi.npz).
#
# Uses the SAME ColorDeconvolution implementation inside
# UNI-Stain-EarlyFusion/ (UNI-Stain-EarlyFusion/color_deconv.py)
# so stain normalization is self-consistent.
#
# Runs on a single GPU or CPU only (NOT under DDP).
#
# Usage:
#   uv run python UNI-Stain-EarlyFusion/precompute_stain_stats.py \
#       --config configs/uni_stain_earlyfusion_config.yaml
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Add project root to path so imports work.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from baseline import HER2BaselineDataset
except ImportError:
    # Self-contained fallback dataset
    from torch.utils.data import Dataset as _Dataset
    from PIL import Image as _Image

    CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    class HER2BaselineDataset(_Dataset):
        def __init__(self, root_dir: str, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = CLASSES
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.image_paths = []
            self.labels = []
            if not self.root_dir.exists():
                raise ValueError(f"Dataset root not found: {self.root_dir}")
            for cls in self.classes:
                cls_dir = self.root_dir / cls
                if not cls_dir.exists():
                    raise ValueError(f"Missing class dir: {cls_dir}")
                label = self.class_to_idx[cls]
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        self.image_paths.append(p)
                        self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = _Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["paths"]["train_dir"]
    image_size = config["model"]["image_size"]
    split_indices_path = config["paths"]["split_indices_path"]
    stain_stats_path = config["paths"]["stain_stats_path"]

    # Ensure output directory exists
    Path(stain_stats_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------
    # LOAD the baseline's WSI-aware split (never regenerate)
    # ------------------------------------------------------------
    if not os.path.exists(split_indices_path):
        raise FileNotFoundError(
            f"Baseline split not found at '{split_indices_path}'. "
            "The proper ViT-B16 baseline creates this file."
        )
    print(f"Loading baseline split from '{split_indices_path}'")
    data = np.load(split_indices_path)
    train_indices = data["train_indices"]
    print(f"  Train: {len(train_indices)}")

    # Build the full training dataset (Resize + ToTensor, no aug)
    dataset = HER2BaselineDataset(
        root_dir=train_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]),
    )

    # Subset to the TRAINING split set only
    train_subset = Subset(dataset, train_indices)
    print(f"Computing stain stats on {len(train_subset)} training images...")

    # ------------------------------------------------------------
    # Accumulate mean/std for H and DAB
    # ------------------------------------------------------------
    deconv = ColorDeconvolution().to(device)
    deconv.eval()

    h_sum = 0.0
    h_sq_sum = 0.0
    dab_sum = 0.0
    dab_sq_sum = 0.0
    total_pixels = 0

    loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            h_channel, dab_channel = deconv(images)  # (B, 1, H, W)

            h_flat = h_channel.flatten()
            dab_flat = dab_channel.flatten()

            h_sum += h_flat.sum().item()
            h_sq_sum += (h_flat ** 2).sum().item()
            dab_sum += dab_flat.sum().item()
            dab_sq_sum += (dab_flat ** 2).sum().item()
            total_pixels += h_flat.numel()

    # ------------------------------------------------------------
    # Compute mean / std
    # ------------------------------------------------------------
    h_mean = h_sum / total_pixels
    h_var = max(h_sq_sum / total_pixels - h_mean ** 2, 0.0)
    h_std = max(np.sqrt(h_var), 1e-6)

    dab_mean = dab_sum / total_pixels
    dab_var = max(dab_sq_sum / total_pixels - dab_mean ** 2, 0.0)
    dab_std = max(np.sqrt(dab_var), 1e-6)

    print(f"H mean: {h_mean:.6f} | H std: {h_std:.6f}")
    print(f"DAB mean: {dab_mean:.6f} | DAB std: {dab_std:.6f}")

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    save_stain_stats(
        h_mean=h_mean,
        h_std=h_std,
        dab_mean=dab_mean,
        dab_std=dab_std,
        path=stain_stats_path,
    )


if __name__ == "__main__":
    main()
from color_deconv import ColorDeconvolution
from stain_stats import save_stain_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute global H/DAB stain statistics (UNI-Stain-EarlyFusion)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/uni_stain_earlyfusion_config.yaml",
        help="Path to the UNI-Stain-EarlyFusion config file.",
    )
    return parser.parse_args()