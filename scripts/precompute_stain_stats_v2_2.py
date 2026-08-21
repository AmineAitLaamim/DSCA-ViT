# ============================================================
# DSS-ViT v2.2 — Precompute Global Stain Statistics
# ============================================================
#
# Computes the global mean/std of the H and DAB channels over the
# TRAINING split only (not validation). Uses the WSI-aware split.
#
# Runs on a single GPU or CPU only (NOT under DDP).
#
# Usage:
#   python scripts/precompute_stain_stats_v2_2.py --config configs/dss_vit_v2_2_config.yaml
# ============================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models_v2_2 import ColorDeconvolution, save_stain_stats
from utils.split_utils_wsi import get_or_create_wsi_split_indices
from baseline.baseline_data import HER2BaselineDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute global H/DAB stain statistics (v2.2)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dss_vit_v2_2_config.yaml",
        help="Path to the DSS-ViT v2.2 config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["paths"]["train_dir"]
    test_dir = config["paths"]["test_dir"]
    image_size = config["dataset"]["image_size"]
    split_indices_path = config["paths"]["split_indices_path"]
    stain_stats_path = config["paths"]["stain_stats_path"]

    # Ensure experiment directory exists before writing generated files
    os.makedirs(str(Path(stain_stats_path).parent), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Load the WSI-aware split and use ONLY the training portion
    # ------------------------------------------------------------
    val_fraction = config["dataset"].get("val_fraction", 0.10)
    val_seed = config["dataset"].get("val_seed", 42)

    train_indices, _ = get_or_create_wsi_split_indices(
        train_dir=train_dir,
        test_dir=test_dir,
        val_fraction=val_fraction,
        seed=val_seed,
        save_path=split_indices_path,
    )

    # Build the full training dataset with test transform (Resize + ToTensor, no aug)
    from torchvision import transforms

    dataset = HER2BaselineDataset(
        root_dir=train_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]),
    )

    # Subset to the TRAINING split only
    train_subset = torch.utils.data.Subset(dataset, train_indices)
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

    loader = torch.utils.data.DataLoader(
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
    # Compute mean/std
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