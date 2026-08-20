# ============================================================
# DSS-ViT — Shared Split Utility
# ============================================================
#
# Creates/loads the deterministic stratified 10% validation
# holdout from the training set (seed 42, stratify=labels).
# Shared by scripts/precompute_stain_stats.py and
# utils/train_dss_vit.py so both use the SAME split.
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from datasets import HER2Dataset


def get_or_create_split_indices(
    train_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    save_path: str = "split_indices_dss_vit.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates (or loads if it exists) the deterministic stratified
    validation holdout indices.

    Args:
        train_dir (str): Path to the official training directory.
        val_fraction (float): Fraction of the training set to hold out.
        seed (int): Random seed for reproducibility.
        save_path (str): Path to save/load the split indices (.npz).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            (train_indices, val_indices) — integer indices into the
            full training dataset.
    """
    save_path = Path(save_path)

    if save_path.exists():
        print(f"Loading existing split indices from '{save_path}'")
        data = np.load(save_path)
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]
        print(
            f"  Train: {len(train_indices)} | Val: {len(val_indices)}"
        )
        return train_indices, val_indices

    # Build the full training dataset (no transform needed for indices)
    dataset = HER2Dataset(root_dir=train_dir, transform=None)
    labels = np.array(dataset.labels)

    # Stratified split
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    # Save
    os.makedirs(save_path.parent, exist_ok=True)
    np.savez(
        save_path,
        train_indices=train_indices,
        val_indices=val_indices,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(f"Created and saved split indices to '{save_path}'")
    print(
        f"  Train: {len(train_indices)} | Val: {len(val_indices)}"
    )

    return train_indices, val_indices