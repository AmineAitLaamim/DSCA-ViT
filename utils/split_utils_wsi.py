# ============================================================
# WSI-Aware Split Utility — Group patches by slide, then split
# ============================================================
#
# Creates/loads a proper WSI-aware train/val split.
#
# The HER2-IHC-40x dataset is composed of patches (tiles) extracted
# from Whole Slide Images (WSIs). If we split patches randomly, tiles
# from the same WSI appear in BOTH train and val — this is data
# leakage.
#
# This utility:
#   1. Parses a WSI identifier from each image filename.
#   2. Groups all patches by their WSI identifier.
#   3. Splits the UNIQUE WSI identifiers into train/val groups,
#      stratified by the dominant class of each WSI.
#   4. Saves train_indices, val_indices, test_indices,
#      val_fraction=0.10 to split_indices_wsi.npz.
#
# Usage:
#   python utils/split_utils_wsi.py \
#       --train-dir /path/to/train \
#       --test-dir /path/to/test \
#       --output /path/to/split_indices_wsi.npz \
#       --val-fraction 0.10 --seed 42
# ============================================================

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# WSI identifier parsing
# ------------------------------------------------------------
def parse_wsi_id(filename: str) -> Optional[str]:
    """
    Extracts a WSI (slide) identifier from an image filename.

    Handles common HER2-IHC-40x naming conventions:

      her2-0-<slide>-<tile>.png          -> slide
      her2_0_<slide>_<tile>.png          -> slide
      her2-0-score_test_<tile>.png       -> test_<tile> (fallback)
      her2-0-score_train_<tile>.png      -> train_<tile> (fallback)

    Args:
        filename (str): Image filename (e.g. "her2-0-abc12-34.png").

    Returns:
        Optional[str]: The WSI identifier, or None if it cannot be parsed.
    """
    stem = Path(filename).stem

    # Strip trailing augmentation suffix (e.g. _gradcam)
    stem = re.sub(r"(_gradcam.*)$", "", stem)

    # Pattern 1: her2-<class>-<slide>-<tile>
    #            her2_<class>_<slide>_<tile>
    match = re.match(
        r"^(?:her2|HER2)[-_](\d+|1\+|2\+|3\+)[-_](.+?)[-_](\d+)$",
        stem,
    )
    if match:
        return f"slide_{match.group(2)}"

    # Pattern 2: her2-<class>-score_<subset>_<tile>
    match = re.match(
        r"^(?:her2|HER2)[-_](\d+|1\+|2\+|3\+)[-_]score[-_](.+?)[-_](\d+)$",
        stem,
    )
    if match:
        return f"slide_{match.group(2)}"

    # Pattern 3: her2-<class>-<slide>  (no tile suffix)
    match = re.match(
        r"^(?:her2|HER2)[-_](\d+|1\+|2\+|3\+)[-_](.+)$",
        stem,
    )
    if match:
        return f"slide_{match.group(2)}"

    # Fallback: use the whole stem (each file is its own "slide")
    return stem


# ------------------------------------------------------------
# Dataset listing
# ------------------------------------------------------------
def collect_dataset(root_dir: str) -> Tuple[List[str], List[int]]:
    """
    Collects all image paths and their class labels (0-3).

    Args:
        root_dir (str): Path to a train/ or test/ directory.

    Returns:
        Tuple[List[str], List[int]]: (filepaths, labels).
    """
    root = Path(root_dir)
    classes = ["class_0", "class_1+", "class_2+", "class_3+"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    exts = {".png", ".jpg", ".jpeg"}

    filepaths: List[str] = []
    labels: List[int] = []

    for cls_name in classes:
        cls_dir = root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")

        label = class_to_idx[cls_name]
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                filepaths.append(str(p))
                labels.append(label)

    return filepaths, labels


# ------------------------------------------------------------
# Main WSI-aware split
# ------------------------------------------------------------
def get_or_create_wsi_split_indices(
    train_dir: str,
    test_dir: Optional[str] = None,
    val_fraction: float = 0.10,
    seed: int = 42,
    save_path: str = "split_indices_wsi.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates (or loads if it exists) the WSI-aware stratified split.

    The split is performed at the WSI (slide) level — all patches
    from the same WSI go into the same split (train or val). This
    prevents the leakage caused by random patch-level splits.

    Args:
        train_dir (str): Path to the official training directory.
        test_dir (Optional[str]): Path to the official test directory.
            If provided, test_indices are also saved.
        val_fraction (float): Fraction of WSIs to hold out for validation.
        seed (int): Random seed for reproducibility.
        save_path (str): Path to save/load the split (.npz).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            (train_indices, val_indices) — integer indices into the
            full training dataset.
    """
    save_path = Path(save_path)

    if save_path.exists():
        print(f"Loading existing WSI split indices from '{save_path}'")
        data = np.load(save_path)
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]
        print(
            f"  Train: {len(train_indices)} | Val: {len(val_indices)}"
        )
        return train_indices, val_indices

    # ------------------------------------------------------------
    # Collect training files + labels
    # ------------------------------------------------------------
    train_files, train_labels = collect_dataset(train_dir)
    n_train = len(train_files)
    print(f"Collected {n_train} training images.")

    # ------------------------------------------------------------
    # Parse WSI IDs and group by slide
    # ------------------------------------------------------------
    wsi_to_indices: Dict[str, List[int]] = defaultdict(list)
    wsi_to_labels: Dict[str, List[int]] = defaultdict(list)

    for idx, (fp, label) in enumerate(zip(train_files, train_labels)):
        wsi = parse_wsi_id(Path(fp).name)
        if wsi is None:
            raise ValueError(f"Cannot parse WSI ID from filename: {fp}")
        wsi_to_indices[wsi].append(idx)
        wsi_to_labels[wsi].append(label)

    unique_wsis = sorted(wsi_to_indices.keys())
    print(f"Found {len(unique_wsis)} unique WSIs.")

    # Dominant class per WSI (for stratified split)
    wsi_dominant_label = np.array([
        np.bincount(wsi_to_labels[w]).argmax() for w in unique_wsis
    ], dtype=np.int64)

    # ------------------------------------------------------------
    # Stratified split at WSI level
    # ------------------------------------------------------------
    if len(unique_wsis) < 2:
        raise ValueError(
            "Need at least 2 WSIs for a train/val split. "
            f"Found {len(unique_wsis)}."
        )

    wsi_train, wsi_val = train_test_split(
        np.arange(len(unique_wsis)),
        test_size=val_fraction,
        random_state=seed,
        stratify=wsi_dominant_label,
    )

    # Expand WSI-level split back to patch-level indices
    train_indices = sorted(
        idx
        for wsi_idx in wsi_train
        for idx in wsi_to_indices[unique_wsis[wsi_idx]]
    )
    val_indices = sorted(
        idx
        for wsi_idx in wsi_val
        for idx in wsi_to_indices[unique_wsis[wsi_idx]]
    )

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)

    # ------------------------------------------------------------
    # Test indices (into the official test set)
    # ------------------------------------------------------------
    test_indices = None
    if test_dir is not None:
        test_files, _ = collect_dataset(test_dir)
        test_indices = np.arange(len(test_files), dtype=np.int64)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    os.makedirs(save_path.parent, exist_ok=True)
    np.savez(
        save_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        val_fraction=val_fraction,
        seed=seed,
        n_wsim_train=len(wsi_train),
        n_wsim_val=len(wsi_val),
    )
    print(f"Created and saved WSI split indices to '{save_path}'")
    print(
        f"  Train: {len(train_indices)} | Val: {len(val_indices)}"
        + (f" | Test: {len(test_indices)}" if test_indices is not None else "")
    )
    print(f"  WSIs: {len(wsi_train)} train | {len(wsi_val)} val")

    return train_indices, val_indices


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a WSI-aware train/val split."
    )
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to the training directory.")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to the test directory (optional).")
    parser.add_argument("--output", type=str,
                        default="split_indices_wsi.npz",
                        help="Output .npz path.")
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Fraction of WSIs to hold out for validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_or_create_wsi_split_indices(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()