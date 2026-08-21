# ============================================================
# Plain ViT-B16 Baseline — Dataset / Transforms / Split
# ============================================================
#
# Self-contained utilities for the baseline ViT training:
#   - dataset loader (class folders)
#   - ImageNet-normalized transforms (as in the Colab notebook)
#   - deterministic stratified train/val split
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


# ------------------------------------------------------------
# Constants (must match the original Colab notebook)
# ------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training transforms — identical to the Colab baseline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=10,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """Test/validation transforms — identical to the Colab baseline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class HER2BaselineDataset(Dataset):
    """HER2-IHC-40x dataset with the classic class-folder layout."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = EXPECTED_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        if not self.root_dir.exists():
            raise ValueError(f"Dataset root directory does not exist: {self.root_dir}")

        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            if not class_dir.exists():
                raise ValueError(f"Missing class directory: {class_dir}")
            label = self.class_to_idx[cls_name]
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(p)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_classes(self) -> int:
        return len(self.classes)

    def get_class_names(self) -> List[str]:
        return self.classes

    def get_class_distribution(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for label in self.labels:
            counts[self.classes[label]] += 1
        return counts


# ------------------------------------------------------------
# Split
# ------------------------------------------------------------
def get_or_create_split_indices(
    train_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    save_path: str = "split_indices.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """Create or load the deterministic stratified val holdout."""
    save_path = Path(save_path)

    if save_path.exists():
        print(f"Loading existing split indices from '{save_path}'")
        data = np.load(save_path)
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]
        print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")
        return train_indices, val_indices

    dataset = HER2BaselineDataset(root_dir=train_dir, transform=None)
    labels = np.array(dataset.labels)

    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    os.makedirs(save_path.parent, exist_ok=True)
    np.savez(
        save_path,
        train_indices=train_indices,
        val_indices=val_indices,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(f"Created and saved split indices to '{save_path}'")
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")

    return train_indices, val_indices