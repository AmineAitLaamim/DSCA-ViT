# ============================================================
# Ensemble — inference + probability ensembling
# ============================================================
#
# For each model we collect probs [N,4] using its own transform.
# With TTA enabled we average probs (never logits) over the 6
# augmentations per sample before combining across models.
#
# Ensemble combination is simple equal-weight probability
# averaging:
#   ensemble_probs = (probs_m1 + probs_m2 + ...) / num_models
#   ensemble_preds = argmax(ensemble_probs, dim=1)
#
# We USE ONLY probs for ensembling — never raw logits. For CORN
# this means its chain-rule-reconstructed [B,4] probs, not the
# raw [B,3] conditional logits.
# ============================================================

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    get_model_transform,
    get_tta_indices,
    get_tta_transform,
)


def collect_probs(
    model,
    device,
    dataset: Dataset,
    model_name: str,
    tta: bool = False,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple:
    """Run a single model over `dataset`, returning (probs, labels).

    Returns:
        probs  : np.ndarray [N, 4]
        labels : np.ndarray [N]  (int)
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    if tta:
        aug_indices = get_tta_indices()  # 6 augmentations
        # Build a TTA-augmented dataset yielding each sample 6 times
        # (one per augmentation), in order base_idx*6 + aug_idx.
        tta_loader = _build_tta_loader(
            dataset, model_name, aug_indices, batch_size, num_workers, pin_memory
        )
        n_aug = len(aug_indices)
        n_samples = len(dataset)
        sums = np.zeros((n_samples, 4), dtype=np.float64)
        counts = np.zeros(n_samples, dtype=np.int64)
        labels_arr = np.zeros(n_samples, dtype=np.int64)
        global_offset = 0
        with torch.no_grad():
            for images, labels in tta_loader:
                images = images.to(device, non_blocking=True)
                probs = model(images)["probs"].cpu().numpy()  # [B, 4]
                B = images.shape[0]
                for b in range(B):
                    abs_idx = global_offset + b
                    base_idx = abs_idx // n_aug
                    sums[base_idx] += probs[b]
                    counts[base_idx] += 1
                    labels_arr[base_idx] = labels[b]
                global_offset += B
        if not np.all(counts == n_aug):
            raise RuntimeError(
                f"TTA averaging incomplete: some samples got {counts.min()} "
                f"augmentations (expected {n_aug})."
            )
        avg_probs = sums / n_aug  # [N, 4]
        return avg_probs, labels_arr

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            pred = model(images)
            probs = pred["probs"].cpu().numpy()  # [B, 4]
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels)


def _build_tta_loader(dataset, model_name, aug_indices, batch_size, num_workers, pin_memory):
    """Wrap dataset so each sample is returned once per TTA augmentation."""
    class _AugmentedWrapper(Dataset):
        def __init__(self, base, model_name, aug_indices):
            self.base = base
            self._transforms = [
                get_tta_transform(model_name, i) for i in aug_indices
            ]
            self.n_aug = len(aug_indices)
            # Preserve __getitem__ semantics (returns (img, label)).
            self.base_len = len(base)

        def __len__(self):
            return self.base_len * self.n_aug

        def __getitem__(self, idx):
            base_idx = idx // self.n_aug
            aug_idx = idx % self.n_aug
            # Dataset returns (PIL image, label); re-apply transform.
            image, label = self.base[base_idx]
            return self._transforms[aug_idx](image), label

    wrapped = _AugmentedWrapper(dataset, model_name, aug_indices)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def average_ensemble_probs(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight average of per-model probs arrays -> [N, 4]."""
    if not probs_list:
        raise ValueError("No probability arrays provided to ensemble.")
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in probs_list], axis=0)
    return stacked.mean(axis=0)


def ensemble_predictions(probs: np.ndarray) -> np.ndarray:
    """argmax over averaged probs -> integer class predictions [N]."""
    return np.argmax(probs, axis=1)
