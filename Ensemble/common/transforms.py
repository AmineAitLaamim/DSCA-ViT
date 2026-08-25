# ============================================================
# Ensemble — per-model transforms + test-time augmentation (TTA)
# ============================================================
#
# Per-model preprocessing is NOT uniform because UNI normalizes
# internally while ViT-B16 CE/CORN expect ImageNet-normalized input.
#
#   vit_ce  : Resize -> [Aug] -> ToTensor -> Normalize(ImageNet)
#   vit_corn: Resize -> [Aug] -> ToTensor -> Normalize(ImageNet)
#   uni_ce  : Resize -> [Aug] -> ToTensor         (raw RGB [0,1])
#
# TTA (6 augmentations per sample, applied BEFORE normalization):
#   0 Original, 1 H-flip, 2 V-flip, 3 Rot90, 4 Rot180, 5 Rot270.
# Probabilities (never logits) are averaged across augmentations.
# ============================================================

from __future__ import annotations

from typing import List

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model names this package supports.
MODEL_NAMES = ("vit_ce", "uni_ce", "vit_corn")

# TTA list: (label, callable transform). Applied to a PIL image right
# after Resize and before ToTensor / Normalize.
TTA_AUGMENTATIONS: List[tuple] = [
    ("original",        lambda img: img),
    ("horizontal_flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("vertical_flip",   transforms.RandomVerticalFlip(p=1.0)),
    ("rot90",           transforms.RandomRotation(degrees=(90, 90))),
    ("rot180",          transforms.RandomRotation(degrees=(180, 180))),
    ("rot270",          transforms.RandomRotation(degrees=(270, 270))),
]


def get_model_transform(
    model_name: str, tta: bool = False, image_size: int = 224
) -> transforms.Compose:
    """Return the (single) test/validation transform for a model.

    If tta=True, only the ORIGINAL (no-aug) pipeline is returned; use
    get_tta_transform() with each aug index for TTA-augmented variants.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of {MODEL_NAMES}."
        )
    normalize = (
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if model_name in ("vit_ce", "vit_corn")
        else transforms.Identity()
    )
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if not isinstance(normalize, transforms.Identity):
        ops.append(normalize)
    return transforms.Compose(ops)


def get_tta_transform(
    model_name: str, aug_index: int, image_size: int = 224
) -> transforms.Compose:
    """Return the TTA-augmented single transform for aug_index (0..5)."""
    if aug_index < 0 or aug_index >= len(TTA_AUGMENTATIONS):
        raise ValueError(
            f"aug_index must be in [0, {len(TTA_AUGMENTATIONS) - 1}], got {aug_index}."
        )
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of {MODEL_NAMES}."
        )
    label, aug = TTA_AUGMENTATIONS[aug_index]
    normalize = (
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if model_name in ("vit_ce", "vit_corn")
        else None
    )
    ops = [transforms.Resize((image_size, image_size))]
    if aug_index > 0:
        ops.append(aug)
    ops.append(transforms.ToTensor())
    if normalize is not None:
        ops.append(normalize)
    return transforms.Compose(ops)


def get_tta_indices() -> List[int]:
    """Return the 6 TTA augmentation indices [0..5]."""
    return list(range(len(TTA_AUGMENTATIONS)))


def describe_transform(model_name: str, tta: bool = False) -> str:
    """Human-readable summary of the transform pipeline (for --debug)."""
    parts = ["Resize(224,224)"]
    if tta:
        parts.append("+TTA(orig,hflip,vflip,rot90,rot180,rot270)")
    parts.append("ToTensor()")
    if model_name in ("vit_ce", "vit_corn"):
        parts.append("Normalize(ImageNet)")
    else:
        parts.append("No-Normalize(raw RGB [0,1])")
    return f"{model_name}: {' -> '.join(parts)}"
