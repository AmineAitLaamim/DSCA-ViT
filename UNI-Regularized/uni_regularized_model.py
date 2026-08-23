# ============================================================
# UNI-Regularized — Main Model
# ============================================================
#
# Same architecture as the UNI-baseline model
# (baseline/UNI-baseline/uni_baseline_model.py), renamed to
# UNIRegularizedModel. No components added or removed.
#
# UNI histopathology foundation model (ViT-L/16, DINOv2,
# mass100k pretraining) with a fresh 4-class classification
# head for HER2-IHC-40x grading.
#
# Loading recipe is EXACTLY the official UNI recipe:
#   - timm "vit_large_patch16_224" with img_size=224,
#     patch_size=16, init_values=1e-5, num_classes=0,
#     dynamic_img_size=True
#   - strict load_state_dict from the local checkpoint path
#
# The dataloader returns RAW RGB in [0,1] — ImageNet
# normalization is applied INSIDE forward().
#
# No network calls, no Hugging Face downloads.
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Dict

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UNIRegularizedModel(nn.Module):
    """UNI (ViT-L/16) backbone + 4-class linear head.

    Args:
        checkpoint_path (str): Absolute path to the local UNI
            checkpoint (pytorch_model.bin).
        num_classes (int): Number of classification classes (default 4).
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4, verbose: bool = False):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"UNI checkpoint not found at '{checkpoint_path}'. "
                "This model requires the local UNI weights."
            )

        # --------------------------------------------------------
        # Official UNI backbone recipe (unchanged kwargs)
        # --------------------------------------------------------
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(state_dict, strict=True)
        if verbose:
            print(f"  [UNI-Regularized] Loaded backbone strictly from '{checkpoint_path}'")

        self.backbone_feature_dim = self.backbone.num_features  # 1024 for ViT-L/16
        self.head = nn.Linear(self.backbone_feature_dim, num_classes)

    # ------------------------------------------------------------
    # Freeze / unfreeze helpers (used by training + --debug)
    # ------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (requires_grad=False)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (requires_grad=True)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: raw RGB in [0, 1], shape [B, 3, 224, 224]
        # Normalize INSIDE the model (dataloader returns raw RGB).
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        features = self.backbone(x)
        if features.dim() == 3:
            # [B, 197, 1024] -> CLS token [B, 1024]
            features = features[:, 0]

        logits = self.head(features)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

    # ------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + head_params

        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": total,
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }