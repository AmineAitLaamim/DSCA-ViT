# ============================================================
# UNI-CORN — Main Model (UNI Backbone + CORN Ordinal Head)
# ============================================================
#
# Combines the UNI histopathology foundation model (ViT-L/16,
# DINOv2, mass100k) with a CORN (Conditional Ordinal Regression
# for Neural networks) head. The head outputs num_classes - 1 = 3
# logits, and the loss/prediction use the official coral-pytorch
# functions (corn_loss, corn_label_from_logits).
#
# This tests whether adding ordinal supervision to UNI improves
# the clinically important boundary classes (class_1+ recall,
# class_2+ precision, macro-F1).
#
# UNI loading recipe is EXACTLY the official UNI recipe:
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


class UNICORN(nn.Module):
    """UNI (ViT-L/16) backbone + CORN ordinal regression head.

    Args:
        checkpoint_path (str): Absolute path to the local UNI
            checkpoint (pytorch_model.bin).
        num_classes (int): Number of ordinal classes (default 4).
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
            print(f"  [UNI-CORN] Loaded backbone strictly from '{checkpoint_path}'")

        self.backbone_feature_dim = self.backbone.num_features  # 1024 for ViT-L/16

        # CORN head: num_classes - 1 = 3 logits
        self.head = nn.Linear(self.backbone_feature_dim, num_classes - 1)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: raw RGB in [0, 1], shape [B, 3, 224, 224]
        # Normalize INSIDE the model (dataloader returns raw RGB).
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std

        features = self.backbone(x_normalized)  # [B, 1024] pooled feature
        if features.dim() == 3:
            # [B, 197, 1024] -> CLS token [B, 1024]
            features = features[:, 0]

        logits = self.head(features)  # [B, 3]

        # Reconstruct genuine unconditional class probabilities from CORN sigmoids:
        s = torch.sigmoid(logits)          # [B, 3]
        p_gt0 = s[:, 0]                    # P(y > 0)
        p_gt1 = s[:, 1]                    # P(y > 1 | y > 0)
        p_gt2 = s[:, 2]                    # P(y > 2 | y > 1)

        p0 = 1 - p_gt0
        p1 = p_gt0 * (1 - p_gt1)
        p2 = p_gt0 * p_gt1 * (1 - p_gt2)
        p3 = p_gt0 * p_gt1 * p_gt2

        probs = torch.stack([p0, p1, p2, p3], dim=1)  # [B, 4], sums to 1 by construction

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