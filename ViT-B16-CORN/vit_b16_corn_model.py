# ============================================================
# ViT-B16-CORN — Main Model (Ordinal Regression Baseline)
# ============================================================
#
# ViT-B16 backbone with a CORN (Conditional Ordinal Regression
# for Neural networks) head. The head outputs num_classes - 1 = 3
# logits, and the loss/prediction use the official coral-pytorch
# functions (corn_loss, corn_label_from_logits).
#
# This is a single-variable change from the proper ViT-B16
# baseline: only the output layer and loss are changed from
# standard 4-way cross-entropy to CORN ordinal regression.
#
# The dataloader applies ImageNet normalization (matching the
# proper ViT-B16 baseline convention). No normalization inside
# the model.
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


class ViTB16CORN(nn.Module):
    """ViT-B16 backbone + CORN ordinal regression head.

    Args:
        num_classes (int): Number of ordinal classes (default 4).
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Same ViT-B16 backbone as the proper baseline, but num_classes=0
        # so we can attach a custom CORN head.
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.in_features = self.backbone.num_features  # 768

        # CORN head: num_classes - 1 = 3 logits
        self.head = nn.Linear(self.in_features, num_classes - 1)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: ImageNet-normalized RGB [B, 3, 224, 224] (dataloader normalizes)
        features = self.backbone(x)   # [B, 768]
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