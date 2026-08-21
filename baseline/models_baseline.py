# ============================================================
# Plain ViT-B16 Baseline — Model
# ============================================================
#
# Plain ViT-B16 (timm vit_base_patch16_224) for HER2-IHC
# 4-class classification. Matches the original Colab baseline.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class PlainViTB16(nn.Module):
    """
    Plain ViT-B/16 baseline model.

    Loads a timm ViT-B/16 (ImageNet pretrained) with a fresh
    4-class classification head.

    Forward returns a dict with keys:
      - logits : (B, num_classes)
      - probs  : (B, num_classes)
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        # Use the exact same timm model as the original Colab baseline
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.in_features = self.vit.num_features  # 768

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        logits = self.vit(x)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

    # ------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())

        head_params = sum(
            p.numel() for n, p in self.vit.named_parameters()
            if n.startswith("head.")
        )
        backbone_params = total - head_params

        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": total,
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }