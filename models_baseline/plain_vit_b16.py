# ============================================================
# Plain ViT-B16 Baseline Model
# ============================================================
#
# Reproduces the plain ViT-B16 baseline that reached 95.02%
# official test accuracy on the HER2-IHC-40x dataset.
#
#   - timm vit_base_patch16_224, ImageNet-pretrained
#   - Internal ImageNet normalization (mean/std applied in forward)
#   - Dataloaders return raw RGB [0,1] — NO normalize in transforms
#   - Returns a dict: {logits: [B,4], probs: [B,4]}
#
# This package is independent of models/, models_v2/, models_v3/,
# and models_v2_1/.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict


class PlainViTB16(nn.Module):
    """
    Plain ViT-B/16 classification baseline.

    Encapsulates the ImageNet-pretrained ViT with the classification
    head, applying ImageNet normalization internally so the
    dataloaders can return raw RGB [0,1].

    Args:
        num_classes (int): Number of output classes (default 4).
        pretrained (bool): Load ImageNet weights (default True).
            Set False for sanity checks / tests.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Pretrained ViT-B/16 with the classification head.
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        # ImageNet normalization (applied inside forward)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Raw RGB images, shape (B, 3, H, W),
                              values in [0, 1].

        Returns:
            Dict[str, torch.Tensor]:
                - logits: shape (B, num_classes)
                - probs: softmax probabilities, shape (B, num_classes)
        """
        # Internal ImageNet normalization
        x_norm = (x - self.imagenet_mean) / self.imagenet_std

        logits = self.vit(x_norm)  # (B, 4)

        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
        }

    def count_parameters(self) -> Dict[str, int]:
        """
        Counts backbone (patch_embed + blocks + norm) and head
        (fc/norm) parameters separately.

        Returns:
            Dict[str, int]: {'backbone', 'head', 'total', 'trainable'}.
        """
        # timm splits: head.fc (linear), head.norm (LayerNorm)
        backbone_params = 0
        head_params = 0
        for name, p in self.vit.named_parameters():
            if name.startswith("head."):
                head_params += p.numel()
            else:
                backbone_params += p.numel()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": total,
            "trainable": trainable,
        }