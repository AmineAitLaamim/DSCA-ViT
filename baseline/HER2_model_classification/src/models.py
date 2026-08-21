from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


class HybridHER2Ensemble(nn.Module):
    """End-to-end hybrid ensemble from the HER2-IHC paper."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        num_classes = len(config["classes"])
        pretrained = bool(model_cfg["pretrained"])
        image_size = int(config["dataset"]["image_size"])

        self.eva = self._create_timm_model(model_cfg["eva_name"], pretrained, image_size=image_size)
        self.vit = self._create_timm_model(model_cfg["vit_name"], pretrained, image_size=image_size)
        self.convnext = timm.create_model(model_cfg["convnext_name"], pretrained=pretrained, num_classes=0)

        self.feature_dims = {
            "eva": int(model_cfg["eva_dim"]),
            "vit": int(model_cfg["vit_dim"]),
            "convnext": int(model_cfg["convnext_dim"]),
        }
        fusion_dim = int(model_cfg["fusion_dim"])
        expected = sum(self.feature_dims.values())
        if fusion_dim != expected:
            raise ValueError(f"fusion_dim={fusion_dim} does not equal expected feature sum={expected}")

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(model_cfg["fusion_hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(int(model_cfg["fusion_hidden_dim"]), num_classes),
        )

    @staticmethod
    def _create_timm_model(name: str, pretrained: bool, image_size: int) -> nn.Module:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=image_size)
        except TypeError:
            return timm.create_model(name, pretrained=pretrained, num_classes=0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h_eva = self.eva(x)
        h_vit = self.vit(x)
        h_convnext = self.convnext(x)
        return torch.cat([h_eva, h_vit, h_convnext], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(config: dict[str, Any]) -> HybridHER2Ensemble:
    return HybridHER2Ensemble(config)
