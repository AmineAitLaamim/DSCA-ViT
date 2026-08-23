# ============================================================
# UNI-Stain-Attention — StainEncoder
# ============================================================
#
# A lightweight CNN that converts normalized H/DAB stain input
# [B, 2, H, W] into a fixed feature vector [B, stain_dim].
#
# Architecture (same as UNI_v2/stain_encoder.py):
#   Conv2d(2, 32, 3, stride=2) -> BN -> GELU
#   Conv2d(32, 64, 3, stride=2) -> BN -> GELU
#   Conv2d(64, 128, 3, stride=2) -> BN -> GELU
#   Conv2d(128, 256, 3, stride=2) -> BN -> GELU
#   AdaptiveAvgPool2d((1,1))
#   Flatten
#   Linear(256, stain_dim) -> GELU
#
# Total params < 2M (default stain_dim=512 gives ~0.5M).
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn


class StainEncoder(nn.Module):
    """
    Converts normalized H/DAB stain input into a fixed feature vector.

    Args:
        in_channels (int): Number of input channels (2 = H + DAB).
        stain_dim (int): Output feature dimension (default 512).
    """

    def __init__(self, in_channels: int = 2, stain_dim: int = 512) -> None:
        super().__init__()

        self.stain_dim = stain_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]

        self.head = nn.Sequential(
            nn.Linear(256, stain_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Normalized stain input, shape (B, 2, H, W).

        Returns:
            torch.Tensor: Stain feature vector, shape (B, stain_dim).
        """
        x = self.stem(x)          # (B, 256, H/16, W/16)
        x = self.pool(x)          # (B, 256, 1, 1)
        x = x.flatten(1)          # (B, 256)
        x = self.head(x)          # (B, stain_dim)
        return x

    def count_parameters(self) -> int:
        """Returns the total number of parameters in the encoder."""
        return sum(p.numel() for p in self.parameters())