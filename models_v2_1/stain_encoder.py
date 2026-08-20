# ============================================================
# DSS-ViT — StainEncoder
# ============================================================
#
# A lightweight CNN that converts the normalized H/DAB stain
# input [B, 2, H, W] into a fixed number of tokens [B, M, D].
#
# Architecture (bottleneck, ~3-4M params):
#   Conv2d(2, 32, 3, stride=2) -> BN -> GELU   # [B,32,H/2,W/2]
#   Conv2d(32, 64, 3, stride=2) -> BN -> GELU  # [B,64,H/4,W/4]
#   Conv2d(64, 128, 3, stride=2) -> BN -> GELU # [B,128,H/8,W/8]
#   Conv2d(128, 256, 3, stride=2) -> BN -> GELU # [B,256,H/16,W/16]
#   AdaptiveAvgPool2d((4,4))                    # [B,256,4,4]
#   Flatten                                     # [B,4096]
#   Linear(4096, bottleneck_dim) -> GELU        # bottleneck
#   Linear(bottleneck_dim, M * D)               # [B, M*D]
#   Reshape to [B, M, D]
#
# The branch is kept lightweight relative to the ~86M ViT backbone.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn


class StainEncoder(nn.Module):
    """
    Converts normalized H/DAB stain input into a fixed number of tokens.

    Args:
        in_channels (int): Number of input channels (2 = H + DAB).
        out_dim (int): Token embedding dimension (matches ViT embed dim).
        num_tokens (int): Number of output tokens.
        bottleneck_dim (int): Hidden bottleneck dimension.
        image_size (int): Input spatial size (assumed square).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_dim: int = 768,
        num_tokens: int = 16,
        bottleneck_dim: int = 512,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.image_size = image_size

        # --------------------------------------------------------
        # CNN stem (stride-2 downsampling, no heavy downsampling)
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Pooling + bottleneck + token projection
        # --------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # After 4 stride-2 convs on 224 -> 14x14; pool to 4x4 -> 16 spatial
        # positions. Flattened: 256 * 16 = 4096.
        self.flatten_dim = 256 * 4 * 4  # 4096

        self.bottleneck = nn.Sequential(
            nn.Linear(self.flatten_dim, bottleneck_dim),
            nn.GELU(),
        )

        self.token_proj = nn.Linear(bottleneck_dim, num_tokens * out_dim)

        # --------------------------------------------------------
        # Initialization
        # --------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes linear layers with Xavier uniform and zero bias."""
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
            torch.Tensor: Stain tokens, shape (B, num_tokens, out_dim).
        """
        # CNN stem
        x = self.stem(x)          # (B, 256, H/16, W/16)

        # Adaptive average pool to fixed grid
        x = self.pool(x)          # (B, 256, 4, 4)

        # Flatten
        x = x.flatten(1)          # (B, 4096)

        # Bottleneck
        x = self.bottleneck(x)    # (B, bottleneck_dim)

        # Token projection
        x = self.token_proj(x)    # (B, num_tokens * out_dim)

        # Reshape to tokens
        x = x.view(-1, self.num_tokens, self.out_dim)  # (B, M, D)

        return x

    def count_parameters(self) -> int:
        """Returns the total number of parameters in the encoder."""
        return sum(p.numel() for p in self.parameters())