# ============================================================
# DSCA-ViT v2 — Input Adapters
# ============================================================
#
# New input-side components for DSCA-ViT v2:
#
#   StainNorm1ch          : per-stain learnable spatial normalization
#                           (GroupNorm(1,1), affine=True) on [B,1,H,W]
#   StainAdapter          : 1 -> 32 -> 3 nonlinear adapter on [B,1,H,W] -> [B,3,H,W]
#   LearnableChannelAffine: per-channel scale/bias on [B,3,H,W] (identity init)
#
# These components are NEW in v2. They are NOT part of the original
# DSCA-ViT implementation and must not be shared between H and DAB.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn


class StainNorm1ch(nn.Module):
    """
    Per-stain learnable spatial normalization for a single-channel stain image.

    Implementation:
        GroupNorm(num_groups=1, num_channels=1, affine=True)

    This is a per-image spatial normalization mechanism (NOT ImageNet
    normalization). It has trainable scale and bias, is independent of
    batch statistics, and is suitable for small medical-image batches.

    Input : [B, 1, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=1,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class StainAdapter(nn.Module):
    """
    Nonlinear 1 -> 32 -> 3 stain adapter.

    Architecture:
        1 channel
            ↓
        Conv2d(1, hidden_channels, kernel_size=3, padding=1)
            ↓
        GELU
            ↓
        Conv2d(hidden_channels, 3, kernel_size=3, padding=1)
            ↓
        3 channels

    Initialization (deterministic, NOT zero):
        conv1: kaiming_normal_(mode="fan_out", nonlinearity="relu"), bias = 0
        conv2: kaiming_normal_(mode="fan_out", nonlinearity="relu"),
               weight *= adapter_final_scale (default 0.1), bias = 0

    The adapter output is a meaningful 1->3 projection — the ViT must
    receive real input at initialization.

    Input : [B, 1, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(self, hidden_channels: int = 32, adapter_final_scale: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.adapter_final_scale = adapter_final_scale

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.conv1.bias)

        nn.init.kaiming_normal_(
            self.conv2.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        self.conv2.weight.data.mul_(self.adapter_final_scale)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class LearnableChannelAffine(nn.Module):
    """
    Learnable per-channel scale/bias for a 3-channel image.

    Implementation:
        y = x * scale + bias

    Parameters:
        scale = nn.Parameter(torch.ones(1, 3, 1, 1))
        bias  = nn.Parameter(torch.zeros(1, 3, 1, 1))

    At initialization: scale = 1, bias = 0, so output = input (identity).

    This gives the model a learnable way to adapt the three channels to
    the pretrained ViT WITHOUT normalizing away absolute stain intensity
    (important because DAB intensity carries biological information).

    Input : [B, 3, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias