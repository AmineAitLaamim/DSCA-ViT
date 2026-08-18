# ============================================================
# DSCA-ViT v3 - Multi-Scale Construction
# ============================================================
#
# CoarseScaleView: creates a low-frequency / coarse multi-resolution
# view of the SAME 224x224 field.
#
# Implementation:
#     224x224 -> bilinear downsample -> 112x112 -> bilinear upsample -> 224x224
#
# NOTE: This is NOT a larger spatial context and NOT a genuine change
# of magnification. The physical field of view remains identical; the
# coarse branch is only a coarser (lower spatial frequency) view of the
# same field, which the shared ViT can use as a contextual prior.
#
# CoarseScaleView has NO trainable parameters.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseScaleView(nn.Module):
    """
    Produces a low-frequency coarse view of a 3-channel image.

    Input : [B, 3, H, W]   (H = W = image_size, e.g. 224)
    Output: [B, 3, H, W]   (same spatial size, low-frequency content)

    The module contains no parameters: it is a deterministic
    downsample -> upsample operation.
    """

    def __init__(self, coarse_size: int = 112) -> None:
        """
        Parameters
        ----------
        coarse_size : int
            Intermediate resolution for the coarse view (e.g. 112).
        """
        super().__init__()
        self.coarse_size = coarse_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image, shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Low-frequency coarse view, shape (B, 3, H, W).
        """
        # (B, 3, H, W) -> (B, 3, coarse_size, coarse_size)
        coarse = F.interpolate(
            x,
            size=(self.coarse_size, self.coarse_size),
            mode="bilinear",
            align_corners=False,
        )
        # (B, 3, coarse_size, coarse_size) -> (B, 3, H, W)
        coarse = F.interpolate(
            coarse,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return coarse