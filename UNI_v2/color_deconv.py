# ============================================================
# UNI-Stain-MLP — Color Deconvolution (self-contained copy)
# ============================================================
#
# Self-contained copy of the Ruifrok & Johnston ColorDeconvolution
# from models_v2_2/color_deconv.py (identical implementation).
#
# Physics-based (non-learnable): converts RGB IHC images into
# Hematoxylin (H) and DAB stain concentrations via the fixed
# H-DAB stain matrix and the Beer-Lambert law.
#
# Existing packages (models_v2_2/ etc.) are NOT modified.
# ============================================================

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class ColorDeconvolution(nn.Module):
    """
    Color Deconvolution Module using the Ruifrok & Johnston method.

    This module separates RGB Immunohistochemistry (IHC) images into
    Hematoxylin and DAB channels by converting the images to Optical
    Density (OD) space and applying the inverse of the H-DAB stain matrix.

    This is a fixed (non-learnable) preprocessing step based on the
    Beer-Lambert law.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

        # Ruifrok H-DAB stain matrix (rows are stain vectors in RGB OD space)
        stain_matrix = torch.tensor(
            [
                [0.6500286, 0.7040310, 0.2860126],
                [0.2688606, 0.5700937, 0.7767574],
                [0.7110272, 0.4234194, 0.5615672],
            ],
            dtype=torch.float32,
        )

        stain_matrix_inv = torch.linalg.inv(stain_matrix)
        self.register_buffer("stain_matrix_inv", stain_matrix_inv)

    def forward(self, x_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies color deconvolution to raw RGB images in [0, 1].

        Args:
            x_rgb (torch.Tensor): (B, 3, H, W) RGB tensor in [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (H, DAB) each (B, 1, H, W),
                clamped to be >= 0.
        """
        od = -torch.log10(x_rgb + self.epsilon)
        od_reshaped = od.permute(0, 2, 3, 1)
        stains = torch.matmul(od_reshaped, self.stain_matrix_inv)
        stains = stains.permute(0, 3, 1, 2)

        h_channel = stains[:, 0:1, :, :]
        dab_channel = stains[:, 1:2, :, :]

        h_channel = torch.clamp(h_channel, min=0.0)
        dab_channel = torch.clamp(dab_channel, min=0.0)

        return h_channel, dab_channel


def deconvolve_numpy(image_rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for numpy-based color deconvolution.

    Args:
        image_rgb_uint8 (np.ndarray): (H, W, 3) uint8 RGB image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (H, DAB) float32 arrays of shape (H, W).
    """
    image_float = image_rgb_uint8.astype(np.float32) / 255.0
    epsilon = 1e-6
    od = -np.log10(image_float + epsilon)

    stain_matrix = np.array(
        [
            [0.6500286, 0.7040310, 0.2860126],
            [0.2688606, 0.5700937, 0.7767574],
            [0.7110272, 0.4234194, 0.5615672],
        ],
        dtype=np.float32,
    )
    stain_matrix_inv = np.linalg.inv(stain_matrix)

    stains = np.dot(od, stain_matrix_inv)
    h_channel = np.clip(stains[:, :, 0], a_min=0.0, a_max=None)
    dab_channel = np.clip(stains[:, :, 1], a_min=0.0, a_max=None)

    return h_channel, dab_channel