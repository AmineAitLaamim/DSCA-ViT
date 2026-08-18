# ============================================================
# DSCA-ViT v3 - Stain-Domain Augmentation
# ============================================================
#
# StainAugmentation is a TRAINING-ONLY, deterministic (seeded)
# stain-domain augmentation transform.
#
# It is implemented as a torchvision-style transform (NOT an
# nn.Module) so that it belongs to the training dataset pipeline.
# Validation and test pipelines never include it, making the
# train/eval separation explicit and auditable.
#
# It perturbs, with probability `probability`:
#   * RGB brightness (pre-deconvolution)
#   * RGB contrast   (pre-deconvolution)
#   * H  OD stain concentration (in stain space, Ruifrok H-DAB matrix)
#   * DAB OD stain concentration (in stain space, Ruifrok H-DAB matrix)
#
# The concentration perturbation is performed in OD / stain space and
# then reconstructed back to RGB, which models plausible histopathology
# staining variation. All ranges are deliberately conservative:
# DAB information is never randomly destroyed.
#
# StainAugmentation has ZERO trainable parameters.
# ============================================================

from __future__ import annotations

import random

import torch


class StainAugmentation:
    """
    Stain-domain augmentation for a single RGB image tensor [3, H, W].

    Input : torch.Tensor in [0, 1], shape [C=3, H, W]
    Output: torch.Tensor in [0, 1], shape [C=3, H, W] (possibly perturbed)

    Parameters
    ----------
    probability : float
        Probability of applying the augmentation. Default 0.5.
    h_concentration_range : tuple[float, float]
        Multiplier range for the Hematoxylin OD concentration. Default (0.85, 1.15).
    dab_concentration_range : tuple[float, float]
        Multiplier range for the DAB OD concentration. Default (0.85, 1.15).
    brightness_range : tuple[float, float]
        Global RGB brightness multiplier range. Default (0.9, 1.1).
    contrast_range : tuple[float, float]
        Global RGB contrast multiplier range. Default (0.9, 1.1).
    """

    def __init__(
        self,
        probability: float = 0.5,
        h_concentration_range: tuple[float, float] = (0.85, 1.15),
        dab_concentration_range: tuple[float, float] = (0.85, 1.15),
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        self.probability = probability
        self.h_concentration_range = h_concentration_range
        self.dab_concentration_range = dab_concentration_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        # Ruifrok H-DAB stain matrix (rows are stain vectors in RGB OD space)
        # H: Hematoxylin, DAB: 3,3'-Diaminobenzidine, Res: Residual
        # This is the SAME matrix used by models_v3/color_deconv.py.
        stain_matrix = torch.tensor(
            [
                [0.6500286, 0.7040310, 0.2860126],
                [0.2688606, 0.5700937, 0.7767574],
                [0.7110272, 0.4234194, 0.5615672],
            ],
            dtype=torch.float32,
        )
        self.stain_matrix = stain_matrix
        self.stain_matrix_inv = torch.linalg.inv(stain_matrix)

        self.epsilon = 1e-6

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Applies the stain-domain augmentation to a single RGB image.

        Parameters
        ----------
        rgb : torch.Tensor
            RGB image, shape (3, H, W), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Augmented RGB image, shape (3, H, W), values in [0, 1].
        """
        if random.random() > self.probability:
            return rgb

        # --------------------------------------------------------
        # Draw perturbation factors (deterministic given the RNG seed)
        # --------------------------------------------------------
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        h_mult = random.uniform(*self.h_concentration_range)
        dab_mult = random.uniform(*self.dab_concentration_range)

        # --------------------------------------------------------
        # RGB-space brightness / contrast (pre-deconvolution)
        # --------------------------------------------------------
        x = rgb * brightness  # brightness

        # Per-channel mean contrast adjustment
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * contrast + mean

        # --------------------------------------------------------
        # Stain-space H / DAB concentration perturbation
        # --------------------------------------------------------
        # Convert to OD space: OD = -log10(I / I0), I0 = 1
        od = -torch.log10(x + self.epsilon)  # (3, H, W)

        # OD (H, W, 3) @ M_inv (3, 3) -> stain concentrations (H, W, 3)
        stains = od.permute(1, 2, 0) @ self.stain_matrix_inv

        # Perturb H (channel 0) and DAB (channel 1) concentrations
        stains[..., 0] = stains[..., 0] * h_mult
        stains[..., 1] = stains[..., 1] * dab_mult

        # Reconstruct OD: stains (H, W, 3) @ M (3, 3) -> OD (H, W, 3)
        od_perturbed = stains @ self.stain_matrix

        # Reconstruct RGB: I = 10^(-OD)
        rgb_recon = torch.pow(10.0, -od_perturbed)  # (H, W, 3)
        rgb_recon = rgb_recon.permute(2, 0, 1)  # (3, H, W)

        # Keep values in [0, 1] (the model's color deconvolution
        # requires raw RGB in [0, 1]).
        rgb_recon = torch.clamp(rgb_recon, min=0.0, max=1.0)

        return rgb_recon