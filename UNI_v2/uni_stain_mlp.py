# ============================================================
# UNI-Stain-MLP — Main Model
# ============================================================
#
# Frozen UNI histopathology foundation model (ViT-L/16, DINOv2,
# mass100k) as a feature extractor + H/DAB stain side-information
# through a small trainable StainEncoder branch, fused and fed
# into an MLP classification head.
#
# UNI loading recipe is EXACTLY the official UNI recipe:
#   - timm "vit_large_patch16_224" with img_size=224,
#     patch_size=16, init_values=1e-5, num_classes=0,
#     dynamic_img_size=True
#   - strict load_state_dict from the local checkpoint path
#
# The UNI backbone is frozen FOREVER. The train() override keeps
# the backbone in eval mode at all times.
#
# No network calls and no Hugging Face downloads anywhere in
# this file; weights are loaded strictly from the local
# checkpoint file.
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Dict

import timm
import torch
import torch.nn as nn

import color_deconv
import stain_encoder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UNIStainMLP(nn.Module):
    """Frozen UNI feature extractor + stain side-information + MLP head.

    Args:
        checkpoint_path (str): Absolute path to the local UNI
            checkpoint (pytorch_model.bin).
        num_classes (int): Number of classification classes.
        stain_dim (int): Stain encoder output dimension (default 512).
        stain_stats (dict): Dict with h_mean, h_std, dab_mean, dab_std.
        verbose (bool): Whether to print load confirmation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 4,
        stain_dim: int = 512,
        stain_stats: Dict[str, float] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"UNI checkpoint not found at '{checkpoint_path}'. "
                "This model requires the local UNI weights."
            )

        # --------------------------------------------------------
        # 1. Frozen UNI backbone (exact official recipe)
        # --------------------------------------------------------
        st_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.backbone.load_state_dict(st_dict, strict=True)

        # Freeze backbone forever
        for param in self.backbone.parameters():
            param.requires_grad = False

        if verbose:
            print(f"  [UNI-Stain-MLP] Loaded frozen UNI backbone from '{checkpoint_path}'")

        # --------------------------------------------------------
        # 2. Stain branch
        # --------------------------------------------------------
        self.deconv = color_deconv.ColorDeconvolution()
        self.stain_encoder = stain_encoder.StainEncoder(
            in_channels=2, stain_dim=stain_dim
        )

        if stain_stats is None:
            raise ValueError(
                "stain_stats dict is required (h_mean, h_std, dab_mean, dab_std)."
            )
        self.register_buffer("h_mean", torch.tensor(float(stain_stats["h_mean"])))
        self.register_buffer("h_std", torch.tensor(float(stain_stats["h_std"])))
        self.register_buffer("dab_mean", torch.tensor(float(stain_stats["dab_mean"])))
        self.register_buffer("dab_std", torch.tensor(float(stain_stats["dab_std"])))

        # --------------------------------------------------------
        # 3. Fusion + MLP head
        # --------------------------------------------------------
        self.rgb_feat_dim = 2048  # cls (1024) + gap (1024)
        self.stain_dim = stain_dim
        self.head = nn.Sequential(
            nn.Linear(self.rgb_feat_dim + stain_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    # ------------------------------------------------------------
    # train() override — keep frozen backbone in eval mode always
    # ------------------------------------------------------------
    def train(self, mode: bool = True) -> "UNIStainMLP":
        """Override train() to keep the frozen UNI backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()  # backbone stays in eval mode regardless of outer mode
        return self

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: raw RGB in [0, 1], shape [B, 3, 224, 224]

        # ---- RGB stream: normalize INSIDE the model ----
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        # Token sequence [B, 197, 1024]
        tokens = self.backbone.forward_features(x_norm)
        if tokens.dim() != 3:
            # Fallback: some timm versions return pooled features [B, 1024].
            # Duplicate to maintain the [B, 2048] contract.
            rgb_feat = torch.cat([tokens, tokens], dim=1)  # [B, 2048]
        else:
            cls = tokens[:, 0]                # [B, 1024]
            gap = tokens[:, 1:].mean(dim=1)   # [B, 1024]
            rgb_feat = torch.cat([cls, gap], dim=1)  # [B, 2048]

        # ---- Stain stream ----
        h, dab = self.deconv(x)  # each [B, 1, H, W]
        h_norm = (h - self.h_mean.to(x.device)) / self.h_std.to(x.device)
        dab_norm = (dab - self.dab_mean.to(x.device)) / self.dab_std.to(x.device)
        stain_input = torch.cat([h_norm, dab_norm], dim=1)  # [B, 2, H, W]
        stain_feat = self.stain_encoder(stain_input)        # [B, stain_dim]

        # ---- Fusion + head ----
        combined = torch.cat([rgb_feat, stain_feat], dim=1)  # [B, 2048 + stain_dim]
        logits = self.head(combined)                         # [B, 4]
        probs = torch.softmax(logits, dim=1)

        return {"logits": logits, "probs": probs}

    # ------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        stain_params = sum(p.numel() for p in self.stain_encoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + stain_params + head_params

        return {
            "backbone": backbone_params,
            "stain_encoder": stain_params,
            "head": head_params,
            "total": total,
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }