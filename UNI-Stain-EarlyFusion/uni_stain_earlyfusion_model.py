# UNI-Stain-EarlyFusion -- Main Model
#
# UNI (ViT-L/16, DINOv2, mass100k) modified to accept a 5-channel input
#   [norm RGB | norm H | norm DAB]
# fused at the first layer by REPLACING the patch-embed projection.
#
# CRITICAL ORDER (strict-load-before-replace):
#   1. create ORIGINAL backbone, 2. strict-load UNI checkpoint,
#   3. replace patch_embed.proj with Conv2d(5,...), 4. copy RGB weights
#      into ch [0:3], bias copied, ch [3:5] (H/DAB) zero-init.
#
# After replacement the model CANNOT be strict-loaded with the original
# UNI checkpoint. For inference we rebuild the same modified projection,
# then load the saved TRAINING checkpoint.
#
# Early fusion via UNI self-attention: once the 5-channel image is
# patch-embedded, transformer self-attention fuses morphology + stain
# in every layer. No cross-attention needed. No network calls.

from __future__ import annotations

import os

# Force any accidental Hugging Face / network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Dict

import timm
import torch
import torch.nn as nn

import color_deconv

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UNIStainEarlyFusion(nn.Module):
    """UNI backbone + 5-channel RGB+H+DAB early fusion + 4-class head.

    Args:
        checkpoint_path: local UNI checkpoint loaded BEFORE projection replace.
        stain_stats: dict with h_mean, h_std, dab_mean, dab_std.
        num_classes: number of classes (default 4).
        verbose: print load/init confirmation.
    """

    def __init__(self, checkpoint_path, stain_stats, num_classes=4,
                 verbose=False):
        super().__init__()
        if stain_stats is None:
            raise ValueError("stain_stats required for UNIStainEarlyFusion.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"UNI checkpoint not found at '{checkpoint_path}'. "
                "This model requires the local UNI weights."
            )
        self.num_classes = num_classes

        # ---- Step 1 + 2: original backbone, then strict-load UNI ----
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=224, patch_size=16, init_values=1e-5,
            num_classes=0, dynamic_img_size=True,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        self.backbone.load_state_dict(state_dict, strict=True)
        self._uni_loaded = True
        self._orig_proj_weight = self.backbone.patch_embed.proj.weight.data.clone()
        if verbose:
            print(f"  [UNI-Stain-EarlyFusion] Strict-loaded UNI from "
                  f"'{checkpoint_path}'")

        # ---- Step 3 + 4: replace projection -> Conv2d(5, ...) ----
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=5,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight[:, :3].copy_(old_proj.weight)
            if new_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
            new_proj.weight[:, 3:].zero_()  # H/DAB zero-init
        self.backbone.patch_embed.proj = new_proj
        self._proj_in_channels = 5

        # ---- Stain stats (global) ----
        for key in ("h_mean", "h_std", "dab_mean", "dab_std"):
            if key not in stain_stats:
                raise ValueError(f"Stain stats missing key '{key}'.")
        self.register_buffer("h_mean", torch.tensor(float(stain_stats["h_mean"])))
        self.register_buffer("h_std", torch.tensor(float(stain_stats["h_std"])))
        self.register_buffer("dab_mean", torch.tensor(float(stain_stats["dab_mean"])))
        self.register_buffer("dab_std", torch.tensor(float(stain_stats["dab_std"])))

        self.deconv = color_deconv.ColorDeconvolution()

        # ---- Head ----
        self.backbone_feature_dim = self.backbone.num_features  # 1024
        self.head = nn.Linear(self.backbone_feature_dim, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: raw RGB [0,1], [B,3,224,224]
        dev = x.device

        # RGB normalization (ImageNet)
        mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        rgb_norm = (x - mean) / std  # [B,3,224,224]

        # Stain deconvolution + normalization
        h, dab = self.deconv(x)  # each [B,1,224,224]
        h_norm = (h - self.h_mean.to(dev)) / self.h_std.to(dev)
        dab_norm = (dab - self.dab_mean.to(dev)) / self.dab_std.to(dev)

        # 5-channel early fusion
        x5 = torch.cat([rgb_norm, h_norm, dab_norm], dim=1)  # [B,5,224,224]

        features = self.backbone(x5)
        if features.dim() == 3:
            features = features[:, 0]  # CLS token [B,1024]

        logits = self.head(features)  # [B,4]
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}

    # ---- Parameter counts ----
    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + head_params
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }