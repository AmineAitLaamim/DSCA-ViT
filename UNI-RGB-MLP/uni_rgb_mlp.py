# ============================================================
# UNI-RGB-MLP — Main Model (RGB-Only Ablation)
# ============================================================
#
# Control ablation: frozen UNI histopathology foundation model
# (ViT-L/16, DINOv2, mass100k) as a feature extractor + an MLP
# head on RGB features ONLY. No stain branch.
#
# This isolates whether the stain side-information in
# UNI-Stain-MLP (UNI_v2/) actually helps.
#
# UNI loading recipe is EXACTLY the official UNI recipe:
#   - timm "vit_large_patch16_224" with img_size=224,
#     patch_size=16, init_values=1e-5, num_classes=0,
#     dynamic_img_size=True
#   - strict load_state_dict from the local checkpoint path
#
# The UNI backbone is frozen FOREVER. The train() override keeps
# the backbone in eval mode at all times. The backbone forward
# pass is wrapped in an explicit torch.no_grad() block.
#
# No network calls, no Hugging Face downloads.
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Dict

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UNIRGBMLP(nn.Module):
    """Frozen UNI feature extractor + RGB-only MLP head.

    Args:
        checkpoint_path (str): Absolute path to the local UNI
            checkpoint (pytorch_model.bin).
        num_classes (int): Number of classification classes.
        verbose (bool): Whether to print load confirmation.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 4, verbose: bool = False) -> None:
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
            print(f"  [UNI-RGB-MLP] Loaded frozen UNI backbone from '{checkpoint_path}'")

        # --------------------------------------------------------
        # 2. RGB-only MLP head
        # --------------------------------------------------------
        self.rgb_feat_dim = 2048  # cls (1024) + gap (1024)
        self.head = nn.Sequential(
            nn.Linear(self.rgb_feat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    # ------------------------------------------------------------
    # train() override — keep frozen backbone in eval mode always
    # ------------------------------------------------------------
    def train(self, mode: bool = True) -> "UNIRGBMLP":
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
        x_rgb_norm = (x - mean) / std

        # ---- Frozen UNI features inside explicit no_grad ----
        with torch.no_grad():
            tokens = self.backbone.forward_features(x_rgb_norm)  # [B, 197, 1024]
            if tokens.dim() != 3:
                # Fallback: some timm versions return pooled features [B, 1024].
                # Duplicate to maintain the [B, 2048] contract.
                rgb_feat = torch.cat([tokens, tokens], dim=1)  # [B, 2048]
            else:
                cls = tokens[:, 0]                # [B, 1024]
                patch_tokens = tokens[:, 1:]      # [B, 196, 1024]
                gap = patch_tokens.mean(dim=1)    # [B, 1024]
                rgb_feat = torch.cat([cls, gap], dim=1)  # [B, 2048]

        # ---- MLP head ----
        logits = self.head(rgb_feat)              # [B, 4]
        probs = torch.softmax(logits, dim=1)

        return {"logits": logits, "probs": probs}

    # ------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + head_params

        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": total,
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }