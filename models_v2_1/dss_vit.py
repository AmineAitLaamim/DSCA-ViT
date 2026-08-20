# ============================================================
# DSS-ViT — Main Architecture Assembly
# ============================================================
#
# Dual-Stream Stain Vision Transformer for HER2 IHC Scoring.
#
# Architecture:
#
#   Raw RGB [B,3,H,W] in [0,1]
#        │
#        ├─────────────────────────────┐
#        │                             │
#        ▼                             ▼
#   ColorDeconvolution            Normalize RGB
#        │                      (ImageNet mean/std)
#        ├──── H [B,1,H,W]           │
#        ├──── DAB [B,1,H,W]         ▼
#        │                      ViT-B16 (timm)
#        ▼                             │
#    StainEncoder                    features [B,197,768]
#        │                             │
#        ▼                             ├── x_cls [B,768]
#    StainTokens [B,16,768]             └── x_patch [B,196,768]
#        │
#        └───────────────┬─────────────┘
#                        ▼
#                Cross-Attention Fusion
#                        │
#                        ▼
#                   fused_cls [B,768]
#                        │
#                        ▼
#                 Ordinal Head
#                        │
#                        ▼
#                cutpoints [B,3]  →  probs [B,4]
#
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Dict, List

from .color_deconv import ColorDeconvolution
from .stain_encoder import StainEncoder
from .ordinal_head import OrdinalHead, cutpoints_to_probs


class DSSViT(nn.Module):
    """
    Dual-Stream Stain Vision Transformer.

    Uses RGB as the main input (pretrained ViT-B16) and H/DAB stain
    information as an auxiliary branch, fused via cross-attention
    with a gated residual, followed by an ordinal classification head.

    Args:
        num_classes (int): Number of ordinal classes (default 4).
        pretrained (bool): Whether to load ImageNet pretrained ViT-B16.
        num_stain_tokens (int): Number of stain tokens (default 16).
        stain_bottleneck_dim (int): StainEncoder bottleneck dim (default 512).
        stain_stats (dict): Global stain stats {h_mean, h_std, dab_mean, dab_std}.
        image_size (int): Input spatial size (default 224).
        classifier_dropout (float): Unused (kept for interface parity).
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        num_stain_tokens: int = 16,
        stain_bottleneck_dim: int = 512,
        stain_stats: Dict[str, float] | None = None,
        image_size: int = 224,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size

        # --------------------------------------------------------
        # 1. Color Deconvolution (fixed, no gradients)
        # --------------------------------------------------------
        self.color_deconv = ColorDeconvolution()

        # --------------------------------------------------------
        # 2. RGB Backbone (pretrained ViT-B16)
        # --------------------------------------------------------
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.embed_dim = self.vit.embed_dim  # 768

        # ImageNet normalization (applied inside forward)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # --------------------------------------------------------
        # 3. Stain normalization (global stats as buffers)
        # --------------------------------------------------------
        if stain_stats is None:
            stain_stats = {
                "h_mean": 0.0, "h_std": 1.0,
                "dab_mean": 0.0, "dab_std": 1.0,
            }

        self.register_buffer(
            "h_mean", torch.tensor(float(stain_stats["h_mean"]))
        )
        self.register_buffer(
            "h_std", torch.tensor(float(stain_stats["h_std"]))
        )
        self.register_buffer(
            "dab_mean", torch.tensor(float(stain_stats["dab_mean"]))
        )
        self.register_buffer(
            "dab_std", torch.tensor(float(stain_stats["dab_std"]))
        )

        # --------------------------------------------------------
        # 4. StainEncoder (auxiliary branch)
        # --------------------------------------------------------
        self.stain_encoder = StainEncoder(
            in_channels=2,
            out_dim=self.embed_dim,
            num_tokens=num_stain_tokens,
            bottleneck_dim=stain_bottleneck_dim,
            image_size=image_size,
        )

        # --------------------------------------------------------
        # 5. Cross-Attention Fusion
        # --------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=12,
            batch_first=True,
        )

        # Gate MLP: Linear(1536 -> 768) -> GELU -> Linear(768 -> 768)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Initialize the last Linear bias to 0 so gate ≈ 0.5 at init
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)

        # --------------------------------------------------------
        # 6. Ordinal Head
        # --------------------------------------------------------
        self.ordinal_head = OrdinalHead(
            in_dim=self.embed_dim,
            num_classes=num_classes,
        )

        # --------------------------------------------------------
        # Store config
        # --------------------------------------------------------
        self.num_stain_tokens = num_stain_tokens

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x_rgb (torch.Tensor): Raw RGB images, shape (B, 3, 224, 224),
                                  values in [0, 1].

        Returns:
            Dict[str, torch.Tensor]:
                - logits: cutpoint logits, shape (B, 3)
                - probs: class probabilities, shape (B, 4)
                - x_cls: ViT CLS features, shape (B, 768)
                - fused_cls: fused features, shape (B, 768)
                - mean_gate: scalar mean gate value
        """
        # --------------------------------------------------------
        # Step 1: Color Deconvolution (no gradients)
        # --------------------------------------------------------
        with torch.no_grad():
            h_channel, dab_channel = self.color_deconv(x_rgb)
        # h_channel: (B, 1, 224, 224), dab_channel: (B, 1, 224, 224)

        # --------------------------------------------------------
        # Step 2: Stain normalization + concat
        # --------------------------------------------------------
        h_norm = (h_channel - self.h_mean) / self.h_std
        dab_norm = (dab_channel - self.dab_mean) / self.dab_std
        stain_input = torch.cat([h_norm, dab_norm], dim=1)  # (B, 2, H, W)

        # --------------------------------------------------------
        # Step 3: StainEncoder -> stain tokens
        # --------------------------------------------------------
        stain_tokens = self.stain_encoder(stain_input)  # (B, 16, 768)

        # --------------------------------------------------------
        # Step 4: RGB normalization + ViT forward
        # --------------------------------------------------------
        rgb_norm = (x_rgb - self.imagenet_mean) / self.imagenet_std
        features = self.vit.forward_features(rgb_norm)  # (B, 197, 768)

        x_cls = features[:, 0]        # (B, 768)
        x_patch = features[:, 1:]     # (B, 196, 768) (kept for future use)

        # --------------------------------------------------------
        # Step 5: Cross-Attention Fusion
        # --------------------------------------------------------
        # Query: x_cls unsqueezed to (B, 1, 768)
        # Key/Value: stain_tokens (B, 16, 768)
        attn_out, _ = self.cross_attn(
            query=x_cls.unsqueeze(1),
            key=stain_tokens,
            value=stain_tokens,
        )  # (B, 1, 768)
        attn_out = attn_out.squeeze(1)  # (B, 768)

        # Gate
        concat = torch.cat([x_cls, attn_out], dim=-1)  # (B, 1536)
        gate = torch.sigmoid(self.gate_mlp(concat))    # (B, 768)
        fused_cls = x_cls + gate * attn_out            # (B, 768)

        # --------------------------------------------------------
        # Step 6: Ordinal Head
        # --------------------------------------------------------
        logits = self.ordinal_head(fused_cls)  # (B, 3)
        probs = cutpoints_to_probs(logits)     # (B, 4)

        return {
            "logits": logits,
            "probs": probs,
            "x_cls": x_cls,
            "fused_cls": fused_cls,
            "mean_gate": gate.mean(),
        }

    # ------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------
    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Returns named parameter groups for staged training.

        Returns:
            Dict[str, List[nn.Parameter]]:
                - vit: RGB ViT backbone
                - stain_encoder: StainEncoder
                - cross_fusion_gate: cross-attention + gate MLP
                - ordinal_head: ordinal head
        """
        return {
            "vit": list(self.vit.parameters()),
            "stain_encoder": list(self.stain_encoder.parameters()),
            "cross_fusion_gate": (
                list(self.cross_attn.parameters())
                + list(self.gate_mlp.parameters())
            ),
            "ordinal_head": list(self.ordinal_head.parameters()),
        }

    # ------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        """
        Counts parameters by component.

        Returns:
            Dict[str, int]: Parameter counts per component + total.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        counts = {
            "color_deconv": _count(self.color_deconv),
            "vit": _count(self.vit),
            "stain_encoder": _count(self.stain_encoder),
            "cross_attn": _count(self.cross_attn),
            "gate_mlp": _count(self.gate_mlp),
            "ordinal_head": _count(self.ordinal_head),
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts