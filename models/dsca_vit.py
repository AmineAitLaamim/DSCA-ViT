# ============================================================
# DSCA-ViT — Main Architecture Assembly
# ============================================================
#
# Dual-Stain Cross-Attention Vision Transformer
# for HER2 IHC Scoring
#
# Architecture:
#
#   RGB -> ColorDeconv -> [H, DAB]
#        -> 1->3 Projection (per stain)
#        -> Shared ViT Blocks 1-9
#        -> Bidirectional Cross-Attention (spatially-biased)
#        -> Shared ViT Blocks 10-12
#        -> Gated Token Fusion
#        -> Refinement Block
#        -> Classification Head
#        -> HER2 Score {0, 1+, 2+, 3+}
#
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn

from .color_deconv import ColorDeconvolution
from .shared_vit import SharedViTEncoder, StainChannelProjection
from .cross_attention import BidirectionalCrossAttention
from .fusion import GatedFusion, RefinementBlock, ClassificationHead


class DSCAViT(nn.Module):
    """
    Dual-Stain Cross-Attention Vision Transformer.

    A biologically-informed architecture for HER2 IHC scoring
    that explicitly separates Hematoxylin (morphology) and DAB
    (HER2 protein expression) stains before processing them
    through a shared Vision Transformer with spatially-biased
    bidirectional cross-attention.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 4 for HER2 scoring).
    pretrained : bool
        Whether to load ImageNet pretrained ViT-B/16 weights.
    split_after : int
        Insert cross-attention after this many ViT blocks (1-indexed).
        Default: 9.
    proj_init : str
        Initialization mode for the 1->3 channel projections.
        "repeat" or "xavier".
    spatial_bias_beta : float
        Self-correspondence bonus in the spatial bias matrix.
    spatial_bias_gamma : float
        Distance penalty scale in the spatial bias matrix.
    classifier_dropout : float
        Dropout rate in the classification head.

    Input
    -----
    x_rgb : torch.Tensor
        RGB images, shape (B, 3, H, W), UNNORMALIZED [0, 1] range.
        (from torchvision.transforms.ToTensor(), WITHOUT Normalize())

    Output
    ------
    logits : torch.Tensor
        Classification logits, shape (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        split_after: int = 9,
        proj_init: str = "repeat",
        spatial_bias_beta: float = 1.0,
        spatial_bias_gamma: float = 0.5,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------
        # 1. Color Deconvolution (fixed, no gradients)
        # --------------------------------------------------------

        self.color_deconv = ColorDeconvolution()

        # --------------------------------------------------------
        # 2. Learnable 1 -> 3 Channel Projections
        # --------------------------------------------------------

        self.proj_h = StainChannelProjection(init_mode=proj_init)
        self.proj_d = StainChannelProjection(init_mode=proj_init)

        # --------------------------------------------------------
        # 3. Shared ViT-B/16 Encoder
        # --------------------------------------------------------

        self.encoder = SharedViTEncoder(
            pretrained=pretrained,
            split_after=split_after,
        )

        # --------------------------------------------------------
        # 4. Bidirectional Cross-Attention
        # --------------------------------------------------------

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
            beta=spatial_bias_beta,
            gamma=spatial_bias_gamma,
        )

        # --------------------------------------------------------
        # 5. Gated Token Fusion
        # --------------------------------------------------------

        self.fusion = GatedFusion(
            embed_dim=self.encoder.embed_dim,
        )

        # --------------------------------------------------------
        # 6. Refinement Block
        # --------------------------------------------------------

        self.refinement = RefinementBlock(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
        )

        # --------------------------------------------------------
        # 7. Classification Head
        # --------------------------------------------------------

        self.classifier = ClassificationHead(
            embed_dim=self.encoder.embed_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        # --------------------------------------------------------
        # Store config
        # --------------------------------------------------------

        self.num_classes = num_classes
        self.split_after = split_after

    def forward(
        self,
        x_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Parameters
        ----------
        x_rgb : torch.Tensor
            RGB images, shape (B, 3, 224, 224), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Classification logits, shape (B, num_classes).
        """

        # --------------------------------------------------------
        # Step 1: Color Deconvolution (no gradients)
        # --------------------------------------------------------

        with torch.no_grad():
            h_channel, d_channel = self.color_deconv(x_rgb)

        # h_channel: (B, 1, 224, 224)
        # d_channel: (B, 1, 224, 224)

        # --------------------------------------------------------
        # Step 2: Channel Projection 1 -> 3
        # --------------------------------------------------------

        h_rgb = self.proj_h(h_channel)   # (B, 3, 224, 224)
        d_rgb = self.proj_d(d_channel)   # (B, 3, 224, 224)

        # --------------------------------------------------------
        # Step 3: Patch Embedding + Positional Encoding
        # --------------------------------------------------------

        h_tokens = self.encoder.embed(h_rgb)   # (B, 197, 768)
        d_tokens = self.encoder.embed(d_rgb)   # (B, 197, 768)

        # --------------------------------------------------------
        # Step 4: Shared Encoder Blocks 1..split_after
        #         (batched for GPU efficiency)
        # --------------------------------------------------------

        batch_size = h_tokens.shape[0]

        stacked = torch.cat(
            [h_tokens, d_tokens], dim=0
        )  # (2B, 197, 768)

        stacked = self.encoder.forward_before(stacked)

        h_tokens, d_tokens = stacked.split(
            batch_size, dim=0
        )  # each (B, 197, 768)

        # --------------------------------------------------------
        # Step 5: Bidirectional Cross-Attention
        # --------------------------------------------------------

        h_tokens, d_tokens = self.cross_attention(
            h_tokens, d_tokens
        )

        # --------------------------------------------------------
        # Step 6: Shared Encoder Blocks (split_after+1)..12
        #         (batched for GPU efficiency)
        # --------------------------------------------------------

        stacked = torch.cat(
            [h_tokens, d_tokens], dim=0
        )  # (2B, 197, 768)

        stacked = self.encoder.forward_after(stacked)

        h_final, d_final = stacked.split(
            batch_size, dim=0
        )  # each (B, 197, 768)

        # --------------------------------------------------------
        # Step 7: Gated Fusion
        # --------------------------------------------------------

        fused_tokens, gate_values = self.fusion(
            h_final, d_final
        )  # (B, 197, 768)

        # Store gate values for visualization
        self._last_gate_values = gate_values

        # --------------------------------------------------------
        # Step 8: Refinement Block
        # --------------------------------------------------------

        refined_tokens = self.refinement(fused_tokens)  # (B, 197, 768)

        # --------------------------------------------------------
        # Step 9: Classification
        # --------------------------------------------------------

        logits = self.classifier(refined_tokens)  # (B, num_classes)

        return logits

    def get_gate_values(self) -> torch.Tensor | None:
        """
        Returns the gate values from the last forward pass.

        Useful for interpretability / visualization.

        Returns
        -------
        torch.Tensor or None
            Gate values of shape (B, 196, 768), or None
            if no forward pass has been performed.
        """
        return getattr(self, "_last_gate_values", None)

    def get_parameter_groups(self) -> dict:
        """
        Returns parameter groups for discriminative learning rates.

        Returns
        -------
        dict
            Dictionary with keys:
            - "encoder": shared ViT encoder parameters
            - "new": all new components (projections, cross-attn,
                     fusion, refinement, classifier)
        """

        encoder_params = list(self.encoder.parameters())

        new_params = (
            list(self.proj_h.parameters())
            + list(self.proj_d.parameters())
            + list(self.cross_attention.parameters())
            + list(self.fusion.parameters())
            + list(self.refinement.parameters())
            + list(self.classifier.parameters())
        )

        return {
            "encoder": encoder_params,
            "new": new_params,
        }

    def count_parameters(self) -> dict:
        """
        Count parameters by component.

        Returns
        -------
        dict
            Parameter counts for each component.
        """

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        counts = {
            "color_deconv": _count(self.color_deconv),
            "proj_h": _count(self.proj_h),
            "proj_d": _count(self.proj_d),
            "encoder": _count(self.encoder),
            "cross_attention": _count(self.cross_attention),
            "fusion": _count(self.fusion),
            "refinement": _count(self.refinement),
            "classifier": _count(self.classifier),
        }

        counts["total"] = sum(counts.values())

        counts["trainable"] = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return counts
