# ============================================================
# DSCA-ViT v3 - Fusion
# ============================================================
#
# Fusion components for DSCA-ViT v3:
#
#   BidirectionalInteraction : D->H and H->D cross-stream enrichment
#                              (zero-initialized residual outputs).
#                              Copied unchanged from v2.
#   StainGate                : token- and channel-wise H/DAB gate
#                              [B,197,768] (same design as v2 AdaptiveGate).
#   ScaleGate                : token- and channel-wise fine/coarse gate
#                              [B,197,768] (NEW in v3).
#
# Copied unchanged from the original DSCA-ViT / v2:
#
#   RefinementBlock
#   ClassificationHead
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalInteraction(nn.Module):
    """
    Explicit bidirectional H <-> DAB interaction.

    Two completely independent MLPs:

        interaction_d_to_h : concat(H_tilde, D_tilde) -> delta_H
                             (Linear 1536->192 -> GELU -> Linear 192->768)
        interaction_h_to_d : concat(D_tilde, H_tilde) -> delta_D
                             (Linear 1536->192 -> GELU -> Linear 192->768)

    Then:

        H_e = H_tilde + delta_H
        D_e = D_tilde + delta_D

    The FINAL Linear layer of each MLP is zero-initialized, so at step 0:

        delta_H = 0
        delta_D = 0

    and therefore:

        H_e = H_tilde
        D_e = D_tilde

    This is deliberate: the new fusion residual starts as a no-op and
    the pretrained representation is preserved at initialization.

    In v3 a SINGLE shared instance is reused independently for each
    scale (fine and coarse).
    """

    def __init__(self, embed_dim: int = 768, interaction_hidden_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.interaction_hidden_dim = interaction_hidden_dim

        # DAB -> H interaction: takes concat(H_tilde, D_tilde)
        self.interaction_d_to_h = nn.Sequential(
            nn.Linear(embed_dim * 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Linear(interaction_hidden_dim, embed_dim),
        )

        # H -> DAB interaction: takes concat(D_tilde, H_tilde)
        self.interaction_h_to_d = nn.Sequential(
            nn.Linear(embed_dim * 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Linear(interaction_hidden_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-initialize the final Linear layer of each interaction MLP
        nn.init.zeros_(self.interaction_d_to_h[-1].weight)
        nn.init.zeros_(self.interaction_d_to_h[-1].bias)

        nn.init.zeros_(self.interaction_h_to_d[-1].weight)
        nn.init.zeros_(self.interaction_h_to_d[-1].bias)

    def forward(
        self,
        h_tilde: torch.Tensor,
        d_tilde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_tilde: H tokens after cross-attention, shape (B, N, C).
            d_tilde: DAB tokens after cross-attention, shape (B, N, C).

        Returns:
            Tuple of (h_enriched, d_enriched), each (B, N, C).
        """
        delta_h = self.interaction_d_to_h(torch.cat([h_tilde, d_tilde], dim=-1))
        delta_d = self.interaction_h_to_d(torch.cat([d_tilde, h_tilde], dim=-1))

        h_enriched = h_tilde + delta_h
        d_enriched = d_tilde + delta_d

        return h_enriched, d_enriched


class StainGate(nn.Module):
    """
    Stain-level adaptive H/DAB fusion gate (v3 naming of the v2 AdaptiveGate).

        gate_input = concat(H_e, D_e)          [B,197,1536]
        g_stain = sigmoid(gate_mlp(gate_input)) [B,197,768]
        F_stain = g_stain * H_e + (1 - g_stain) * D_e  [B,197,768]

    The gate is explicitly per-image, per-token, per-channel.
    It is NOT reduced to [B,768] or [B,197,1].

    The final gate Linear layer is zero-initialized, so:

        sigmoid(0) = 0.5

    and therefore G_stain ~= 0.5 at initialization (balanced start).

    In v3 a SINGLE shared instance is reused independently for each
    scale (fine and coarse).
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-initialize the final gate Linear layer -> sigmoid(0) = 0.5
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.zeros_(self.gate_mlp[2].bias)

    def forward(
        self,
        h_enriched: torch.Tensor,
        d_enriched: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_enriched: H tokens after bidirectional interaction, shape (B, N, C).
            d_enriched: DAB tokens after bidirectional interaction, shape (B, N, C).

        Returns:
            Tuple of (fused, gate_values):
                fused       : (B, N, C)
                gate_values : (B, N, C)  (before fusion, for telemetry)
        """
        gate_input = torch.cat([h_enriched, d_enriched], dim=-1)  # (B, N, 2C)
        gate_values = self.gate_mlp(gate_input)                   # (B, N, C)
        fused = gate_values * h_enriched + (1.0 - gate_values) * d_enriched

        return fused, gate_values


class ScaleGate(nn.Module):
    """
    Scale-adaptive fine/coarse fusion gate (NEW in v3).

        scale_input = concat(F_fine, F_coarse)      [B,197,1536]
        g_scale = sigmoid(scale_mlp(scale_input))   [B,197,768]
        F = g_scale * F_fine + (1 - g_scale) * F_coarse  [B,197,768]

    This answers, for each token/channel:

        "Should this token/channel rely more on fine cellular
         information or coarse contextual information?"

    The final scale-gate Linear layer is zero-initialized, so:

        sigmoid(0) = 0.5

    and therefore G_scale ~= 0.5 at initialization (balanced
    fine/coarse fusion; the gate does not strongly prefer either scale).
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.scale_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-initialize the final gate Linear layer -> sigmoid(0) = 0.5
        nn.init.zeros_(self.scale_mlp[2].weight)
        nn.init.zeros_(self.scale_mlp[2].bias)

    def forward(
        self,
        f_fine: torch.Tensor,
        f_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_fine: Fine-scale fused representation, shape (B, N, C).
            f_coarse: Coarse-scale fused representation, shape (B, N, C).

        Returns:
            Tuple of (fused, gate_values):
                fused       : (B, N, C)
                gate_values : (B, N, C)  (before fusion, for telemetry)
        """
        scale_input = torch.cat([f_fine, f_coarse], dim=-1)   # (B, N, 2C)
        gate_values = self.scale_mlp(scale_input)             # (B, N, C)
        fused = gate_values * f_fine + (1.0 - gate_values) * f_coarse

        return fused, gate_values


# ============================================================
# Copied unchanged from v2 / models/fusion.py
# ============================================================


class RefinementBlock(nn.Module):
    """
    Refinement Block: A single standard transformer block (self-attention + FFN).

    Args:
        embed_dim (int): Embedding dimension. Default is 768.
        num_heads (int): Number of attention heads. Default is 12.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        dropout (float): Dropout rate. Default is 0.0.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention and FFN.

        Args:
            x (torch.Tensor): Input tokens of shape (B, N, embed_dim)

        Returns:
            torch.Tensor: Refined tokens of shape (B, N, embed_dim)
        """
        # Pre-norm style
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class ClassificationHead(nn.Module):
    """
    Classification Head for DSCA-ViT.
    Extracts CLS token and global average pooling (GAP) of patch tokens, concatenates them,
    and applies a classifier to predict class logits.

    Args:
        num_classes (int): Number of classes for classification.
        embed_dim (int): Embedding dimension. Default is 768.
        dropout (float): Dropout rate in the classifier. Default is 0.1.
    """

    def __init__(self, num_classes: int, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extracts features and computes classification logits.

        Args:
            tokens (torch.Tensor): Refined tokens of shape (B, N, embed_dim)

        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        cls_token = tokens[:, 0, :]  # (B, embed_dim)
        gap = tokens[:, 1:, :].mean(dim=1)  # (B, embed_dim)

        z = torch.cat([cls_token, gap], dim=-1)  # (B, embed_dim * 2)
        z = self.norm(z)
        logits = self.classifier(z)  # (B, num_classes)
        return logits