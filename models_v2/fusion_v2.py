# ============================================================
# DSCA-ViT v2 — Fusion
# ============================================================
#
# New fusion components for DSCA-ViT v2:
#
#   BidirectionalInteraction : D->H and H->D cross-stream enrichment
#                              (zero-initialized residual outputs)
#   AdaptiveGate             : token- and channel-wise gate [B,197,768]
#
# Copied unchanged from the original DSCA-ViT:
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

        interaction_d_to_h : concat(H̃, D̃) -> ΔH   (Linear 1536->192 -> GELU -> Linear 192->768)
        interaction_h_to_d : concat(D̃, H̃) -> ΔD   (Linear 1536->192 -> GELU -> Linear 192->768)

    Then:

        H_e = H̃ + ΔH
        D_e = D̃ + ΔD

    The FINAL Linear layer of each MLP is zero-initialized, so at step 0:

        ΔH = 0
        ΔD = 0

    and therefore:

        H_e = H̃
        D_e = D̃

    This is deliberate: the new fusion residual starts as a no-op.
    """

    def __init__(self, embed_dim: int = 768, interaction_hidden_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.interaction_hidden_dim = interaction_hidden_dim

        # DAB -> H interaction: takes concat(H̃, D̃)
        self.interaction_d_to_h = nn.Sequential(
            nn.Linear(embed_dim * 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Linear(interaction_hidden_dim, embed_dim),
        )

        # H -> DAB interaction: takes concat(D̃, H̃)
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


class AdaptiveGate(nn.Module):
    """
    Token- and channel-wise adaptive fusion gate.

        gate_input = concat(H_e, D_e)          [B,197,1536]
        g = sigmoid(gate_mlp(gate_input))      [B,197,768]
        F = g * H_e + (1 - g) * D_e            [B,197,768]

    The gate is explicitly per-image, per-token, per-channel.
    It is NOT reduced to [B,768] or [B,197,1].

    The final gate Linear layer is zero-initialized, so:

        sigmoid(0) = 0.5

    and therefore G ≈ 0.5 at initialization (balanced start).
    """

    def __init__(self, embed_dim: int = 768, interaction_hidden_dim: int = 192) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.interaction_hidden_dim = interaction_hidden_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, interaction_hidden_dim),
            nn.GELU(),
            nn.Linear(interaction_hidden_dim, embed_dim),
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


# ============================================================
# Copied unchanged from models/fusion.py
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