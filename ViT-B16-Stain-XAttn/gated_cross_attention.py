# ============================================================
# ViT-B16-Stain-XAttn — Gated Cross-Attention Block
# ============================================================
#
# A single header-light cross-attention block that lets RGB tokens
# (query) pull spatial information from stain tokens (keys/values).
# The learned gate `alpha` is initialized to EXACTLY 0.0, so at
# initialization the returned delta is 0 and the model is numerically
# identical to the plain ViT-B16 baseline (zero-contribution-at-init).
#
# This module returns ONLY the gated delta (not the full residual);
# the caller is responsible for adding it to the RGB residual stream.
#
# Inspiration: Flamingo's gated cross-attention (zero-init gate so a
# freshly initialized fusion path cannot corrupt a working backbone
# at step zero; gate and weights then co-adapt during training).
#
# No network calls.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross-attention from RGB (query) to stain (keys/values).

    Args:
        embed_dim (int): Model dimension (768 for ViT-B16).
        num_heads (int): Number of attention heads (12 default).
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm_ca = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # THE critical zero-init gate parameter.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_tokens: torch.Tensor, stain_tokens: torch.Tensor) -> torch.Tensor:
        # rgb_tokens: [B, N, D]; stain_tokens: [B, M, D]
        B, N, D = rgb_tokens.shape
        M = stain_tokens.shape[1]

        x = self.norm_ca(rgb_tokens)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(stain_tokens).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(stain_tokens).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        gate = torch.tanh(self.alpha)
        return gate * attn_out  # gated DELTA only; caller adds residual