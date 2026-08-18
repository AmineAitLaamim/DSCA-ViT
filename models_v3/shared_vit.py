# ============================================================
# Shared ViT Encoder
# ============================================================
#
# Wraps a pretrained timm ViT-B/16 for dual-stream processing.
#
# The encoder is split at a configurable layer so that:
#   - Blocks 1..split_after  are run BEFORE cross-attention
#   - Blocks split_after+1..12 are run AFTER cross-attention
#
# Both stain streams share the SAME encoder weights (100%).
#
# In v3 the SAME encoder instance is also reused for the fine and
# coarse multi-resolution views (no second ViT, no duplicated params).
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class StainChannelProjection(nn.Module):
    """
    Learnable 1 -> 3 channel projection.

    Converts a single-channel stain image to a pseudo-RGB image
    so that the pretrained ViT patch embedding (which expects 3 channels)
    can be used without modification.

    Parameters
    ----------
    init_mode : str
        How to initialize the 1x1 convolution weights.
        - "repeat" : initialize so output ≈ [x, x, x] (identity-like)
        - "xavier" : Xavier uniform initialization
    """

    def __init__(self, init_mode: str = "repeat") -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # --------------------------------------------------------
        # Initialization
        # --------------------------------------------------------

        if init_mode == "repeat":
            # Initialize so that output ≈ [x, x, x]
            nn.init.ones_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif init_mode == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(
                f"Unknown init_mode: {init_mode}. "
                f"Use 'repeat' or 'xavier'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Single-channel image, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Pseudo-RGB image, shape (B, 3, H, W).
        """
        return self.proj(x)


class SharedViTEncoder(nn.Module):
    """
    Shared ViT-B/16 encoder with split-point for cross-attention.

    Loads a pretrained ViT-B/16 from timm and exposes:
      - patch_embed   : the patch embedding layer
      - cls_token     : the CLS token
      - pos_embed     : positional embeddings
      - blocks_before : transformer blocks 0..split_after-1
      - blocks_after  : transformer blocks split_after..11
      - norm          : final LayerNorm

    The classification head is removed (we use our own).

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    split_after : int
        Insert cross-attention after this many blocks (1-indexed).
        Default: 9 (blocks 1-9 before, blocks 10-12 after).
    """

    def __init__(
        self,
        pretrained: bool = True,
        split_after: int = 9,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------
        # Load pretrained ViT-B/16
        # --------------------------------------------------------

        vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # --------------------------------------------------------
        # Extract components
        # --------------------------------------------------------

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token        # (1, 1, 768)
        self.pos_embed = vit.pos_embed        # (1, 197, 768)
        self.pos_drop = vit.pos_drop

        # --------------------------------------------------------
        # Split transformer blocks
        # --------------------------------------------------------

        all_blocks = list(vit.blocks)

        if not (1 <= split_after <= len(all_blocks)):
            raise ValueError(
                f"split_after must be in [1, {len(all_blocks)}], "
                f"got {split_after}."
            )

        self.blocks_before = nn.Sequential(*all_blocks[:split_after])
        self.blocks_after = nn.Sequential(*all_blocks[split_after:])

        self.norm = vit.norm

        # --------------------------------------------------------
        # Store config
        # --------------------------------------------------------

        self.embed_dim = vit.embed_dim     # 768
        self.num_tokens = 197              # 1 CLS + 196 patches
        self.split_after = split_after

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patch embedding + CLS token + positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input image, shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Token sequence, shape (B, 197, 768).
        """

        # Patch embedding: (B, 3, 224, 224) -> (B, 196, 768)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)

        # Add positional encoding
        x = x + self.pos_embed

        x = self.pos_drop(x)

        return x

    def forward_before(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run blocks BEFORE cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            Token sequence, shape (B, 197, 768).

        Returns
        -------
        torch.Tensor
            Processed tokens, shape (B, 197, 768).
        """
        return self.blocks_before(x)

    def forward_after(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run blocks AFTER cross-attention, including final LayerNorm.

        Parameters
        ----------
        x : torch.Tensor
            Token sequence, shape (B, 197, 768).

        Returns
        -------
        torch.Tensor
            Final tokens, shape (B, 197, 768).
        """
        x = self.blocks_after(x)
        x = self.norm(x)
        return x