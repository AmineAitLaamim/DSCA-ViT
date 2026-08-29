# ============================================================
# ViT-B16-Stain-XAttn -- Main Model
# ============================================================
# ViT-B16 fine-tuned on RGB, optionally pulling spatial info from
# H and DAB stain maps via GATED cross-attention in the last blocks
# (cross_attn_layers, default [8,9,10,11]).
#
# MOST IMPORTANT PROPERTY: at init the model is NUMERICALLY IDENTICAL
# to the plain ViT-B16 baseline because each gate alpha=0 -> tanh(0)=0,
# so the stain pathway adds an exact-zero delta and can only ADD signal
# during training.
#
# Dataloader outputs RAW RGB [0,1] (NO ImageNet Normalize) because color
# deconvolution needs real intensities. RGB normalization happens inside
# forward(). H/DAB use FIXED GLOBAL normalization (never per-instance).
#
# Reimplements the backbone block loop per timm 1.0.28's verified Block
# structure; never calls backbone.forward().
# ============================================================

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Dict, List

import timm
import torch
import torch.nn as nn

import color_deconv
from gated_cross_attention import GatedCrossAttentionBlock

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTB16StainXAttn(nn.Module):
    """ViT-B16 + gated cross-attention from stain tokens into last blocks."""

    def __init__(self, pretrained=True, num_classes=4, cross_attn_layers=None,
                 num_heads=12, stain_stats=None, verbose=False):
        super().__init__()
        if stain_stats is None:
            raise ValueError("stain_stats required for ViTB16StainXAttn.")
        if cross_attn_layers is None:
            cross_attn_layers = [8, 9, 10, 11]
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.embed_dim
        self.num_heads = num_heads
        self.cross_attn_layers = sorted(set(int(i) for i in cross_attn_layers))

        # Two independent stain patch embeddings (H and DAB).
        self.h_patch_embed = nn.Conv2d(1, self.embed_dim, 16, stride=16)
        self.dab_patch_embed = nn.Conv2d(1, self.embed_dim, 16, stride=16)

        # Stain pos embedding: independent copy of backbone patch pos_embed.
        self.stain_pos_embed = nn.Parameter(
            self.backbone.pos_embed[:, 1:, :].clone())

        # Two modality-type embeddings (H vs DAB), not three.
        self.h_type_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dab_type_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.cross_attn_modules = nn.ModuleDict({
            str(i): GatedCrossAttentionBlock(self.embed_dim, self.num_heads)
            for i in self.cross_attn_layers
        })

        for k in ("h_mean", "h_std", "dab_mean", "dab_std"):
            if k not in stain_stats:
                raise ValueError(f"Stain stats missing key '{k}'.")
        self.register_buffer("h_mean", torch.tensor(float(stain_stats["h_mean"])))
        self.register_buffer("h_std", torch.tensor(float(stain_stats["h_std"])))
        self.register_buffer("dab_mean", torch.tensor(float(stain_stats["dab_mean"])))
        self.register_buffer("dab_std", torch.tensor(float(stain_stats["dab_std"])))

        self.color_deconv = color_deconv.ColorDeconvolution()
        self.head = nn.Linear(self.embed_dim, num_classes)

        if verbose:
            print("  [ViT-B16-Stain-XAttn] built; gate alpha=0 at init.")

    def forward(self, x_raw):
        B = x_raw.shape[0]
        dev = x_raw.device

        # RGB path: normalize internally (ImageNet).
        mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        x_rgb_norm = (x_raw - mean) / std

        # Stain: deconv on RAW x, then fixed global normalize.
        h, dab = self.color_deconv(x_raw)
        h_norm = (h - self.h_mean.to(dev)) / self.h_std.to(dev)
        dab_norm = (dab - self.dab_mean.to(dev)) / self.dab_std.to(dev)

        # RGB tokens.
        rgb_patches = self.backbone.patch_embed(x_rgb_norm)  # [B,196,768]
        cls_tok = self.backbone.cls_token.expand(B, -1, -1)
        rgb_tokens = torch.cat([cls_tok, rgb_patches], dim=1)  # [B,197,768]
        rgb_tokens = rgb_tokens + self.backbone.pos_embed
        if hasattr(self.backbone, "pos_drop"):
            rgb_tokens = self.backbone.pos_drop(rgb_tokens)

        # Stain tokens (H first, then DAB; fixed order).
        h_tokens = self.h_patch_embed(h_norm).flatten(2).transpose(1, 2)
        h_tokens = h_tokens + self.stain_pos_embed + self.h_type_embed
        dab_tokens = self.dab_patch_embed(dab_norm).flatten(2).transpose(1, 2)
        dab_tokens = dab_tokens + self.stain_pos_embed + self.dab_type_embed
        stain_tokens = torch.cat([h_tokens, dab_tokens], dim=1)  # [B,392,768]

        # Custom block loop (timm 1.0.28 Block structure).
        for i, block in enumerate(self.backbone.blocks):
            xa = block.drop_path1(block.ls1(block.attn(block.norm1(rgb_tokens))))
            rgb_tokens = rgb_tokens + xa
            if str(i) in self.cross_attn_modules:
                delta = self.cross_attn_modules[str(i)](rgb_tokens, stain_tokens)
                rgb_tokens = rgb_tokens + delta
            rgb_tokens = rgb_tokens + block.drop_path2(
                block.ls2(block.mlp(block.norm2(rgb_tokens))))

        rgb_tokens = self.backbone.norm(rgb_tokens)
        cls_out = rgb_tokens[:, 0]
        logits = self.head(cls_out)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}

    def pretrained_backbone_parameters(self):
        return list(self.backbone.parameters())

    def new_component_parameters(self):
        params = []
        params += list(self.head.parameters())
        params += list(self.h_patch_embed.parameters())
        params += list(self.dab_patch_embed.parameters())
        params += [self.stain_pos_embed, self.h_type_embed, self.dab_type_embed]
        params += list(self.cross_attn_modules.parameters())
        return params

    def freeze_backbone(self):
        for p in self.pretrained_backbone_parameters():
            p.requires_grad = False
        for p in self.new_component_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_parameters(self):
        bb = sum(p.numel() for p in self.pretrained_backbone_parameters())
        nc = sum(p.numel() for p in self.new_component_parameters())
        return {
            "pretrained_backbone": bb,
            "new_components": nc,
            "total": bb + nc,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
