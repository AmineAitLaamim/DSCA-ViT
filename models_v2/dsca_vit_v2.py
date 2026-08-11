# ============================================================
# DSCA-ViT v2 — Main Architecture Assembly
# ============================================================
#
# Dual-Stain Cross-Attention Vision Transformer v2
# for HER2 IHC Scoring
#
# Architecture:
#
#   RGB -> Fixed ColorDeconv (no_grad) -> [H, DAB]
#        -> StainNorm_H / StainNorm_DAB          (GroupNorm(1,1), independent)
#        -> StainAdapter_H / StainAdapter_DAB    (1->32->3, independent)
#        -> LearnableChannelAffine_H / _DAB      (per-channel scale/bias, independent)
#        -> Shared ViT (embed + blocks 1-9)
#        -> Bidirectional Cross-Attention (spatially-biased)
#        -> Shared ViT (blocks 10-12)
#        -> BidirectionalInteraction (D->H, H->D, zero-init residuals)
#        -> AdaptiveGate (token/channel-wise [B,197,768])
#        -> RefinementBlock
#        -> ClassificationHead
#        -> HER2 Score {0, 1+, 2+, 3+}
#
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn

from .color_deconv import ColorDeconvolution
from .shared_vit import SharedViTEncoder
from .cross_attention import BidirectionalCrossAttention
from .input_adapters import StainNorm1ch, StainAdapter, LearnableChannelAffine
from .fusion_v2 import (
    BidirectionalInteraction,
    AdaptiveGate,
    RefinementBlock,
    ClassificationHead,
)


class DSCAViTv2(nn.Module):
    """
    Dual-Stain Cross-Attention Vision Transformer v2.

    The v2 architecture improves the weak interfaces around the existing
    DSCA architecture:

        stain representation
            -> ViT-compatible representation
            -> H/DAB interaction
            -> fusion

    The ViT backbone, cross-attention, spatial bias, refinement, and
    classifier are preserved unchanged from the original DSCA-ViT.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        split_after: int = 9,
        hidden_channels: int = 32,
        interaction_hidden_dim: int = 192,
        adapter_final_scale: float = 0.1,
        spatial_bias_beta: float = 1.0,
        spatial_bias_gamma: float = 0.1,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------
        # 1. Fixed Color Deconvolution (no gradients)
        # --------------------------------------------------------

        self.color_deconv = ColorDeconvolution()

        # --------------------------------------------------------
        # 2. New input-side components (independent per stream)
        # --------------------------------------------------------

        self.norm_h = StainNorm1ch()
        self.norm_d = StainNorm1ch()

        self.adapter_h = StainAdapter(
            hidden_channels=hidden_channels,
            adapter_final_scale=adapter_final_scale,
        )
        self.adapter_d = StainAdapter(
            hidden_channels=hidden_channels,
            adapter_final_scale=adapter_final_scale,
        )

        self.channel_affine_h = LearnableChannelAffine()
        self.channel_affine_d = LearnableChannelAffine()

        # --------------------------------------------------------
        # 3. Shared ViT-B/16 Encoder (preserved unchanged)
        # --------------------------------------------------------

        self.encoder = SharedViTEncoder(
            pretrained=pretrained,
            split_after=split_after,
        )

        # --------------------------------------------------------
        # 4. Bidirectional Cross-Attention (preserved unchanged)
        # --------------------------------------------------------

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
            beta=spatial_bias_beta,
            gamma=spatial_bias_gamma,
        )

        # --------------------------------------------------------
        # 5. New fusion: bidirectional interaction + adaptive gate
        # --------------------------------------------------------

        self.interaction = BidirectionalInteraction(
            embed_dim=self.encoder.embed_dim,
            interaction_hidden_dim=interaction_hidden_dim,
        )
        self.gate = AdaptiveGate(
            embed_dim=self.encoder.embed_dim,
            interaction_hidden_dim=interaction_hidden_dim,
        )

        # --------------------------------------------------------
        # 6. Refinement Block (preserved unchanged)
        # --------------------------------------------------------

        self.refinement = RefinementBlock(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
        )

        # --------------------------------------------------------
        # 7. Classification Head (preserved unchanged)
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
        self.hidden_channels = hidden_channels
        self.interaction_hidden_dim = interaction_hidden_dim
        self.adapter_final_scale = adapter_final_scale

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
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
        # Step 1: Fixed Color Deconvolution (no gradients)
        # --------------------------------------------------------

        with torch.no_grad():
            h_channel, d_channel = self.color_deconv(x_rgb)

        # h_channel: (B, 1, 224, 224)
        # d_channel: (B, 1, 224, 224)

        # --------------------------------------------------------
        # Step 2: New input-side processing (independent per stream)
        # --------------------------------------------------------

        h = self.norm_h(h_channel)
        d = self.norm_d(d_channel)

        h = self.adapter_h(h)   # (B, 3, 224, 224)
        d = self.adapter_d(d)   # (B, 3, 224, 224)

        h = self.channel_affine_h(h)
        d = self.channel_affine_d(d)

        # --------------------------------------------------------
        # Step 3: Patch Embedding + Positional Encoding
        # --------------------------------------------------------

        h_tokens = self.encoder.embed(h)   # (B, 197, 768)
        d_tokens = self.encoder.embed(d)   # (B, 197, 768)

        # --------------------------------------------------------
        # Step 4: Shared Encoder Blocks 1..split_after
        #         (batched for GPU efficiency)
        # --------------------------------------------------------

        batch_size = h_tokens.shape[0]

        stacked = torch.cat([h_tokens, d_tokens], dim=0)  # (2B, 197, 768)
        stacked = self.encoder.forward_before(stacked)
        h_tokens, d_tokens = stacked.split(batch_size, dim=0)

        # --------------------------------------------------------
        # Step 5: Bidirectional Cross-Attention
        # --------------------------------------------------------

        h_tokens, d_tokens = self.cross_attention(h_tokens, d_tokens)

        # --------------------------------------------------------
        # Step 6: Shared Encoder Blocks (split_after+1)..12
        # --------------------------------------------------------

        stacked = torch.cat([h_tokens, d_tokens], dim=0)  # (2B, 197, 768)
        stacked = self.encoder.forward_after(stacked)
        h_final, d_final = stacked.split(batch_size, dim=0)

        # --------------------------------------------------------
        # Step 7: Bidirectional Interaction
        # --------------------------------------------------------

        h_enriched, d_enriched = self.interaction(h_final, d_final)

        # --------------------------------------------------------
        # Step 8: Adaptive Gate
        # --------------------------------------------------------

        fused_tokens, gate_values = self.gate(h_enriched, d_enriched)

        # Store gate values for telemetry
        self._last_gate_values = gate_values

        # --------------------------------------------------------
        # Step 9: Refinement Block
        # --------------------------------------------------------

        refined_tokens = self.refinement(fused_tokens)

        # --------------------------------------------------------
        # Step 10: Classification
        # --------------------------------------------------------

        logits = self.classifier(refined_tokens)

        return logits

    def get_gate_values(self) -> torch.Tensor | None:
        """
        Returns the gate values from the last forward pass.

        Returns
        -------
        torch.Tensor or None
            Gate values of shape (B, 197, 768), or None
            if no forward pass has been performed.
        """
        return getattr(self, "_last_gate_values", None)

    def get_parameter_groups(self) -> dict:
        """
        Returns exactly 5 parameter groups:

            vit             : shared ViT encoder
            existing_dsca   : cross-attention (incl. spatial bias) + refinement
            input_modules   : norm_h, norm_d, adapter_h, adapter_d,
                              channel_affine_h, channel_affine_d
            fusion_modules  : interaction_d_to_h, interaction_h_to_d, gate_mlp
            classifier      : classification head

        Verifies that no parameter appears twice and that every trainable
        parameter belongs to exactly one group. Raises an error otherwise.
        """

        vit = list(self.encoder.parameters())

        existing_dsca = (
            list(self.cross_attention.parameters())
            + list(self.refinement.parameters())
        )

        input_modules = (
            list(self.norm_h.parameters())
            + list(self.norm_d.parameters())
            + list(self.adapter_h.parameters())
            + list(self.adapter_d.parameters())
            + list(self.channel_affine_h.parameters())
            + list(self.channel_affine_d.parameters())
        )

        fusion_modules = (
            list(self.interaction.parameters())
            + list(self.gate.parameters())
        )

        classifier = list(self.classifier.parameters())

        groups = {
            "vit": vit,
            "existing_dsca": existing_dsca,
            "input_modules": input_modules,
            "fusion_modules": fusion_modules,
            "classifier": classifier,
        }

        # --------------------------------------------------------
        # Parameter-group validation
        # --------------------------------------------------------

        seen: set[int] = set()
        for name, params in groups.items():
            for p in params:
                pid = id(p)
                if pid in seen:
                    raise RuntimeError(
                        f"Parameter appears in multiple groups: {name} "
                        f"(param id={pid})."
                    )
                seen.add(pid)

        all_model_params = [p for p in self.parameters() if p.requires_grad]
        if len(seen) != len(all_model_params):
            raise RuntimeError(
                "Parameter-group validation failed: "
                f"{len(seen)} params in groups vs "
                f"{len(all_model_params)} trainable model params."
            )

        return groups

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
            "norm_h": _count(self.norm_h),
            "norm_d": _count(self.norm_d),
            "adapter_h": _count(self.adapter_h),
            "adapter_d": _count(self.adapter_d),
            "channel_affine_h": _count(self.channel_affine_h),
            "channel_affine_d": _count(self.channel_affine_d),
            "encoder": _count(self.encoder),
            "cross_attention": _count(self.cross_attention),
            "interaction": _count(self.interaction),
            "gate": _count(self.gate),
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

    def load_original_weights(self, checkpoint_path: str, device: torch.device) -> dict:
        """
        Loads weights from the original DSCA-ViT checkpoint into v2.

        Preserved modules must receive all their weights:
            encoder.*, cross_attention.*, refinement.*, classifier.*

        New modules are expected to be missing (fresh):
            norm_h.*, norm_d.*, adapter_h.*, adapter_d.*,
            channel_affine_h.*, channel_affine_d.*,
            interaction.*, gate.*

        Legacy modules are ignored:
            proj_h.*, proj_d.*, fusion.*

        Raises an exception if any preserved module is missing weights.
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        model_state = self.state_dict()

        # --------------------------------------------------------
        # Classify keys
        # --------------------------------------------------------

        preserved_prefixes = ["encoder.", "cross_attention.", "refinement.", "classifier."]
        new_prefixes = [
            "norm_h.", "norm_d.",
            "adapter_h.", "adapter_d.",
            "channel_affine_h.", "channel_affine_d.",
            "interaction.", "gate.",
        ]
        legacy_prefixes = ["proj_h.", "proj_d.", "fusion."]

        loaded_keys = []
        missing_preserved = []
        expected_new = []
        legacy_ignored = []

        for key in state_dict:
            if any(key.startswith(p) for p in preserved_prefixes):
                loaded_keys.append(key)
            elif any(key.startswith(p) for p in new_prefixes):
                expected_new.append(key)
            elif any(key.startswith(p) for p in legacy_prefixes):
                legacy_ignored.append(key)
            # else: unexpected key — ignore silently (not in any category)

        # Check preserved modules for missing keys
        for key in model_state:
            if any(key.startswith(p) for p in preserved_prefixes):
                if key not in state_dict:
                    missing_preserved.append(key)

        if missing_preserved:
            raise RuntimeError(
                "Preserved modules are missing weights from the original "
                f"checkpoint:\n{missing_preserved}"
            )

        # --------------------------------------------------------
        # Load
        # --------------------------------------------------------

        load_result = self.load_state_dict(state_dict, strict=False)

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        print("=" * 60)
        print("V2 COMPATIBILITY LOAD")
        print("=" * 60)

        print("\nLoaded old parameters:")
        for prefix in preserved_prefixes:
            n = sum(1 for k in loaded_keys if k.startswith(prefix))
            print(f"  {prefix:<22} ✓ ({n} tensors)")

        print("\nExpected new parameters (fresh):")
        for prefix in new_prefixes:
            n = sum(1 for k in expected_new if k.startswith(prefix))
            print(f"  {prefix:<22} fresh ({n} tensors)")

        print("\nLegacy parameters ignored:")
        for prefix in legacy_prefixes:
            n = sum(1 for k in legacy_ignored if k.startswith(prefix))
            print(f"  {prefix:<22} ignored ({n} tensors)")

        print("\nPreserved parameters missing:")
        print("  NONE" if not missing_preserved else f"  {missing_preserved}")

        print("=" * 60)

        return {
            "loaded_keys": loaded_keys,
            "expected_new": expected_new,
            "legacy_ignored": legacy_ignored,
            "missing_preserved": missing_preserved,
            "load_result": load_result,
        }