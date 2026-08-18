# ============================================================
# DSCA-ViT v3 - Main Architecture Assembly
# ============================================================
#
# Dual-Stain Cross-Attention Vision Transformer v3
# for HER2 IHC Scoring
#
# Architecture:
#
#   RGB -> [training-only StainAugmentation is applied in the
#           dataset pipeline, NOT inside the model]
#        -> Fixed ColorDeconv (no_grad) -> [H, DAB]
#        -> StainNorm_H / StainNorm_DAB          (GroupNorm(1,1), independent)
#        -> StainAdapter_H / StainAdapter_DAB    (1->32->3, independent)
#        -> LearnableChannelAffine_H / _DAB      (per-channel scale/bias, independent)
#        -> Fine view (224x224)  /  Coarse view (224->112->224, low-frequency)
#        -> Shared ViT (embed + blocks 1-9) per view independently
#        -> Existing Bidirectional Cross-Attention (spatially-biased) per view
#        -> Shared ViT (blocks 10-12) per view independently
#        -> Shared BidirectionalInteraction (zero-init residuals) per view
#        -> Shared StainGate (H/DAB, ~0.5 init) per view
#        -> ScaleGate (fine/coarse, ~0.5 init)
#        -> Existing RefinementBlock
#        -> Existing ClassificationHead
#        -> HER2 Score {0, 1+, 2+, 3+}
#
# SHARED-MODULE RULE (locked):
#   self.encoder, self.cross_attention, self.interaction and
#   self.stain_gate are SINGLE instances reused independently for
#   the fine and the coarse branch. There are NO duplicate module
#   attributes (no fine_vit / coarse_vit, etc.).
#
# ============================================================

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .color_deconv import ColorDeconvolution
from .shared_vit import SharedViTEncoder
from .cross_attention import BidirectionalCrossAttention
from .input_adapters_v3 import StainNorm1ch, StainAdapter, LearnableChannelAffine
from .multiscale_v3 import CoarseScaleView
from .fusion_v3 import (
    BidirectionalInteraction,
    StainGate,
    ScaleGate,
    RefinementBlock,
    ClassificationHead,
)


class DSCAViTv3(nn.Module):
    """
    Dual-Stain Cross-Attention Vision Transformer v3.

    v3 builds on v2 and adds two architectural ingredients (plus a
    training-only stain-domain augmentation implemented in the dataset
    pipeline):

        1. Multi-scale representation:
           - fine view   : 224x224 (cellular morphology, membrane staining)
           - coarse view : 224 -> 112 -> 224 bilinear downsample/upsample
                           (a low-frequency / coarse multi-resolution view
                           of the SAME 224x224 field - NOT a larger spatial
                           context, NOT a change of magnification)
        2. Scale-adaptive gating / hierarchical fusion:
           - StainGate  (per scale): token/channel-wise H vs DAB choice
           - ScaleGate  (after both scales): token/channel-wise fine vs
             coarse choice

    The shared pretrained ViT, cross-attention, interaction and stain
    gate are applied to each scale INDEPENDENTLY (weights shared,
    computation separate). The new modules break the v2 symmetry issue
    in a controlled way and are zero-initialized / ~0.5-gated so the
    pretrained representation is preserved at initialization.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        split_after: int = 9,
        hidden_channels: int = 32,
        interaction_hidden_dim: int = 192,
        adapter_final_scale: float = 0.1,
        coarse_size: int = 112,
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
        # 2. Input-side components (independent per stream)
        # --------------------------------------------------------

        self.norm_h = StainNorm1ch()
        self.norm_dab = StainNorm1ch()

        self.adapter_h = StainAdapter(
            hidden_channels=hidden_channels,
            adapter_final_scale=adapter_final_scale,
        )
        self.adapter_dab = StainAdapter(
            hidden_channels=hidden_channels,
            adapter_final_scale=adapter_final_scale,
        )

        self.channel_affine_h = LearnableChannelAffine()
        self.channel_affine_dab = LearnableChannelAffine()

        # --------------------------------------------------------
        # 3. Multi-scale construction (zero trainable parameters)
        # --------------------------------------------------------

        self.coarse = CoarseScaleView(coarse_size=coarse_size)

        # --------------------------------------------------------
        # 4. Shared ViT-B/16 Encoder (preserved unchanged).
        #    ONE instance reused for both fine and coarse views.
        # --------------------------------------------------------

        self.encoder = SharedViTEncoder(
            pretrained=pretrained,
            split_after=split_after,
        )

        # --------------------------------------------------------
        # 5. Existing Bidirectional Cross-Attention (preserved).
        #    ONE instance reused for both fine and coarse views.
        # --------------------------------------------------------

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
            beta=spatial_bias_beta,
            gamma=spatial_bias_gamma,
        )

        # --------------------------------------------------------
        # 6. Fusion components.
        #    ONE shared interaction + ONE shared stain gate reused
        #    for both fine and coarse views; ONE scale gate applied
        #    after both views are available.
        # --------------------------------------------------------

        self.interaction = BidirectionalInteraction(
            embed_dim=self.encoder.embed_dim,
            interaction_hidden_dim=interaction_hidden_dim,
        )
        self.stain_gate = StainGate(
            embed_dim=self.encoder.embed_dim,
            hidden_dim=interaction_hidden_dim,
        )
        self.scale_gate = ScaleGate(
            embed_dim=self.encoder.embed_dim,
            hidden_dim=interaction_hidden_dim,
        )

        # --------------------------------------------------------
        # 7. Existing Refinement Block (preserved unchanged)
        # --------------------------------------------------------

        self.refinement = RefinementBlock(
            embed_dim=self.encoder.embed_dim,
            num_heads=12,
        )

        # --------------------------------------------------------
        # 8. Existing Classification Head (preserved unchanged)
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
        self.coarse_size = coarse_size

    # ============================================================
    # Shared-branch processing
    # ============================================================

    def _process_scale(
        self,
        h_tokens: torch.Tensor,
        d_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes ONE scale (H + DAB token streams) independently
        through the SAME shared modules:

            encoder.forward_before
            -> cross_attention
            -> encoder.forward_after
            -> interaction
            -> stain_gate

        Parameters
        ----------
        h_tokens : torch.Tensor
            H tokens for this scale, shape (B, 197, 768).
        d_tokens : torch.Tensor
            DAB tokens for this scale, shape (B, 197, 768).

        Returns
        -------
        torch.Tensor
            Fused stain representation for this scale, shape (B, 197, 768).
        """
        batch_size = h_tokens.shape[0]

        # Shared ViT blocks 1..split_after (batched for GPU efficiency)
        stacked = torch.cat([h_tokens, d_tokens], dim=0)  # (2B, 197, 768)
        stacked = self.encoder.forward_before(stacked)
        h_tokens, d_tokens = stacked.split(batch_size, dim=0)

        # Existing bidirectional cross-attention (spatially biased)
        h_tokens, d_tokens = self.cross_attention(h_tokens, d_tokens)

        # Shared ViT blocks after + final LayerNorm
        stacked = torch.cat([h_tokens, d_tokens], dim=0)  # (2B, 197, 768)
        stacked = self.encoder.forward_after(stacked)
        h_final, d_final = stacked.split(batch_size, dim=0)

        # Shared bidirectional interaction (zero-init residuals)
        h_enriched, d_enriched = self.interaction(h_final, d_final)

        # Shared stain gate (H vs DAB, ~0.5 at init)
        fused_scale, stain_gate_values = self.stain_gate(h_enriched, d_enriched)

        # Store stain gate values for telemetry
        self._last_stain_gate_values = stain_gate_values

        return fused_scale

    # ============================================================
    # Forward
    # ============================================================

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
            h_channel, dab_channel = self.color_deconv(x_rgb)

        # h_channel : (B, 1, 224, 224)
        # dab_channel: (B, 1, 224, 224)

        # --------------------------------------------------------
        # Step 2: Input-side processing (independent per stream)
        # --------------------------------------------------------

        h = self.norm_h(h_channel)
        d = self.norm_dab(dab_channel)

        h = self.adapter_h(h)   # (B, 3, 224, 224)
        d = self.adapter_dab(d)  # (B, 3, 224, 224)

        h = self.channel_affine_h(h)
        d = self.channel_affine_dab(d)

        # --------------------------------------------------------
        # Step 3: Fine and coarse (low-frequency) views.
        #         Both remain (B, 3, 224, 224).
        # --------------------------------------------------------

        h_fine, d_fine = h, d
        h_coarse, d_coarse = self.coarse(h), self.coarse(d)

        # --------------------------------------------------------
        # Step 4: Patch embedding + positional encoding for each
        #         stream and each scale (shared encoder).
        # --------------------------------------------------------

        h_fine_tokens = self.encoder.embed(h_fine)      # (B, 197, 768)
        d_fine_tokens = self.encoder.embed(d_fine)
        h_coarse_tokens = self.encoder.embed(h_coarse)
        d_coarse_tokens = self.encoder.embed(d_coarse)

        # --------------------------------------------------------
        # Step 5: Process each scale independently through the
        #         SAME shared modules (weights shared, computation
        #         performed separately per scale).
        # --------------------------------------------------------

        f_fine = self._process_scale(h_fine_tokens, d_fine_tokens)      # (B, 197, 768)
        f_coarse = self._process_scale(h_coarse_tokens, d_coarse_tokens)  # (B, 197, 768)

        # --------------------------------------------------------
        # Step 6: Scale-adaptive gate (fine vs coarse, ~0.5 at init)
        # --------------------------------------------------------

        fused_tokens, scale_gate_values = self.scale_gate(f_fine, f_coarse)

        # Store scale gate values for telemetry
        self._last_scale_gate_values = scale_gate_values

        # --------------------------------------------------------
        # Step 7: Existing Refinement Block
        # --------------------------------------------------------

        refined_tokens = self.refinement(fused_tokens)

        # --------------------------------------------------------
        # Step 8: Existing Classification Head
        # --------------------------------------------------------

        logits = self.classifier(refined_tokens)

        return logits

    # ============================================================
    # Gate accessors (telemetry)
    # ============================================================

    def get_stain_gate_values(self) -> torch.Tensor | None:
        """
        Returns the MOST RECENT stain-gate values from the last
        forward pass (per scale: last scale processed writes last).

        Returns
        -------
        torch.Tensor or None
            Stain gate values of shape (B, N, C), or None if no
            forward pass has been performed.
        """
        return getattr(self, "_last_stain_gate_values", None)

    def get_scale_gate_values(self) -> torch.Tensor | None:
        """
        Returns the scale-gate values from the last forward pass.

        Returns
        -------
        torch.Tensor or None
            Scale gate values of shape (B, N, C), or None if no
            forward pass has been performed.
        """
        return getattr(self, "_last_scale_gate_values", None)

    # ============================================================
    # Parameter groups
    # ============================================================

    def get_parameter_groups(self) -> dict:
        """
        Returns exactly 5 parameter groups:

            vit             : shared ViT encoder
            existing_dsca   : cross-attention (incl. spatial bias) + refinement
            input_modules   : norm_h, norm_dab, adapter_h, adapter_dab,
                              channel_affine_h, channel_affine_dab
            fusion_modules  : interaction, stain_gate, scale_gate
            classifier      : classification head

        CoarseScaleView has zero parameters and is therefore NOT part
        of any group.

        Verifies that no parameter appears twice and that EVERY model
        parameter belongs to exactly one group - regardless of whether
        it is currently frozen (requires_grad). Stage freezing is handled
        separately via set_stage_requires_grad and must NOT change the
        definition of the architectural groups.
        Raises an error otherwise.
        """

        vit = list(self.encoder.parameters())

        existing_dsca = (
            list(self.cross_attention.parameters())
            + list(self.refinement.parameters())
        )

        input_modules = (
            list(self.norm_h.parameters())
            + list(self.norm_dab.parameters())
            + list(self.adapter_h.parameters())
            + list(self.adapter_dab.parameters())
            + list(self.channel_affine_h.parameters())
            + list(self.channel_affine_dab.parameters())
        )

        fusion_modules = (
            list(self.interaction.parameters())
            + list(self.stain_gate.parameters())
            + list(self.scale_gate.parameters())
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
        #
        # IMPORTANT: get_parameter_groups() must return ALL model
        # parameters, regardless of which are currently frozen.
        # Stage freezing is controlled separately via requires_grad
        # (set_stage_requires_grad). We therefore validate against
        # ALL self.parameters(), NOT only trainable ones.
        # --------------------------------------------------------

        grouped_ids: set[int] = set()
        for name, params in groups.items():
            for p in params:
                pid = id(p)
                if pid in grouped_ids:
                    raise RuntimeError(
                        f"Parameter appears in multiple groups: {name} "
                        f"(param id={pid})."
                    )
                grouped_ids.add(pid)

        all_ids = {id(p) for p in self.parameters()}

        if grouped_ids != all_ids:
            missing = all_ids - grouped_ids
            extra = grouped_ids - all_ids
            raise RuntimeError(
                "Parameter-group validation failed: "
                f"missing={len(missing)}, extra={len(extra)}. "
                "Every model parameter must belong to exactly one "
                "architectural group, regardless of requires_grad."
            )

        return groups

    # ============================================================
    # Single-instance verification (locked spec)
    # ============================================================

    def assert_single_shared_instances(self) -> None:
        """
        Asserts that exactly ONE instance of each shared module
        exists and that no duplicated ViT / cross-attention /
        interaction / stain-gate / scale-gate attribute was created.
        """
        from .shared_vit import SharedViTEncoder
        from .cross_attention import BidirectionalCrossAttention
        from .fusion_v3 import (
            BidirectionalInteraction,
            StainGate,
            ScaleGate,
        )

        checks = {
            "SharedViTEncoder": SharedViTEncoder,
            "BidirectionalCrossAttention": BidirectionalCrossAttention,
            "BidirectionalInteraction": BidirectionalInteraction,
            "StainGate": StainGate,
            "ScaleGate": ScaleGate,
        }

        module_counts = {name: 0 for name in checks}
        for module in self.modules():
            for name, cls in checks.items():
                if isinstance(module, cls):
                    module_counts[name] += 1

        for name, count in module_counts.items():
            if count != 1:
                raise RuntimeError(
                    f"Expected exactly one {name} instance, found {count}. "
                    "Duplicated shared modules are forbidden in v3."
                )

        # Verify the same Python module objects are actually reused by the
        # branch helper `_process_scale` (attribute-level check, complementary
        # to the type count above). This protects against accidentally wiring
        # a second, unregistered instance into the forward path.
        expected_attrs = [
            self.encoder,
            self.cross_attention,
            self.interaction,
            self.stain_gate,
            self.scale_gate,
        ]
        for attr in expected_attrs:
            if sum(1 for m in self.modules() if m is attr) != 1:
                raise RuntimeError(
                    "Shared-module reuse check failed: a shared module object "
                    "appears more than once in the module tree."
                )

    # ============================================================
    # Parameter counts
    # ============================================================

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
            "norm_dab": _count(self.norm_dab),
            "adapter_h": _count(self.adapter_h),
            "adapter_dab": _count(self.adapter_dab),
            "channel_affine_h": _count(self.channel_affine_h),
            "channel_affine_dab": _count(self.channel_affine_dab),
            "coarse": _count(self.coarse),
            "encoder": _count(self.encoder),
            "cross_attention": _count(self.cross_attention),
            "interaction": _count(self.interaction),
            "stain_gate": _count(self.stain_gate),
            "scale_gate": _count(self.scale_gate),
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

    # ============================================================
    # V2 compatibility loading (corrected locked spec, section 16)
    # ============================================================

    def load_v2_weights(self, checkpoint_path: str, device: torch.device) -> dict:
        """
        Loads compatible preserved weights from a v2 checkpoint into v3.

        PRESERVED weights (loaded from v2, must all be present):
            encoder.*, cross_attention.*, refinement.*, classifier.*

        NEW v3 modules (start FRESH, never loaded from v2):
            norm_h.*, norm_dab.*, adapter_h.*, adapter_dab.*,
            channel_affine_h.*, channel_affine_dab.*,
            coarse.*, interaction.*, stain_gate.*, scale_gate.*

        LEGACY parameters in the v2 checkpoint (ignored):
            proj_h.*, proj_d.*, fusion.*

        The loader explicitly reports which v2 parameters were loaded
        and which v3 parameters are new, and raises an exception if any
        preserved parameter is missing.
        """
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
        fresh_prefixes = [
            "norm_h.", "norm_dab.",
            "adapter_h.", "adapter_dab.",
            "channel_affine_h.", "channel_affine_dab.",
            "coarse.", "interaction.", "stain_gate.", "scale_gate.",
        ]
        legacy_prefixes = ["proj_h.", "proj_d.", "fusion."]

        # Only preserved keys are passed to load_state_dict (strict=False),
        # so none of the v2 keys for the fresh modules can leak in.
        preserved_state = {
            key: value
            for key, value in state_dict.items()
            if any(key.startswith(p) for p in preserved_prefixes)
        }

        # Check preserved modules for missing keys
        missing_preserved = [
            key
            for key in model_state
            if any(key.startswith(p) for p in preserved_prefixes)
            and key not in state_dict
        ]

        if missing_preserved:
            raise RuntimeError(
                "Preserved modules are missing weights from the v2 "
                f"checkpoint:\n{missing_preserved}"
            )

        # Load ONLY the preserved weights
        load_result = self.load_state_dict(preserved_state, strict=False)

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        print("=" * 60)
        print("V3 COMPATIBILITY LOAD")
        print("=" * 60)

        print("\nLoaded preserved parameters (from v2):")
        for prefix in preserved_prefixes:
            n = sum(1 for k in state_dict if k.startswith(prefix))
            print(f"  {prefix:<22} ✓ ({n} tensors)")

        print("\nLoaded compatible v2 parameters:")
        print("  (none - all new v3 modules start fresh per locked spec)")

        print("\nFresh v3 parameters (not loaded):")
        for prefix in fresh_prefixes:
            n = sum(1 for k in model_state if k.startswith(prefix))
            print(f"  {prefix:<22} fresh ({n} tensors)")

        print("\nUnexpected legacy parameters (ignored):")
        for prefix in legacy_prefixes:
            n = sum(1 for k in state_dict if k.startswith(prefix))
            print(f"  {prefix:<22} ignored ({n} tensors)")

        # v2 keys that correspond to fresh v3 modules but were present
        # in the v2 checkpoint (deliberately NOT loaded)
        v2_fresh_keys = [
            k
            for k in state_dict
            if any(k.startswith(p) for p in fresh_prefixes)
        ]
        if v2_fresh_keys:
            print("\nv2 fresh-module keys present in checkpoint (NOT loaded):")
            for prefix in fresh_prefixes:
                n = sum(1 for k in v2_fresh_keys if k.startswith(prefix))
                if n > 0:
                    print(f"  {prefix:<22} skipped ({n} tensors)")

        print("\nPreserved parameters missing:")
        print("  NONE" if not missing_preserved else f"  {missing_preserved}")

        print("=" * 60)

        return {
            "loaded_preserved": sorted(preserved_state.keys()),
            "fresh_v3_prefixes": fresh_prefixes,
            "legacy_ignored": [
                k for k in state_dict
                if any(k.startswith(p) for p in legacy_prefixes)
            ],
            "missing_preserved": missing_preserved,
            "load_result": load_result,
        }