# ============================================================
# ViT-B16-Stain-XAttn — Package
# ============================================================
#
# ViT-B16 fine-tuned on RGB, optionally fusing spatial stain
# (H + DAB) info via GATED cross-attention in the last blocks.
# Zero-init gate => numerically identical to plain ViT-B16 at init.
#
# Experiment: vit_b16_stain_xattn_001
# ============================================================

from .vit_b16_stain_xattn_model import ViTB16StainXAttn
from .gated_cross_attention import GatedCrossAttentionBlock
from .color_deconv import ColorDeconvolution, deconvolve_numpy
from .stain_stats import save_stain_stats, load_stain_stats

__all__ = [
    "ViTB16StainXAttn",
    "GatedCrossAttentionBlock",
    "ColorDeconvolution",
    "deconvolve_numpy",
    "save_stain_stats",
    "load_stain_stats",
]