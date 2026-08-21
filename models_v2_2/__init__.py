# ============================================================
# DSS-ViT v2.2 Model Package
# Dual-Stream Stain Vision Transformer for HER2 IHC
# ============================================================
#
# DSS-ViT v2.2: Strong regularization + reduced capacity.
#   - StainEncoder: 8 tokens, bottleneck 256 (reduced from 16/512)
#   - Same cross-attention + gate + ordinal head as v2.1
#   - Uses the retrained baseline's WSI-aware split
#
# This package is independent of models/, models_v2/, models_v3/,
# and models_v2_1/.
# ============================================================

from .dss_vit import DSSViT
from .color_deconv import ColorDeconvolution
from .stain_encoder import StainEncoder
from .ordinal_head import OrdinalHead, ordinal_loss, total_loss
from .stain_stats import load_stain_stats, save_stain_stats

__all__ = [
    "DSSViT",
    "ColorDeconvolution",
    "StainEncoder",
    "OrdinalHead",
    "ordinal_loss",
    "total_loss",
    "load_stain_stats",
    "save_stain_stats",
]