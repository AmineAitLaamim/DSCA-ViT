# ============================================================
# UNI-Stain-EarlyFusion — Package
# ============================================================
#
# UNI (ViT-L/16, DINOv2, mass100k) modified to accept a 5-channel
# input [RGB + Hematoxylin + DAB] via a replaced patch-embed
# projection (early fusion), then FULL fine-tuning (Stage 2).
#
# Experiment: uni_stain_earlyfusion_001
# ============================================================

from .uni_stain_earlyfusion_model import UNIStainEarlyFusion
from .color_deconv import ColorDeconvolution, deconvolve_numpy
from .stain_stats import save_stain_stats, load_stain_stats

__all__ = [
    "UNIStainEarlyFusion",
    "ColorDeconvolution",
    "deconvolve_numpy",
    "save_stain_stats",
    "load_stain_stats",
]