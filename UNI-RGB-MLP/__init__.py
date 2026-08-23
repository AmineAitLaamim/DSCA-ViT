# ============================================================
# UNI-RGB-MLP — Package
# ============================================================
#
# Control ablation: frozen UNI feature extractor + RGB-only MLP
# head. No stain branch. Isolates the contribution of stain
# side-information in UNI-Stain-MLP (UNI_v2/).
#
# Experiment: uni_rgb_mlp_001
# ============================================================

from .uni_rgb_mlp import UNIRGBMLP

__all__ = ["UNIRGBMLP"]