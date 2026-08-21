# ============================================================
# Plain ViT-B16 Baseline Model Package
# ============================================================
#
# Independent baseline for the HER2-IHC-40x classification task.
# Reproduces the plain ViT-B16 (ImageNet-pretrained) that reached
# 95.02% official test accuracy.
#
# This package is independent of models/, models_v2/, models_v3/,
# and models_v2_1/.
# ============================================================

from .plain_vit_b16 import PlainViTB16

__all__ = ["PlainViTB16"]