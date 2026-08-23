# ============================================================
# ViT-B16-CORN — Package
# ============================================================
#
# ViT-B16 backbone + CORN ordinal regression head for HER2
# severity order (0 < 1+ < 2+ < 3+).
#
# Experiment: vit_b16_corn_001
# ============================================================

from .vit_b16_corn_model import ViTB16CORN

__all__ = ["ViTB16CORN"]