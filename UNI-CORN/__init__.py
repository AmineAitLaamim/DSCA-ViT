# ============================================================
# UNI-CORN — Package
# ============================================================
#
# UNI backbone + CORN ordinal regression head for HER2
# severity order (0 < 1+ < 2+ < 3+).
#
# Experiment: uni_corn_001
# ============================================================

from .uni_corn_model import UNICORN

__all__ = ["UNICORN"]