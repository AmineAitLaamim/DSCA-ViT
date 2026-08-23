# ============================================================
# UNI-Regularized — Package
# ============================================================
#
# Same architecture as UNI-baseline, with a regularized
# fine-tuning recipe (AdamW + early stopping) and corrected
# Stage 2 initialization from best_stage1.pt.
#
# Experiment: uni_regularized_001
# ============================================================

from .uni_regularized_model import UNIRegularizedModel

__all__ = ["UNIRegularizedModel"]