# ============================================================
# UNI-Regularized-v2 — Package
# ============================================================
#
# Follow-up to UNI-Regularized (v1), targeting the same
# diagnosed problem:
#   "Train loss still collapsed to near-zero too early, and
#    the previous regularization did not change the
#    overfitting curve enough."
#
# v2 bundles THREE regularization changes:
#   1. Stochastic depth  : UNI backbone drop_path_rate=0.2
#   2. Head dropout      : nn.Dropout(0.3) before the Linear head
#   3. Rebalanced wd     : Stage 2 backbone wd=0.5, head wd=0.05
#                          -> lr*wd parity = 5.00e-06 both sides
#
# Experiment: uni_regularized_v2_001
# ============================================================

from .uni_regularized_v2_model import UNIRegularizedV2

__all__ = ["UNIRegularizedV2"]