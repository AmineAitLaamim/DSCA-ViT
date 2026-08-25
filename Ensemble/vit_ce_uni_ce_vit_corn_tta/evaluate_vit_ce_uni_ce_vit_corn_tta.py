# ============================================================
# Ensemble — Variant: ViT-B16 CE + UNI CE + ViT-B16 CORN + TTA
# ============================================================
#
# Equal-weight probability averaging of:
#   - ViT-B16 CE   (PlainViTB16, plain_vit_baseline_001)
#   - UNI CE       (UNIBaselineModel, uni_baseline_001)
#   - ViT-B16 CORN (ViTB16CORN, vit_b16_corn_001)
#
# TTA on: each model's probabilities are averaged over 6
# augmentations (original, hflip, vflip, rot90, rot180, rot270)
# BEFORE combining across models. Only probabilities are averaged,
# never logits (CORN uses chain-rule [B,4] probs).
#
# Run by path:
#   uv run python Ensemble/vit_ce_uni_ce_vit_corn_tta/evaluate_vit_ce_uni_ce_vit_corn_tta.py \
#       --config configs/ensemble_config.yaml
# ============================================================

import os

# Force any accidental Hugging Face / timm network access to fail loudly
# BEFORE any torch/timm imports below.
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from pathlib import Path

# Insert Ensemble parent dir into sys.path for `from common...` imports.
ENSEMBLE_DIR = Path(__file__).resolve().parent.parent
if str(ENSEMBLE_DIR) not in sys.path:
    sys.path.insert(0, str(ENSEMBLE_DIR))

from common.evaluator import run_variant

VARIANT_NAME = "vit_ce_uni_ce_vit_corn_tta"
EXPERIMENT_NAME = "vit_ce_uni_ce_vit_corn_tta_001"
MODEL_NAMES = ["vit_ce", "uni_ce", "vit_corn"]
TTA = True


if __name__ == "__main__":
    run_variant(VARIANT_NAME, EXPERIMENT_NAME, MODEL_NAMES, TTA)
