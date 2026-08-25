# ============================================================
# Ensemble — Variant: ViT-B16 CE + UNI CE (no TTA)
# ============================================================
#
# Equal-weight probability averaging of:
#   - ViT-B16 CE  (PlainViTB16, plain_vit_baseline_001)
#   - UNI CE      (UNIBaselineModel, uni_baseline_001)
#
# Run by path:
#   uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
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

VARIANT_NAME = "vit_ce_uni_ce"
EXPERIMENT_NAME = "vit_ce_uni_ce_001"
MODEL_NAMES = ["vit_ce", "uni_ce"]
TTA = False


if __name__ == "__main__":
    run_variant(VARIANT_NAME, EXPERIMENT_NAME, MODEL_NAMES, TTA)
