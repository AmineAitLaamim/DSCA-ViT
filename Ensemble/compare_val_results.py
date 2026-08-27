# ============================================================
# Ensemble — compare validation results across all four variants
# ============================================================
#
# Reads each variant's results/Ensemble/<variant>/<variant>_001/
# val_results.json and prints a comparison table.
#
# Usage:
#   uv run python Ensemble/compare_val_results.py \
#       --config configs/ensemble_config.yaml
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Run by path — loop over all six variant folders.
VARIANTS = [
    ("vit_ce_uni_ce",             ["vit_ce", "uni_ce"],                       False),
    ("vit_ce_uni_ce_tta",         ["vit_ce", "uni_ce"],                       True),
    ("vit_ce_uni_ce_vit_corn",    ["vit_ce", "uni_ce", "vit_corn"],           False),
    ("vit_ce_uni_ce_vit_corn_tta",["vit_ce", "uni_ce", "vit_corn"],           True),
    ("vit_corn_uni_ce",           ["vit_corn", "uni_ce"],                     False),
    ("vit_corn_uni_ce_tta",       ["vit_corn", "uni_ce"],                     True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ensemble val results")
    parser.add_argument("--config", type=str,
                        default="configs/ensemble_config.yaml")
    return parser.parse_args()


def load_val(config, variant, exp):
    results_dir = os.path.join(
        config["paths"]["results_dir_base"], variant, exp
    )
    path = os.path.join(results_dir, "val_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    import yaml
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    rows = []
    for variant, models, tta in VARIANTS:
        exp = f"{variant}_001"
        data = load_val(config, variant, exp)
        if data is None:
            rows.append((variant, models, tta, None))
            continue
        m = data["metrics"] if "metrics" in data else data
        rows.append((variant, models, tta, m))

    print("=" * 88)
    print(f"{'Variant':<26} {'Models':<34} {'TTA':<5} {'Acc':>8} "
          f"{'BalAcc':>8} {'MacroF1':>8} {'QWk':>8}")
    print("=" * 88)
    for variant, models, tta, m in rows:
        if m is None:
            print(f"{variant:<26} {','.join(models):<34} {str(tta):<5} "
                  f"{'N/A':>8}")
            continue
        print(f"{variant:<26} {','.join(models):<34} {str(tta):<5} "
              f"{m['accuracy']*100:>7.2f}% "
              f"{m['balanced_accuracy']*100:>7.2f}% "
              f"{m['macro_f1']*100:>7.2f}% "
              f"{m['qwk']:>8.4f}")
    print("=" * 88)
    print("Reference — best single-model test accuracy: 94.69%")


if __name__ == "__main__":
    main()
