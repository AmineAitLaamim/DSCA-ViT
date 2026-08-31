# ============================================================
# Ensemble — collect all variants' test_report.txt into one file
# ============================================================
#
# Reads each variant's results/Ensemble/<variant>/<variant>_001/
# test_report.txt and writes ONE combined file:
#
#   Ensemble/test_reports_all_variants.txt
#
# Format per variant:
#   ================
#   <variant_name>  (experiment <variant>_001)
#   ================
#   <full contents of that variant's test_report.txt>
#   ============================================================
#
# Variants without a test_report.txt are listed as "MISSING" so you
# know which ones have not yet run the official test.
#
# Usage (run on Toubkal, where the results live):
#   uv run python Ensemble/collect_test_reports.py \
#       --config configs/ensemble_config.yaml
#   # optional: --out Ensemble/test_reports_all_variants.txt
# ============================================================

from __future__ import annotations

import argparse
import os
from pathlib import Path

# The six ensemble variants. A variant only gets an official test result
# once `--eval-test` has been run for it.
VARIANTS = [
    "vit_ce_uni_ce",
    "vit_ce_uni_ce_tta",
    "vit_ce_uni_ce_vit_corn",
    "vit_ce_uni_ce_vit_corn_tta",
    "vit_corn_uni_ce",
    "vit_corn_uni_ce_tta",
]

DEFAULT_OUT = Path(__file__).resolve().parent / "test_reports_all_variants.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all Ensemble test_reports into one txt file."
    )
    parser.add_argument("--config", type=str,
                        default="configs/ensemble_config.yaml")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output txt path (default: Ensemble/test_reports_all_variants.txt)")
    return parser.parse_args()


def main() -> None:
    import yaml
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_base = config["paths"]["results_dir_base"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "=" * 70 + "\n"
    header += "ENSEMBLE — ALL VARIANT TEST REPORTS (official test)\n"
    header += "=" * 70 + "\n"
    lines = [header]

    for variant in VARIANTS:
        exp = f"{variant}_001"
        report = os.path.join(results_base, variant, exp, "test_report.txt")
        lines.append("\n" + "=" * 70)
        lines.append(f"{variant}  (experiment {exp})")
        lines.append("=" * 70)
        if os.path.exists(report):
            with open(report, "r") as rf:
                content = rf.read().rstrip("\n")
            lines.append(content if content else "(report is empty)")
        else:
            lines.append(f"[MISSING] no test_report.txt at:\n  {report}")

    lines.append("\n" + "=" * 70)
    lines.append("END OF COLLECTION")
    lines.append("=" * 70)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote combined test reports to:\n  {out_path}")
    for variant in VARIANTS:
        exp = f"{variant}_001"
        report = os.path.join(results_base, variant, exp, "test_report.txt")
        status = os.path.exists(report)
        print(f"  [{'OK' if status else 'MISSING'}] {variant}")


if __name__ == "__main__":
    main()