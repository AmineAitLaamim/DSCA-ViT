from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Allow this script to be executed directly:
#     python scripts/train_hybrid.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import load_config, save_config
from src.dataset import (
    HER2PatchDataset,
    read_manifest,
    samples_for_split,
    scan_directory_dataset,
    split_train_val,
    validate_samples,
    verify_official_counts,
)
from src.logging_utils import setup_logger
from src.reproducibility import environment_metadata, seed_everything
from src.smoke_tests import run_smoke_tests
from src.train import train_hybrid
from src.transforms import build_transforms
from src.utils import next_experiment_dir, write_json


def load_samples(config):
    paths = config["paths"]
    if paths.get("train_manifest") and paths.get("val_manifest") and paths.get("test_manifest"):
        samples = read_manifest(paths["train_manifest"]) + read_manifest(paths["val_manifest"]) + read_manifest(paths["test_manifest"])
    else:
        samples = scan_directory_dataset(paths["data_dir"], config)
        verify_official_counts(samples, config)
        samples = split_train_val(samples, float(config["dataset"]["val_fraction"]), int(config["seed"]))
    validate_samples(samples, config)
    return samples


def print_startup_report(config, samples, exp_dir: Path, logger) -> None:
    expected = config["dataset"].get("expected_counts", {})
    class_to_idx = config["dataset"].get("class_mapping") or config["classes"]
    effective_report = validate_samples(samples, config)
    source_train = sum(int(v) for v in expected.get("train", {}).values())
    source_test = sum(int(v) for v in expected.get("test", {}).values())
    lines = [
        "=" * 50,
        "HER2-IHC Hybrid Ensemble Training",
        "=" * 50,
        "",
        "Dataset: HER2-IHC-40x",
        "",
        f"Training images : {source_train}",
        f"Testing images  : {source_test}",
        "",
        "Class Distribution",
        "------------------",
    ]
    for label in class_to_idx:
        train_count = int(expected.get("train", {}).get(label, 0))
        test_count = int(expected.get("test", {}).get(label, 0))
        lines.append(f"{label:<8} : {train_count:<4} / {test_count}")
    lines.extend(
        [
            "",
            "Dataset verification: PASSED",
            "",
            "Effective Training Split",
            "------------------------",
        ]
    )
    for split in ("train", "val", "test"):
        split_total = sum(effective_report["class_counts"].get(split, {}).values())
        if split_total:
            lines.append(f"{split:<5}: {split_total}")
    lines.extend(
        [
            "",
            "Backbones",
            "----------",
            f"EVA-02-Large     : {config['model']['eva_name']}",
            f"ViT-Base         : {config['model']['vit_name']}",
            f"ConvNeXt-V2-Nano : {config['model']['convnext_name']}",
            "",
            "Training Mode",
            "-------------",
            "Paper-faithful end-to-end hybrid training",
            "",
            "Checkpoint directory",
            "--------------------",
            str(exp_dir),
            "",
            "Documented assumptions",
            "----------------------",
            "See config.yaml and IMPLEMENTATION_PLAN.md for paper ambiguities and chosen defaults.",
            "",
            "=" * 50,
        ]
    )
    report = "\n".join(lines)
    print(report)
    logger.info("\n%s", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--skip-smoke-tests", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["paths"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    seed_everything(int(config["seed"]))
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = next_experiment_dir(config["paths"]["output_dir"], config["experiment_prefix"], config["experiment_name"])
    save_config(config, exp_dir / "config.yaml")
    write_json(environment_metadata(), exp_dir / "environment.json")
    logger = setup_logger(exp_dir / "training.log")
    logger.info("Experiment directory: %s", exp_dir)

    samples = load_samples(config)
    print_startup_report(config, samples, exp_dir, logger)
    class_to_idx = config["classes"]
    train_dataset = HER2PatchDataset(samples_for_split(samples, "train"), class_to_idx, build_transforms(config, "train"))
    val_dataset = HER2PatchDataset(samples_for_split(samples, "val"), class_to_idx, build_transforms(config, "val"))
    test_dataset = HER2PatchDataset(samples_for_split(samples, "test"), class_to_idx, build_transforms(config, "test"))

    if not args.skip_smoke_tests:
        run_smoke_tests(config, train_dataset, val_dataset, Path(exp_dir))
    metrics = train_hybrid(config, train_dataset, val_dataset, test_dataset, exp_dir, logger)
    logger.info("Final test metrics: %s", metrics)


if __name__ == "__main__":
    main()
