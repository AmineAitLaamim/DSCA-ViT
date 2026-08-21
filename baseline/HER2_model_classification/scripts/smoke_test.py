from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Allow running:
#     python scripts/smoke_test.py
# from the project root by making "src" importable.
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
)
from src.reproducibility import environment_metadata, seed_everything
from src.smoke_tests import run_smoke_tests
from src.transforms import build_transforms
from src.utils import write_json


def load_samples(config):
    paths = config["paths"]

    if (
        paths.get("train_manifest")
        and paths.get("val_manifest")
        and paths.get("test_manifest")
    ):
        samples = (
            read_manifest(paths["train_manifest"])
            + read_manifest(paths["val_manifest"])
            + read_manifest(paths["test_manifest"])
        )
    else:
        samples = scan_directory_dataset(paths["data_dir"], config)
        samples = split_train_val(
            samples,
            float(config["dataset"]["val_fraction"]),
            int(config["seed"]),
        )

    validate_samples(samples, config)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data_dir is not None:
        config["paths"]["data_dir"] = args.data_dir

    seed_everything(int(config["seed"]))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_config(config, out / "smoke_config.yaml")
    write_json(environment_metadata(), out / "smoke_environment.json")

    samples = load_samples(config)

    class_to_idx = config["classes"]

    train_dataset = HER2PatchDataset(
        samples_for_split(samples, "train"),
        class_to_idx,
        build_transforms(config, "train"),
    )

    val_dataset = HER2PatchDataset(
        samples_for_split(samples, "val"),
        class_to_idx,
        build_transforms(config, "val"),
    )

    run_smoke_tests(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=out,
    )

    print(f"Smoke tests passed. Outputs: {out}")


if __name__ == "__main__":
    main()