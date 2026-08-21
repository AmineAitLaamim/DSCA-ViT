from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.dataset import scan_directory_dataset, split_train_val, validate_samples, write_manifest
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    config = load_config(args.config)
    samples = scan_directory_dataset(args.data_dir, config)
    samples = split_train_val(samples, float(config["dataset"]["val_fraction"]), int(config["seed"]))
    report = validate_samples(samples, config)

    out = Path(args.out_dir)
    write_manifest([s for s in samples if s.split == "train"], out / "train_manifest.csv")
    write_manifest([s for s in samples if s.split == "val"], out / "val_manifest.csv")
    write_manifest([s for s in samples if s.split == "test"], out / "test_manifest.csv")
    write_json(report, out / "dataset_report.json")
    print(report)


if __name__ == "__main__":
    main()

