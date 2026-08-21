from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.checkpoints import load_checkpoint
from src.config import load_config
from src.dataset import HER2PatchDataset, read_manifest, samples_for_split, scan_directory_dataset, split_train_val, validate_samples
from src.evaluate import evaluate_and_save
from src.models import build_model
from src.transforms import build_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["paths"]["data_dir"] = args.data_dir
    paths = config["paths"]
    if paths.get("train_manifest") and paths.get("val_manifest") and paths.get("test_manifest"):
        samples = read_manifest(paths["train_manifest"]) + read_manifest(paths["val_manifest"]) + read_manifest(paths["test_manifest"])
    else:
        samples = scan_directory_dataset(config["paths"]["data_dir"], config)
        samples = split_train_val(samples, float(config["dataset"]["val_fraction"]), int(config["seed"]))
    validate_samples(samples, config)
    test_dataset = HER2PatchDataset(samples_for_split(samples, "test"), config["classes"], build_transforms(config, "test"))
    loader = DataLoader(test_dataset, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, device=device)
    if ckpt.get("ema"):
        model.load_state_dict(ckpt["ema"])
    class_names = [k for k, _ in sorted(config["classes"].items(), key=lambda item: item[1])]
    evaluate_and_save(model, loader, device, class_names, args.output_dir)


if __name__ == "__main__":
    main()
