# ============================================================
# DSS-ViT — Official Test Evaluation Script (CLI)
# ============================================================
#
# Loads the best Stage 3 checkpoint and evaluates on the official
# test set. Reports accuracy, balanced accuracy, macro-F1, QWK,
# per-class metrics, and confusion matrix.
#
# Usage:
#   python utils/evaluate_dss_vit.py --config configs/dss_vit_config.yaml
# ============================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import HER2Dataset, get_test_transform
from models_v2_1 import DSSViT, load_stain_stats
from utils.metrics_dss_vit import compute_metrics, print_metrics


def setup_logging(log_dir: str) -> logging.Logger:
    """Sets up console + file logging."""
    logger = logging.getLogger("dss_vit_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "eval.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSS-ViT on official test set")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dss_vit_config.yaml",
        help="Path to the DSS-ViT config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint. Defaults to <checkpoint_dir>/best_stage3.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    test_dir = config["paths"]["test_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    log_dir = config["paths"]["log_dir"]
    results_dir = config["paths"]["results_dir"]
    stain_stats_path = config["paths"]["stain_stats_path"]
    image_size = config["dataset"]["image_size"]

    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, "best_stage3.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Train the model first or provide --checkpoint."
        )

    logger = setup_logging(log_dir)
    logger.info(f"Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------
    # Load stain stats
    # ------------------------------------------------------------
    stain_stats = load_stain_stats(stain_stats_path)

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = DSSViT(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        num_stain_tokens=config["model"]["num_stain_tokens"],
        stain_bottleneck_dim=config["model"]["stain_bottleneck_dim"],
        stain_stats=stain_stats,
        image_size=config["model"]["image_size"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Model loaded from checkpoint.")

    # ------------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------------
    test_dataset = HER2Dataset(
        root_dir=test_dir,
        transform=get_test_transform(image_size=image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Test images: {len(test_dataset)}")

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    all_predictions: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            pred = model(images)
            preds = pred["probs"].argmax(dim=1)

            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    class_names = test_dataset.get_class_names()
    metrics = compute_metrics(all_labels, all_predictions, class_names)

    print_metrics(metrics)

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    results = {
        "checkpoint": checkpoint_path,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "qwk": metrics["qwk"],
            "per_class": metrics["per_class"],
        },
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": class_names,
    }

    results_path = os.path.join(results_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to '{results_path}'")

    # Also save a plain-text report
    report_path = os.path.join(results_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DSS-ViT Official Test Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy          : {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1          : {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1       : {metrics['weighted_f1']:.4f}\n")
        f.write(f"QWK               : {metrics['qwk']:.4f}\n\n")
        f.write("Per-class:\n")
        f.write(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}\n")
        for name, pc in metrics["per_class"].items():
            f.write(
                f"{name:<12} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
                f"{pc['f1']:>10.4f} {pc['support']:>8d}\n"
            )
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics["confusion_matrix"]) + "\n")
    logger.info(f"Report saved to '{report_path}'")


if __name__ == "__main__":
    main()