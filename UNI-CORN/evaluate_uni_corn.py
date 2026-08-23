# ============================================================
# UNI-CORN — Official Test Evaluation Script (CLI)
# ============================================================
#
# Loads the best Stage 2 checkpoint (best_stage2.pt — NEVER
# last.pt) and evaluates on the official test set exactly once.
#
# Predictions use corn_label_from_logits(logits) exclusively.
#
# Reports accuracy, balanced accuracy, macro-F1, weighted-F1,
# per-class P/R/F1, QWK, MAE, and confusion matrix.
#
# Usage:
#   uv run python UNI-CORN/evaluate_uni_corn.py \
#       --config configs/uni_corn_config.yaml
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from coral_pytorch.dataset import corn_label_from_logits
from torch.utils.data import DataLoader

# Add project root to path so `from baseline import ...` works.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from baseline import HER2BaselineDataset, compute_metrics
except ImportError:
    # Self-contained fallback (keeps the hyphen folder usable standalone).
    from torch.utils.data import Dataset as _Dataset
    from PIL import Image as _Image
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        cohen_kappa_score,
    )

    CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]

    class HER2BaselineDataset(_Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = CLASSES
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.image_paths = []
            self.labels = []
            exts = {".png", ".jpg", ".jpeg"}
            if not self.root_dir.exists():
                raise ValueError(f"Dataset root not found: {self.root_dir}")
            for cls in self.classes:
                cls_dir = self.root_dir / cls
                if not cls_dir.exists():
                    raise ValueError(f"Missing class dir: {cls_dir}")
                label = self.class_to_idx[cls]
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in exts:
                        self.image_paths.append(p)
                        self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = _Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

        def get_class_names(self):
            return self.classes

    def compute_metrics(y_true, y_pred, class_names=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if class_names is None:
            num_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
            class_names = [f"class_{i}" for i in range(num_classes)]
        accuracy = float((y_true == y_pred).mean())
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
        )
        per_class = {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(len(class_names))
        }
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        return {
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "qwk": qwk,
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_names": class_names,
        }


# Sibling imports (hyphen folder, run by path).
from uni_corn_model import UNICORN
from train_uni_corn import get_test_transform


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging(log_dir):
    logger = logging.getLogger("uni_corn_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "eval.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNI-CORN on Official Test Set")
    parser.add_argument("--config", type=str, default="configs/uni_corn_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    test_dir = config["paths"]["test_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    log_dir = config["paths"]["log_dir"]
    results_dir = config["paths"]["results_dir"]
    uni_checkpoint_path = config["paths"]["uni_checkpoint_path"]
    image_size = config["model"]["image_size"]

    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, "best_stage2.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Train the model first (best_stage2.pt is required) or provide --checkpoint."
        )

    logger = setup_logging(log_dir)
    logger.info(f"Checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build the UNI-CORN and load best_stage2.pt.
    model = UNICORN(
        checkpoint_path=uni_checkpoint_path,
        num_classes=config["model"]["num_classes"],
        verbose=True,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded from checkpoint (best_stage2.pt).")

    # Official test set — raw RGB [0,1] transforms (model normalizes internally).
    test_dataset = HER2BaselineDataset(
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

    all_predictions: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(images)
            # CORN predictions use the official coral-pytorch conversion.
            preds = corn_label_from_logits(pred["logits"])
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    class_names = test_dataset.get_class_names()
    metrics = compute_metrics(all_labels, all_predictions, class_names)

    # MAE (mean absolute error between predicted and true integer labels)
    mae = float(np.mean(np.abs(all_predictions - all_labels)))

    print("=" * 60)
    print("UNI-CORN — Official Test Evaluation")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"QWK               : {metrics['qwk']:.4f}")
    print(f"MAE               : {mae:.4f}")
    print("-" * 60)
    print("Per-class:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    for name, pc in metrics["per_class"].items():
        print(
            f"{name:<12} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
            f"{pc['f1']:>10.4f} {pc['support']:>8d}"
        )
    print("-" * 60)
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 60)

    # Save JSON results
    results = {
        "checkpoint": checkpoint_path,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "qwk": metrics["qwk"],
            "mae": mae,
            "per_class": metrics["per_class"],
        },
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": class_names,
    }
    results_path = os.path.join(results_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to '{results_path}'")

    # Save human-readable report
    report_path = os.path.join(results_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("UNI-CORN — Official Test Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint        : {checkpoint_path}\n")
        f.write(f"Accuracy          : {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1          : {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1       : {metrics['weighted_f1']:.4f}\n")
        f.write(f"QWK               : {metrics['qwk']:.4f}\n")
        f.write(f"MAE               : {mae:.4f}\n\n")
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