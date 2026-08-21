from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics, curve_data
from .utils import write_json


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device):
    model.eval()
    labels, preds, probs, rows = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        labels.extend(batch["label"].cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
        for i, path in enumerate(batch["image_path"]):
            rows.append(
                {
                    "image_path": path,
                    "patient_id": batch.get("patient_id", [""] * len(batch["image_path"]))[i],
                    "wsi_id": batch.get("wsi_id", [""] * len(batch["image_path"]))[i],
                }
            )
    return np.array(labels), np.array(preds), np.array(probs), rows


def save_predictions(rows, y_true, y_pred, y_prob, class_names: list[str], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "patient_id", "wsi_id", "true_label", "predicted_label"] + [
        f"probability_{name}" for name in class_names
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, true, pred, prob in zip(rows, y_true, y_pred, y_prob):
            item = dict(row)
            item["true_label"] = class_names[int(true)]
            item["predicted_label"] = class_names[int(pred)]
            for i, name in enumerate(class_names):
                item[f"probability_{name}"] = float(prob[i])
            writer.writerow(item)


def save_confusion_matrix(cm, class_names: list[str], path: str | Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_curves(curves: dict[str, Any], path_roc: str | Path, path_pr: str | Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, data in curves["roc"].items():
        plt.plot(data["fpr"], data["tpr"], label=name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_roc, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, data in curves["pr"].items():
        plt.plot(data["recall"], data["precision"], label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_pr, dpi=200)
    plt.close()


def evaluate_and_save(model, loader: DataLoader, device: torch.device, class_names: list[str], output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_prob, rows = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    curves = curve_data(y_true, y_prob, class_names)
    write_json(metrics, out / "metrics.json")
    write_json(metrics["classification_report"], out / "classification_report.json")
    save_predictions(rows, y_true, y_pred, y_prob, class_names, out / "predictions.csv")
    save_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names, out / "confusion_matrix.png")
    save_curves(curves, out / "roc.png", out / "pr.png")
    return metrics

