# ============================================================
# Ensemble — metrics computation (sklearn suite)
# ============================================================
#
# Same metric suite as every other package in this project:
# accuracy, balanced accuracy, macro-F1, weighted-F1, QWK,
# per-class precision/recall/F1/support, and confusion matrix.
# ============================================================

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["class_0", "class_1+", "class_2+", "class_3+"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute the full classification metric suite."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_names is None:
        num_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
        class_names = [f"class_{i}" for i in range(num_classes)]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "qwk": float(qwk),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def metrics_to_serializable(metrics: Dict) -> Dict:
    """Convert numpy-containing metrics dict to JSON-serializable form."""
    out = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "qwk": metrics["qwk"],
        "per_class": metrics["per_class"],
        "confusion_matrix": np.asarray(
            metrics["confusion_matrix"]
        ).tolist(),
        "class_names": metrics["class_names"],
    }
    return out


def format_report(metrics: Dict, title: str) -> str:
    """Return a formatted human-readable report string."""
    lines = []
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    lines.append(f"Accuracy          : {metrics['accuracy']:.4f}")
    lines.append(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    lines.append(f"Macro F1          : {metrics['macro_f1']:.4f}")
    lines.append(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    lines.append(f"QWK               : {metrics['qwk']:.4f}")
    lines.append("-" * 60)
    lines.append("Per-class:")
    lines.append(
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'Support':>8}"
    )
    for name, pc in metrics["per_class"].items():
        lines.append(
            f"{name:<12} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
            f"{pc['f1']:>10.4f} {pc['support']:>8d}"
        )
    lines.append("-" * 60)
    lines.append("Confusion Matrix:")
    lines.append(str(np.asarray(metrics["confusion_matrix"])))
    lines.append("=" * 60)
    return "\n".join(lines)


def save_results(
    results_dir: str,
    prefix: str,
    metrics: Dict,
    extra: Optional[Dict] = None,
) -> None:
    """Save {prefix}_results.json and {prefix}_report.txt to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    serial = metrics_to_serializable(metrics)
    if extra:
        serial = {**extra, **serial}

    json_path = os.path.join(results_dir, f"{prefix}_results.json")
    with open(json_path, "w") as f:
        json.dump(serial, f, indent=2)

    report_path = os.path.join(results_dir, f"{prefix}_report.txt")
    with open(report_path, "w") as f:
        f.write(format_report(metrics, f"{prefix.upper()} EVALUATION"))
    print(f"  Saved {prefix} results -> '{json_path}'")
    print(f"  Saved {prefix} report  -> '{report_path}'")
