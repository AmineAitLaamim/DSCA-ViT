# ============================================================
# Plain ViT-B16 Baseline — Metrics
# ============================================================
#
# Provides accuracy, balanced accuracy, macro-F1, weighted-F1,
# quadratic weighted kappa (QWK), per-class metrics, and
# confusion matrix computation.
# ============================================================

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute a comprehensive set of classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        num_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
        class_names = [f"class_{i}" for i in range(num_classes)]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Quadratic weighted kappa (ordinal agreement)
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

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

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


def print_metrics(metrics: Dict) -> None:
    """Print a formatted summary of the metrics."""
    print("=" * 60)
    print("Classification Metrics")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"QWK               : {metrics['qwk']:.4f}")
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