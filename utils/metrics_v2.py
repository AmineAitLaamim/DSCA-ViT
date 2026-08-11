# ============================================================
# DSCA-ViT v2 — Metrics
# ============================================================
#
# Metrics for the v2 experiment:
#   accuracy, balanced accuracy, macro F1,
#   per-class precision/recall/F1, confusion matrix
#
# The original utils/metrics.py is NOT modified.
# ============================================================

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics_v2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Computes the full v2 metric set.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class names (length = num_classes).

    Returns:
        dict with keys:
            accuracy, balanced_accuracy, macro_f1,
            per_class_precision, per_class_recall, per_class_f1,
            per_class_support, confusion_matrix, class_names
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    macro_f1 = float(np.mean(f1))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None:
        class_names = [str(c) for c in labels]

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_support": supp,
        "confusion_matrix": cm,
        "class_names": class_names,
        "labels": labels,
    }


def print_metrics_v2(metrics: dict) -> None:
    """
    Pretty-print the v2 metric set.
    """
    print("=" * 60)
    print("Metrics (v2)")
    print("=" * 60)
    print(f"  Accuracy         : {metrics['accuracy']:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print("  Per-class (P/R/F1):")
    for i, cls in enumerate(metrics["class_names"]):
        print(
            f"    {cls}: P={metrics['per_class_precision'][i]:.4f} "
            f"R={metrics['per_class_recall'][i]:.4f} "
            f"F1={metrics['per_class_f1'][i]:.4f} "
            f"(n={metrics['per_class_support'][i]})"
        )
    print("  Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 60)