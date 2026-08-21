from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_specificity(cm: np.ndarray) -> dict[str, float]:
    total = cm.sum()
    values = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        values[str(i)] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    values["macro"] = float(np.mean(list(values.values())))
    return values


def compute_metrics(y_true, y_pred, y_prob, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    y_bin = label_binarize(y_true, classes=labels)
    roc_auc = None
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "roc_auc_ovr_macro": roc_auc,
        "specificity": compute_specificity(cm),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
    }


def curve_data(y_true, y_prob, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=labels)
    curves = {"roc": {}, "pr": {}}
    for i, name in enumerate(class_names):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        curves["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        curves["pr"][name] = {"precision": precision.tolist(), "recall": recall.tolist()}
    return curves

