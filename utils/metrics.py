"""Metrics computation and printing utilities."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import List, Dict, Any

def compute_metrics(
    labels: List[int],
    predictions: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes various classification metrics.

    Args:
        labels (List[int]): Ground truth labels.
        predictions (List[int]): Model predictions.
        class_names (List[str]): List of class names corresponding to label indices.

    Returns:
        Dict[str, Any]: Dictionary containing computed metrics:
            - accuracy: float (percentage, 0-100)
            - precision: float (weighted average)
            - recall: float (weighted average)
            - f1: float (weighted average)
            - classification_report: str
            - confusion_matrix: np.ndarray
    """
    accuracy = accuracy_score(labels, predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    report = classification_report(
        labels, predictions, target_names=class_names, zero_division=0
    )
    
    conf_matrix = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }

def print_metrics(metrics_dict: Dict[str, Any]) -> None:
    """Pretty prints the computed metrics.

    Args:
        metrics_dict (Dict[str, Any]): Dictionary of metrics returned by compute_metrics.
    """
    print("-" * 50)
    print("Classification Metrics:")
    print("-" * 50)
    print(f"Accuracy:  {metrics_dict.get('accuracy', 0.0):.4f}%")
    print(f"Precision: {metrics_dict.get('precision', 0.0):.4f} (Weighted Avg)")
    print(f"Recall:    {metrics_dict.get('recall', 0.0):.4f} (Weighted Avg)")
    print(f"F1 Score:  {metrics_dict.get('f1', 0.0):.4f} (Weighted Avg)")
    print("\nClassification Report:")
    print(metrics_dict.get("classification_report", "N/A"))
    print("Confusion Matrix:")
    print(metrics_dict.get("confusion_matrix", "N/A"))
    print("-" * 50)
