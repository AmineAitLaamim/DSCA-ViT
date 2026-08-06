"""Evaluation utility functions."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List

def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Validates the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to validate.
        dataloader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run the validation on.

    Returns:
        Tuple[float, float, List[int], List[int]]: 
            - Average loss for the epoch
            - Accuracy (0-100) for the epoch
            - List of all predictions
            - List of all ground truth labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels
