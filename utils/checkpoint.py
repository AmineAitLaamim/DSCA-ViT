"""Model checkpointing utilities."""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    val_acc: float,
    save_path: str,
    **extra_info: Any
) -> None:
    """Saves a model checkpoint.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (Optional[Any]): The learning rate scheduler.
        epoch (int): Current epoch number.
        val_acc (float): Validation accuracy at the current epoch.
        save_path (str): File path to save the checkpoint.
        **extra_info: Any additional information to save in the checkpoint dictionary.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val_accuracy": val_acc,
    }
    checkpoint.update(extra_info)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved successfully at '{save_path}'")

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """Loads a model checkpoint.

    Args:
        path (str): File path to load the checkpoint from.
        model (nn.Module): The PyTorch model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to load state into.
        scheduler (Optional[Any]): The scheduler to load state into.
        device (torch.device): Device to map the loaded tensors to.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
        
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint
