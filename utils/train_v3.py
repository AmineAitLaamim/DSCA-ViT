# ============================================================
# DSCA-ViT v3 - Training Utilities
# ============================================================
#
# Provides:
#   train_one_epoch_v3
#   validate_one_epoch_v3
#   set_stage_requires_grad
#   set_stage_lrs
#   clip_grads
#   collect_telemetry_v3
#   snapshot_initial_params_v3
#   save_stage_checkpoint
#
# Locked v3 training design (identical to v2):
#   - one persistent AdamW
#   - 3 stages with requires_grad freezing
#   - per-stage CosineAnnealingLR (LRs changed before scheduler creation)
#   - weight decay 0.05 on weights only
#   - gradient clipping max_norm=1.0
#   - no AMP
#
# The original utils/train.py and utils/train_v2.py are NOT modified.
# ============================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint


def train_one_epoch_v3(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader for the training dataset.
        criterion: The loss function.
        optimizer: The optimizer.
        device: Device to run the training on.
        gradient_clip: Max norm for gradient clipping (0 = disabled).

    Returns:
        Tuple[float, float]: Average loss and accuracy (0-100) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # --------------------------------------------------------
        # Gradient clipping (only parameters that currently have gradients)
        # --------------------------------------------------------
        if gradient_clip > 0.0:
            params_with_grad = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=gradient_clip)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch_v3(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """
    Validates the model for one epoch.

    Returns:
        Tuple[float, float, List[int], List[int]]:
            (epoch_loss, epoch_acc, all_predictions, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

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


def set_stage_requires_grad(model: nn.Module, stage: int) -> None:
    """
    Sets requires_grad for every parameter group per the locked v3 spec.

    Stage 1: train input_modules + fusion_modules + classifier;
             freeze vit and existing_dsca.
    Stage 2: train input_modules + existing_dsca + fusion_modules + classifier;
             freeze vit.
    Stage 3: everything trainable.
    """
    groups = model.get_parameter_groups()

    # Default: freeze everything, then unfreeze per stage
    for name, params in groups.items():
        for p in params:
            p.requires_grad = False

    if stage == 1:
        for p in groups["input_modules"]:
            p.requires_grad = True
        for p in groups["fusion_modules"]:
            p.requires_grad = True
        for p in groups["classifier"]:
            p.requires_grad = True
    elif stage == 2:
        for p in groups["input_modules"]:
            p.requires_grad = True
        for p in groups["existing_dsca"]:
            p.requires_grad = True
        for p in groups["fusion_modules"]:
            p.requires_grad = True
        for p in groups["classifier"]:
            p.requires_grad = True
    elif stage == 3:
        for name, params in groups.items():
            for p in params:
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def set_stage_lrs(
    optimizer: torch.optim.Optimizer,
    stage: int,
    stage_config: dict,
) -> None:
    """
    Sets the parameter-group learning rates for the given stage.

    Must be called BEFORE creating the new CosineAnnealingLR so that
    base_lrs are correct.

    Locked v3 spec:
        Stage 1: input_modules=2e-4, fusion_modules=2e-4, classifier=1e-5
        Stage 2: input_modules=1e-4, existing_dsca=5e-5,
                 fusion_modules=2e-4, classifier=1e-4
        Stage 3: vit=1e-5, existing_dsca=5e-5, input_modules=1e-4,
                 fusion_modules=1e-4, classifier=1e-4
    """
    group_names = [
        "vit",
        "existing_dsca",
        "input_modules",
        "fusion_modules",
        "classifier",
    ]

    # The optimizer may contain 5 groups (one per architecture group) or
    # 10 groups (split into decay/no-decay per architecture group). Each
    # param group must carry a "name" key identifying its architecture
    # group. We match by name, not by position.
    for param_group in optimizer.param_groups:
        name = param_group.get("name", None)
        if name not in group_names:
            raise RuntimeError(
                "Optimizer param group is missing a valid 'name' key. "
                f"Expected one of {group_names}, got {name!r}. "
                "When building the optimizer, add name=<group_name> to "
                "each param group dict."
            )

        if name == "vit":
            lr = stage_config.get("vit_lr", 0.0 if stage < 3 else 1.0e-5)
        elif name == "existing_dsca":
            lr = stage_config.get("existing_lr", 0.0 if stage == 1 else 5.0e-5)
        elif name == "input_modules":
            lr = stage_config.get("input_lr", 2.0e-4 if stage == 1 else 1.0e-4)
        elif name == "fusion_modules":
            lr = stage_config.get(
                "fusion_lr",
                0.0 if stage == 1 else 2.0e-4 if stage == 2 else 1.0e-4,
            )
        elif name == "classifier":
            lr = stage_config.get("classifier_lr", 1.0e-5 if stage == 1 else 1.0e-4)

        param_group["lr"] = lr
        param_group["initial_lr"] = lr


def clip_grads(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clips gradients of parameters that currently have gradients.

    Returns:
        float: The total gradient norm before clipping.
    """
    params_with_grad = [
        p for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not params_with_grad:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=max_norm)
    return float(total_norm)


def _module_names_v3() -> list[str]:
    """New v3 trainable modules for telemetry."""
    return [
        "norm_h",
        "norm_dab",
        "adapter_h",
        "adapter_dab",
        "channel_affine_h",
        "channel_affine_dab",
        "interaction",
        "stain_gate",
        "scale_gate",
    ]


def collect_telemetry_v3(
    model: nn.Module,
    initial_state: Optional[dict[str, torch.Tensor]] = None,
    delta_h: Optional[torch.Tensor] = None,
    delta_d: Optional[torch.Tensor] = None,
    stain_gate: Optional[torch.Tensor] = None,
    scale_gate: Optional[torch.Tensor] = None,
) -> dict:
    """
    Collects lightweight telemetry for the new v3 modules.

    Per module (norm_h, norm_dab, adapter_h, adapter_dab,
    channel_affine_h, channel_affine_dab, interaction, stain_gate,
    scale_gate):
        grad_norm          : ||gradient||
        parameter_delta    : ||p_current - p_initial||
        relative_delta     : delta / (||p_initial|| + eps)

    Also optionally records:
        mean(|delta_H|), mean(|delta_D|)          (interaction outputs)
        stain gate mean/std/min/max/median
        scale gate mean/std/min/max/median

    Args:
        model: The v3 model.
        initial_state: Optional snapshot of initial parameters
                       (module_name -> concatenated flat tensor).
        delta_h: Optional tensor of delta_H (interaction output telemetry).
        delta_d: Optional tensor of delta_D.
        stain_gate: Optional tensor of stain gate values (B, N, C).
        scale_gate: Optional tensor of scale gate values (B, N, C).
    """
    eps = 1e-8
    telemetry = {}

    for name in _module_names_v3():
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        grad_norm = (
            torch.sqrt(sum(g.flatten().norm() ** 2 for g in grads)).item()
            if grads
            else 0.0
        )

        flat = torch.cat([p.detach().flatten() for p in module.parameters()])
        param_norm = flat.norm().item()

        if initial_state is not None and name in initial_state:
            delta = (flat - initial_state[name].to(flat.device)).norm().item()
            relative = delta / (initial_state[name].norm().item() + eps)
        else:
            delta = 0.0
            relative = 0.0

        telemetry[name] = {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "parameter_delta": delta,
            "relative_delta": relative,
        }

    if delta_h is not None:
        telemetry["interaction_d_to_h_output_norm"] = float(delta_h.abs().mean().item())
    if delta_d is not None:
        telemetry["interaction_h_to_d_output_norm"] = float(delta_d.abs().mean().item())

    if stain_gate is not None:
        sg = stain_gate.detach().cpu().numpy()
        telemetry["stain_gate_mean"] = float(sg.mean())
        telemetry["stain_gate_std"] = float(sg.std())
        telemetry["stain_gate_min"] = float(sg.min())
        telemetry["stain_gate_max"] = float(sg.max())
        telemetry["stain_gate_median"] = float(np.median(sg))

    if scale_gate is not None:
        scg = scale_gate.detach().cpu().numpy()
        telemetry["scale_gate_mean"] = float(scg.mean())
        telemetry["scale_gate_std"] = float(scg.std())
        telemetry["scale_gate_min"] = float(scg.min())
        telemetry["scale_gate_max"] = float(scg.max())
        telemetry["scale_gate_median"] = float(np.median(scg))

    return telemetry


def snapshot_initial_params_v3(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Snapshots initial parameters of the new v3 modules for telemetry.

    Returns:
        dict: module_name -> concatenated flat CPU tensor.
    """
    snapshot = {}
    for name in _module_names_v3():
        module = getattr(model, name)
        flat = torch.cat(
            [p.detach().cpu().flatten() for p in module.parameters()]
        )
        snapshot[name] = flat
    return snapshot


def save_stage_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    stage: int,
    val_acc: float,
    save_path: str,
    config: dict,
    seed: int,
    split_indices_path: str,
) -> None:
    """
    Saves a checkpoint with full reproducibility info per the locked spec.
    """
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        val_acc=val_acc,
        save_path=save_path,
        stage=stage,
        config=config,
        seed=seed,
        split_indices_path=split_indices_path,
    )