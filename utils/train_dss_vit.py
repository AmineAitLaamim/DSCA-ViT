# ============================================================
# DSS-ViT — Training Script (CLI, HPC/DDP-ready)
# ============================================================
#
# Usage:
#   python utils/train_dss_vit.py --config configs/dss_vit_config.yaml
#   python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --resume
#   python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug
#
# DDP: auto-detects LOCAL_RANK/WORLD_SIZE/RANK env vars (set by
# SLURM srun). Use --distributed to force DDP.
# ============================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import HER2Dataset, get_train_transform, get_test_transform
from models_v2_1 import DSSViT, load_stain_stats, total_loss
from utils.metrics_dss_vit import compute_metrics
from utils.split_utils import get_or_create_split_indices


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Sets up console + file logging. Only rank 0 writes to file."""
    logger = logging.getLogger("dss_vit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler (all ranks)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    # File handler (rank 0 only)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


# ------------------------------------------------------------
# DDP helpers
# ------------------------------------------------------------
def setup_ddp(force_distributed: bool = False) -> Tuple[int, int, int]:
    """
    Initializes the distributed process group if DDP is requested.

    Args:
        force_distributed (bool): Force DDP even without env vars.

    Returns:
        Tuple[int, int, int]: (rank, world_size, local_rank).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if force_distributed and world_size == 1:
        # Single-node multi-GPU fallback: use all visible GPUs
        world_size = torch.cuda.device_count()
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    """Destroys the distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ------------------------------------------------------------
# Parameter group / optimizer helpers
# ------------------------------------------------------------
def build_optimizer(
    model: nn.Module,
    config: dict,
    stage: int,
) -> torch.optim.Optimizer:
    """
    Builds the AdamW optimizer with named parameter groups and
    per-stage learning rates. Weight decay applied to weights only.

    Args:
        model: The DSSViT model.
        config: The full config dict.
        stage: Current stage (1, 2, or 3).

    Returns:
        torch.optim.Optimizer: The AdamW optimizer.
    """
    groups = model.get_parameter_groups()
    wd = config["training"]["weight_decay"]

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    stage3 = config["stage3"]

    # Determine per-group LR for this stage
    if stage == 1:
        lrs = {
            "vit": 0.0,  # frozen
            "stain_encoder": stage1["lr"],
            "cross_fusion_gate": stage1["lr"],
            "ordinal_head": stage1["lr"],
        }
    elif stage == 2:
        lrs = {
            "vit": 0.0,  # frozen
            "stain_encoder": stage2["lr"],
            "cross_fusion_gate": stage2["lr"],
            "ordinal_head": stage2["lr"],
        }
    elif stage == 3:
        lrs = {
            "vit": stage3["vit_lr"],
            "stain_encoder": stage3["new_lr"],
            "cross_fusion_gate": stage3["new_lr"],
            "ordinal_head": stage3["new_lr"],
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Build param groups with decay/no-decay split
    param_groups = []
    for name, params in groups.items():
        decay_params = []
        no_decay_params = []
        for p in params:
            if p.ndim <= 1:  # bias / LayerNorm / BatchNorm
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": lrs[name],
                "weight_decay": wd,
                "name": name,
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": lrs[name],
                "weight_decay": 0.0,
                "name": name,
            })

    return torch.optim.AdamW(param_groups)


def set_stage_requires_grad(model: nn.Module, stage: int, config: dict) -> None:
    """
    Sets requires_grad for each parameter group per the stage config.

    Args:
        model: The DSSViT model.
        stage: Current stage (1, 2, or 3).
        config: The full config dict.
    """
    groups = model.get_parameter_groups()

    # Freeze everything first
    for name, params in groups.items():
        for p in params:
            p.requires_grad = False

    # Determine which groups to unfreeze
    if stage == 1:
        unfreeze = ["stain_encoder", "cross_fusion_gate", "ordinal_head"]
    elif stage == 2:
        unfreeze = ["stain_encoder", "cross_fusion_gate", "ordinal_head"]
    elif stage == 3:
        unfreeze = ["vit", "stain_encoder", "cross_fusion_gate", "ordinal_head"]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    for name in unfreeze:
        for p in groups[name]:
            p.requires_grad = True


# ------------------------------------------------------------
# MixUp / CutMix
# ------------------------------------------------------------
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation.

    Returns:
        Tuple: (mixed_x, y_a, y_b, lam).
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    config: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the total loss with MixUp (CE + ordinal on both targets).

    Args:
        pred: Model output dict.
        y_a: First target labels.
        y_b: Second target labels.
        lam: MixUp lambda.
        config: Config dict.

    Returns:
        Tuple: (total, ce, ord).
    """
    logits = pred["logits"]
    probs = pred["probs"]

    alpha = config["training"]["ordinal_alpha"]
    label_smoothing = config["training"]["label_smoothing"]

    # CE on probs.log() for both targets
    ce_a = F.cross_entropy(probs.log(), y_a, label_smoothing=label_smoothing)
    ce_b = F.cross_entropy(probs.log(), y_b, label_smoothing=label_smoothing)
    ce = lam * ce_a + (1.0 - lam) * ce_b

    # Ordinal loss on both targets
    ord_a = total_loss(logits, y_a, probs, alpha=0.0, label_smoothing=0.0)[2]
    ord_b = total_loss(logits, y_b, probs, alpha=0.0, label_smoothing=0.0)[2]
    ord = lam * ord_a + (1.0 - lam) * ord_b

    total = ce + alpha * ord
    return total, ce, ord


# ------------------------------------------------------------
# Training / validation loops
# ------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    config: dict,
    epoch: int,
    logger: logging.Logger,
    rank: int,
    debug: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Trains the model for one epoch.

    Returns:
        Tuple: (avg_total_loss, avg_ce_loss, avg_ord_loss, accuracy).
    """
    model.train()
    running_total = 0.0
    running_ce = 0.0
    running_ord = 0.0
    correct = 0
    total = 0

    mixup_alpha = config["training"]["mixup_alpha"]
    use_mixup = mixup_alpha > 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            y_a = y_b = labels
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            pred = model(images)
            if use_mixup:
                total_loss_val, ce_loss, ord_loss = mixup_criterion(
                    pred, y_a, y_b, lam, config
                )
            else:
                total_loss_val, ce_loss, ord_loss = total_loss(
                    pred["logits"],
                    labels,
                    pred["probs"],
                    alpha=config["training"]["ordinal_alpha"],
                    label_smoothing=config["training"]["label_smoothing"],
                )

        if scaler is not None:
            scaler.scale(total_loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=config["training"]["gradient_clip"],
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=config["training"]["gradient_clip"],
            )
            optimizer.step()

        batch_size = images.size(0)
        running_total += total_loss_val.item() * batch_size
        running_ce += ce_loss.item() * batch_size
        running_ord += ord_loss.item() * batch_size

        # Accuracy (use probs argmax)
        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Stopping after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return (
        running_total / n,
        running_ce / n,
        running_ord / n,
        100.0 * correct / n,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    logger: logging.Logger,
    rank: int,
    debug: bool = False,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Validates the model for one epoch (rank 0 only).

    Returns:
        Tuple: (avg_total_loss, avg_ce_loss, avg_ord_loss, accuracy,
                all_predictions, all_labels).
    """
    model.eval()
    running_total = 0.0
    running_ce = 0.0
    running_ord = 0.0
    correct = 0
    total = 0
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred = model(images)
            total_loss_val, ce_loss, ord_loss = total_loss(
                pred["logits"],
                labels,
                pred["probs"],
                alpha=config["training"]["ordinal_alpha"],
                label_smoothing=config["training"]["label_smoothing"],
            )

        batch_size = images.size(0)
        running_total += total_loss_val.item() * batch_size
        running_ce += ce_loss.item() * batch_size
        running_ord += ord_loss.item() * batch_size

        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        all_predictions.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Validation stopped after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return (
        running_total / n,
        running_ce / n,
        running_ord / n,
        100.0 * correct / n,
        np.array(all_predictions),
        np.array(all_labels),
    )


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    stage: int,
    metrics: dict,
    config: dict,
    split_indices_path: str,
    save_path: str,
    rank: int,
) -> None:
    """Saves a checkpoint (rank 0 only)."""
    if rank != 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # If DDP, unwrap the model
    state_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": state_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "config": config,
        "split_indices_path": split_indices_path,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to '{save_path}'")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Loads a checkpoint into the model (and optionally optimizer/scheduler)."""
    # weights_only=False: checkpoints contain numpy arrays in the metrics
    # dict, which PyTorch 2.6+ rejects by default (weights_only=True).
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # If DDP, unwrap the model
    state_model = model.module if hasattr(model, "module") else model
    state_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def run_training(config: dict, resume: bool, debug: bool, force_distributed: bool = False) -> None:
    # ------------------------------------------------------------
    # DDP setup
    # ------------------------------------------------------------
    rank, world_size, local_rank = setup_ddp(force_distributed=force_distributed)
    is_distributed = world_size > 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    log_dir = config["paths"]["log_dir"]
    logger = setup_logging(log_dir, rank)
    logger.info(f"Rank {rank}/{world_size} | Device: {device}")

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    train_dir = config["paths"]["train_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    split_indices_path = config["paths"]["split_indices_path"]
    stain_stats_path = config["paths"]["stain_stats_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    results_dir = config["paths"]["results_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Write experiment metadata once at training start (rank 0 only)
    if rank == 0:
        import datetime
        import subprocess as _subprocess
        try:
            git_commit = _subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = "unknown"
        _meta = {
            "experiment_name": config["paths"]["experiment_name"],
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit,
            "seed": config["training"]["seed"],
            "config": config,
        }
        _meta_path = os.path.join(experiment_dir, "experiment_meta.json")
        with open(_meta_path, "w") as _f:
            json.dump(_meta, _f, indent=2)
        # logger not yet initialised at this point — print is intentional
        print(f"Experiment metadata written to '{_meta_path}'")

    # ------------------------------------------------------------
    # Load stain stats
    # ------------------------------------------------------------
    stain_stats = load_stain_stats(stain_stats_path)
    logger.info(f"Stain stats: {stain_stats}")

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = DSSViT(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        num_stain_tokens=config["model"]["num_stain_tokens"],
        stain_bottleneck_dim=config["model"]["stain_bottleneck_dim"],
        stain_stats=stain_stats,
        image_size=config["model"]["image_size"],
    ).to(device)

    # Parameter counts (rank 0)
    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("DSS-ViT Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<20} : {count:>12,}")
        logger.info("=" * 60)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    image_size = config["dataset"]["image_size"]
    train_transform = get_train_transform(image_size=image_size)
    test_transform = get_test_transform(image_size=image_size)

    train_indices, val_indices = get_or_create_split_indices(
        train_dir=train_dir,
        val_fraction=config["dataset"]["val_fraction"],
        seed=config["dataset"]["val_seed"],
        save_path=split_indices_path,
    )

    full_train_dataset = HER2Dataset(root_dir=train_dir, transform=train_transform)
    val_dataset = HER2Dataset(root_dir=train_dir, transform=test_transform)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # Training sampler (DistributedSampler if DDP)
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Validation: rank 0 only, no DistributedSampler
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_indices)} | "
        f"Batch (per GPU): {batch_size}"
    )

    # ------------------------------------------------------------
    # DDP wrap
    # ------------------------------------------------------------
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # ------------------------------------------------------------
    # AMP scaler
    # ------------------------------------------------------------
    # AMP is enabled by default (config amp: true) and automatically
    # activates when a GPU is available. Set amp: false to disable.
    amp_config = config["training"].get("amp", True)
    use_amp = amp_config and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    # ------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------
    stages = [
        (1, config["stage1"]["epochs"]),
        (2, config["stage2"]["epochs"]),
        (3, config["stage3"]["epochs"]),
    ]

    # Resume state
    start_stage = 1
    start_epoch = 0
    best_val_acc = 0.0
    best_metrics = {}

    if resume:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from '{resume_path}'")
            # Build optimizer for the stage in the checkpoint to load state
            # (we'll rebuild after loading; load state into a temp optimizer)
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            start_stage = ckpt.get("stage", 1)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("metrics", {}).get("accuracy", 0.0)
            best_metrics = ckpt.get("metrics", {})

            # Load model weights
            state_model = model.module if hasattr(model, "module") else model
            state_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Resumed at stage {start_stage}, epoch {start_epoch}, "
                f"best val acc {best_val_acc:.4f}"
            )
        else:
            logger.warning(f"No checkpoint found at '{resume_path}'. Starting fresh.")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for stage, stage_epochs in stages:
        if stage < start_stage:
            logger.info(f"Skipping stage {stage} (resumed at stage {start_stage})")
            continue

        # Set requires_grad for this stage
        set_stage_requires_grad(model, stage, config)

        # Rebuild optimizer (param groups must reference current tensors)
        optimizer = build_optimizer(model, config, stage)

        # Per-stage cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )

        # If resuming mid-stage, load optimizer/scheduler state
        if resume and stage == start_stage and start_epoch > 0:
            resume_path = os.path.join(checkpoint_dir, "last.pt")
            if os.path.exists(resume_path):
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.info("=" * 60)
        logger.info(f"Stage {stage} — {stage_epochs} epochs")
        logger.info("=" * 60)

        for epoch in range(start_epoch if stage == start_stage else 0, stage_epochs):
            if is_distributed:
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()

            train_total, train_ce, train_ord, train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                config=config,
                epoch=epoch,
                logger=logger,
                rank=rank,
                debug=debug,
            )

            scheduler.step()

            # Validation (rank 0 only)
            if rank == 0:
                val_total, val_ce, val_ord, val_acc, preds, labels = validate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    config=config,
                    logger=logger,
                    rank=rank,
                    debug=debug,
                )

                # Compute full metrics
                class_names = full_train_dataset.get_class_names()
                metrics = compute_metrics(labels, preds, class_names)
                metrics["val_total_loss"] = val_total
                metrics["val_ce_loss"] = val_ce
                metrics["val_ord_loss"] = val_ord
                metrics["train_total_loss"] = train_total
                metrics["train_ce_loss"] = train_ce
                metrics["train_ord_loss"] = train_ord
                metrics["train_acc"] = train_acc
                metrics["epoch"] = epoch + 1
                metrics["stage"] = stage

                elapsed = time.time() - epoch_start
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
                    f"Train Loss {train_total:.4f} (CE {train_ce:.4f}, Ord {train_ord:.4f}) | "
                    f"Train Acc {train_acc:.2f}% | "
                    f"Val Loss {val_total:.4f} | Val Acc {val_acc:.2f}% | "
                    f"QWK {metrics['qwk']:.4f} | {elapsed:.1f}s"
                )

                # Save metrics to CSV/JSON
                metrics_path = os.path.join(log_dir, "metrics.jsonl")
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                # Track best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = metrics
                    if stage == 3:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch + 1,
                            stage=stage,
                            metrics=metrics,
                            config=config,
                            split_indices_path=split_indices_path,
                            save_path=os.path.join(checkpoint_dir, "best_stage3.pt"),
                            rank=rank,
                        )
                        logger.info(f"  ✅ New best Stage 3 (val acc {val_acc:.2f}%)")

                # Save last checkpoint (periodic + end of stage)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    stage=stage,
                    metrics=metrics,
                    config=config,
                    split_indices_path=split_indices_path,
                    save_path=os.path.join(checkpoint_dir, "last.pt"),
                    rank=rank,
                )
            else:
                # Non-rank-0: just train, no validation
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
                    f"Train Loss {train_total:.4f} | Train Acc {train_acc:.2f}%"
                )

        # End of stage checkpoint
        if rank == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=stage_epochs,
                stage=stage,
                metrics=best_metrics,
                config=config,
                split_indices_path=split_indices_path,
                save_path=os.path.join(checkpoint_dir, f"stage{stage}_end.pt"),
                rank=rank,
            )
            logger.info(f"Stage {stage} complete. Checkpoint saved.")

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}%")
        logger.info(f"Best QWK: {best_metrics.get('qwk', 0.0):.4f}")
        logger.info("=" * 60)

    cleanup_ddp()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DSS-ViT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dss_vit_config.yaml",
        help="Path to the DSS-ViT config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a few batches for quick testing.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Force DDP (auto-detected via env vars otherwise).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override debug from config if not set on CLI
    if args.debug:
        config["training"]["debug"] = True
    debug = config["training"]["debug"]

    run_training(
        config,
        resume=args.resume,
        debug=debug,
        force_distributed=args.distributed,
    )


if __name__ == "__main__":
    main()