# ============================================================
# Plain ViT-B16 Baseline — Training Script (CLI, HPC/DDP-ready)
# ============================================================
#
# Reproduces the plain ViT-B16 baseline (95.02% test accuracy)
# on the HER2-IHC-40x dataset.
#
# Usage:
#   python utils/train_plain_vit_baseline.py --config configs/plain_vit_baseline_config.yaml
#   python utils/train_plain_vit_baseline.py --config configs/plain_vit_baseline_config.yaml --resume
#   python utils/train_plain_vit_baseline.py --config configs/plain_vit_baseline_config.yaml --debug
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
from models_baseline import PlainViTB16
from utils.metrics_dss_vit import compute_metrics
from utils.split_utils import get_or_create_split_indices


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Sets up console + file logging. Only rank 0 writes to file."""
    logger = logging.getLogger("plain_vit_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

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
    """Initializes the distributed process group if DDP is requested."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if force_distributed and world_size == 1:
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
# Freeze / optimizer helpers
# ------------------------------------------------------------
def set_stage_requires_grad(
    model: nn.Module,
    stage: int,
    config: dict,
) -> None:
    """
    Sets requires_grad per stage.

    Stage 1: freeze backbone, train head only.
    Stage 2: train everything.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if stage == 1:
        # Head only trainable
        for name, p in model.vit.named_parameters():
            if name.startswith("head."):
                p.requires_grad = True
    elif stage == 2:
        # Everything trainable
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_optimizer(
    model: nn.Module,
    config: dict,
    stage: int,
) -> torch.optim.Optimizer:
    """
    Builds the AdamW optimizer for the given stage.

    Weight decay applied to weights only; bias/LayerNorm → 0.
    """
    wd = config["training"]["weight_decay"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]

    if stage == 1:
        # Head only
        head_params_decay = []
        head_params_no_decay = []
        for name, p in model.vit.named_parameters():
            if not name.startswith("head."):
                continue
            if p.ndim <= 1:
                head_params_no_decay.append(p)
            else:
                head_params_decay.append(p)

        param_groups = []
        if head_params_decay:
            param_groups.append({
                "params": head_params_decay,
                "lr": stage1["lr"],
                "weight_decay": wd,
                "name": "head",
            })
        if head_params_no_decay:
            param_groups.append({
                "params": head_params_no_decay,
                "lr": stage1["lr"],
                "weight_decay": 0.0,
                "name": "head_norm",
            })
    elif stage == 2:
        # Backbone (slow) + head (faster), each with decay/no-decay split
        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []

        for name, p in model.vit.named_parameters():
            if name.startswith("head."):
                if p.ndim <= 1:
                    head_no_decay.append(p)
                else:
                    head_decay.append(p)
            else:
                if p.ndim <= 1:
                    backbone_no_decay.append(p)
                else:
                    backbone_decay.append(p)

        param_groups = []
        if backbone_decay:
            param_groups.append({
                "params": backbone_decay,
                "lr": stage2["backbone_lr"],
                "weight_decay": wd,
                "name": "backbone",
            })
        if backbone_no_decay:
            param_groups.append({
                "params": backbone_no_decay,
                "lr": stage2["backbone_lr"],
                "weight_decay": 0.0,
                "name": "backbone_norm",
            })
        if head_decay:
            param_groups.append({
                "params": head_decay,
                "lr": stage2["head_lr"],
                "weight_decay": wd,
                "name": "head",
            })
        if head_no_decay:
            param_groups.append({
                "params": head_no_decay,
                "lr": stage2["head_lr"],
                "weight_decay": 0.0,
                "name": "head_norm",
            })
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return torch.optim.AdamW(param_groups)


# ------------------------------------------------------------
# MixUp
# ------------------------------------------------------------
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies MixUp augmentation."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


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
) -> Tuple[float, float]:
    """Trains one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_total = 0.0
    correct = 0
    total = 0

    mixup_alpha = config["training"]["mixup_alpha"]
    use_mixup = mixup_alpha > 0.0
    label_smoothing = config["training"]["label_smoothing"]

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
            logits = pred["logits"]

            if use_mixup:
                ce_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                ce_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                loss = lam * ce_a + (1.0 - lam) * ce_b
            else:
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=config["training"]["gradient_clip"],
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=config["training"]["gradient_clip"],
            )
            optimizer.step()

        batch_size = images.size(0)
        running_total += loss.item() * batch_size

        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Stopping after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return running_total / n, 100.0 * correct / n


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    logger: logging.Logger,
    rank: int,
    debug: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validates one epoch. Returns (avg_loss, accuracy%, preds, labels)."""
    model.eval()
    running_total = 0.0
    correct = 0
    total = 0
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred = model(images)
            loss = F.cross_entropy(
                pred["logits"],
                labels,
                label_smoothing=config["training"]["label_smoothing"],
            )

        batch_size = images.size(0)
        running_total += loss.item() * batch_size

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
    # weights_only=False: numpy arrays in metrics dict.
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    state_model = model.module if hasattr(model, "module") else model
    state_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ------------------------------------------------------------
# Sanity check (used by --debug to verify shapes/backward)
# ------------------------------------------------------------
def run_sanity_check(model, device, image_size, config):
    """Verifies forward/backward, shapes, no NaN/Inf, prints counts."""
    print("=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    model.eval()

    x_check = torch.rand(2, 3, image_size, image_size).to(device)
    with torch.no_grad():
        pred = model(x_check)
        print(f"  Input RGB    : {tuple(x_check.shape)}")
        print(f"  Logits       : {tuple(pred['logits'].shape)}")
        print(f"  Probs        : {tuple(pred['probs'].shape)}")

    # Backward check
    model.train()
    x_back = x_check.clone().requires_grad_(True)
    pred = model(x_back)
    loss = F.cross_entropy(
        pred["logits"],
        torch.zeros(2, dtype=torch.long).to(device),
        label_smoothing=config["training"]["label_smoothing"],
    )
    loss.backward()
    print(f"  loss.backward() : OK (loss={loss.item():.4f})")

    # NaN/Inf/dtype/device
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"NaN/Inf in parameter: {name}")
        if p.dtype != torch.float32:
            raise RuntimeError(f"Unexpected dtype for {name}: {p.dtype}")
        if str(p.device) != str(device):
            raise RuntimeError(f"Unexpected device for {name}: {p.device}")

    counts = model.count_parameters()
    print("=" * 60)
    print("Plain ViT-B16 Parameter Summary")
    print("=" * 60)
    for k, v in counts.items():
        print(f"  {k:<12} : {v:>12,}")
    print("=" * 60)
    print("✅ Sanity check passed.")


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def run_training(config: dict, resume: bool, debug: bool, force_distributed: bool = False) -> None:
    # DDP setup
    rank, world_size, local_rank = setup_ddp(force_distributed=force_distributed)
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Logging
    log_dir = config["paths"]["log_dir"]
    logger = setup_logging(log_dir, rank)
    logger.info(f"Rank {rank}/{world_size} | Device: {device}")

    # Paths
    train_dir = config["paths"]["train_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    split_indices_path = config["paths"]["split_indices_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    results_dir = config["paths"]["results_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Experiment metadata (rank 0)
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
        print(f"Experiment metadata written to '{_meta_path}'")

    # Build model
    model = PlainViTB16(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    # Parameter counts + sanity check (rank 0)
    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("Plain ViT-B16 Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<22} : {count:>12,}")
        logger.info("=" * 60)

    if debug and rank == 0:
        run_sanity_check(model, device, config["model"]["image_size"], config)

    # Data
    image_size = config["model"]["image_size"]
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

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # AMP
    amp_config = config["training"].get("amp", True)
    use_amp = amp_config and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    # Stages
    stages = [
        (1, config["stage1"]["epochs"]),
        (2, config["stage2"]["epochs"]),
    ]

    start_stage = 1
    start_epoch = 0
    best_val_acc = 0.0
    best_metrics = {}

    if resume:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from '{resume_path}'")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            start_stage = ckpt.get("stage", 1)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("metrics", {}).get("accuracy", 0.0)
            best_metrics = ckpt.get("metrics", {})

            state_model = model.module if hasattr(model, "module") else model
            state_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Resumed at stage {start_stage}, epoch {start_epoch}, "
                f"best val acc {best_val_acc:.4f}"
            )
        else:
            logger.warning(f"No checkpoint found at '{resume_path}'. Starting fresh.")

    # Training loop
    for stage, stage_epochs in stages:
        if stage < start_stage:
            logger.info(f"Skipping stage {stage} (resumed at stage {start_stage})")
            continue

        set_stage_requires_grad(model, stage, config)
        optimizer = build_optimizer(model, config, stage)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )

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

            train_loss, train_acc = train_one_epoch(
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

            if rank == 0:
                val_loss, val_acc, preds, labels = validate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    config=config,
                    logger=logger,
                    rank=rank,
                    debug=debug,
                )

                class_names = full_train_dataset.get_class_names()
                metrics = compute_metrics(labels, preds, class_names)
                metrics["val_loss"] = val_loss
                metrics["train_loss"] = train_loss
                metrics["train_acc"] = train_acc
                metrics["epoch"] = epoch + 1
                metrics["stage"] = stage

                elapsed = time.time() - epoch_start
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
                    f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
                    f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
                    f"QWK {metrics['qwk']:.4f} | {elapsed:.1f}s"
                )

                metrics_path = os.path.join(log_dir, "metrics.jsonl")
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                checkpoint_file = "best_stage1.pt" if stage == 1 else "best_stage2.pt"
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = metrics
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        stage=stage,
                        metrics=metrics,
                        config=config,
                        split_indices_path=split_indices_path,
                        save_path=os.path.join(checkpoint_dir, checkpoint_file),
                        rank=rank,
                    )
                    logger.info(f"  ✅ New best stage {stage} (val acc {val_acc:.2f}%)")

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
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
                    f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}%"
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
    parser = argparse.ArgumentParser(description="Train Plain ViT-B16 Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/plain_vit_baseline_config.yaml",
        help="Path to the baseline config file.",
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