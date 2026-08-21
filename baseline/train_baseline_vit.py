# ============================================================
# Plain ViT-B16 Baseline — Training Script (CLI, HPC/DDP-ready)
# ============================================================
#
# Reproduces the plain ViT-B16 baseline (95.02% test accuracy)
# on the HER2-IHC-40x dataset.
#
# Usage:
#   python baseline/train_baseline_vit.py --config configs/plain_vit_baseline_config.yaml
#   python baseline/train_baseline_vit.py --config configs/plain_vit_baseline_config.yaml --resume
#   python baseline/train_baseline_vit.py --config configs/plain_vit_baseline_config.yaml --debug
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

# Add project root to path (so `from baseline import ...` works)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline import (
    PlainViTB16,
    compute_metrics,
    HER2BaselineDataset,
    get_train_transform,
    get_test_transform,
    get_or_create_split_indices,
)


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("plain_vit_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    return logger


# ------------------------------------------------------------
# DDP helpers
# ------------------------------------------------------------
def setup_ddp(force_distributed: bool = False) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if force_distributed and world_size == 1:
        world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ------------------------------------------------------------
# Freeze / optimizer helpers
# ------------------------------------------------------------
def set_stage_requires_grad(model: nn.Module, stage: int) -> None:
    """Stage 1: freeze backbone, train head only. Stage 2: train all."""
    for p in model.parameters():
        p.requires_grad = False
    if stage == 1:
        for name, p in model.vit.named_parameters():
            if name.startswith("head."):
                p.requires_grad = True
    elif stage == 2:
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_optimizer(model: nn.Module, config: dict, stage: int) -> torch.optim.Optimizer:
    """Builds the optimizer for the given stage.

    Matches the original Colab notebook:
      - Stage 1: Adam on head params only, lr=1e-4
      - Stage 2: Adam with backbone lr=1e-5, head lr=1e-4
    """
    optimizer_name = config["training"].get("optimizer", "Adam")
    wd = config["training"].get("weight_decay", 0.0)
    stage1 = config["stage1"]
    stage2 = config["stage2"]

    def _split(params):
        decay, no_decay = [], []
        for p in params:
            if p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    if stage == 1:
        head_params = [p for n, p in model.vit.named_parameters() if n.startswith("head.")]
        hd, hnd = _split(head_params)
        groups = []
        if hd:
            groups.append({"params": hd, "lr": stage1["lr"], "weight_decay": wd})
        if hnd:
            groups.append({"params": hnd, "lr": stage1["lr"], "weight_decay": 0.0})
    elif stage == 2:
        backbone = [p for n, p in model.vit.named_parameters() if not n.startswith("head.")]
        head = [p for n, p in model.vit.named_parameters() if n.startswith("head.")]
        bd, bnd = _split(backbone)
        hd, hnd = _split(head)
        groups = []
        if bd:
            groups.append({"params": bd, "lr": stage2["backbone_lr"], "weight_decay": wd})
        if bnd:
            groups.append({"params": bnd, "lr": stage2["backbone_lr"], "weight_decay": 0.0})
        if hd:
            groups.append({"params": hd, "lr": stage2["head_lr"], "weight_decay": wd})
        if hnd:
            groups.append({"params": hnd, "lr": stage2["head_lr"], "weight_decay": 0.0})
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Use Adam (as in the original notebook) unless explicitly set to AdamW
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(groups)
    return torch.optim.Adam(groups)


# ------------------------------------------------------------
# Training / validation loops
# ------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, scaler, device, config, logger, rank, debug=False
) -> Tuple[float, float]:
    model.train()
    running_total = 0.0
    correct = 0
    total = 0
    use_mixup = config["training"]["mixup_alpha"] > 0.0
    label_smoothing = config["training"]["label_smoothing"]

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            alpha = config["training"]["mixup_alpha"]
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(images.size(0), device=images.device)
            images = lam * images + (1.0 - lam) * images[index]
            y_a, y_b = labels, labels[index]
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

        clip = config["training"].get("gradient_clip", 0.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    max_norm=clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    max_norm=clip,
                )
            optimizer.step()

        bs = images.size(0)
        running_total += loss.item() * bs
        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Stopping after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return running_total / n, 100.0 * correct / n


@torch.no_grad()
def validate_one_epoch(
    model, dataloader, device, config, logger, rank, debug=False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_total = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images)
        loss = F.cross_entropy(
            pred["logits"], labels,
            label_smoothing=config["training"]["label_smoothing"],
        )
        bs = images.size(0)
        running_total += loss.item() * bs
        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Validation stopped after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return (
        running_total / n,
        100.0 * correct / n,
        np.array(all_preds),
        np.array(all_labels),
    )


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------
def save_checkpoint(
    model, optimizer, scheduler, epoch, stage, metrics, config,
    split_indices_path, save_path, rank,
) -> None:
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "model_state_dict": state_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "config": config,
        "split_indices_path": split_indices_path,
    }
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to '{save_path}'")


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def run_training(config: dict, resume: bool, debug: bool, force_distributed: bool = False) -> None:
    rank, world_size, local_rank = setup_ddp(force_distributed=force_distributed)
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_dir = config["paths"]["log_dir"]
    logger = setup_logging(log_dir, rank)
    logger.info(f"Rank {rank}/{world_size} | Device: {device}")

    train_dir = config["paths"]["train_dir"]
    test_dir = config["paths"]["test_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    split_indices_path = config["paths"]["split_indices_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    results_dir = config["paths"]["results_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

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

    model = PlainViTB16(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("Plain ViT-B16 Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<22} : {count:>12,}")
        logger.info("=" * 60)

    image_size = config["model"]["image_size"]
    train_transform = get_train_transform(image_size=image_size)
    test_transform = get_test_transform(image_size=image_size)

    val_fraction = config["dataset"]["val_fraction"]

    # Always create/save the shared split_indices.npz (train/val/test)
    # so ALL future models reuse the exact same split.
    train_indices, val_indices = get_or_create_split_indices(
        train_dir=train_dir,
        test_dir=test_dir,
        val_fraction=val_fraction,
        seed=config["dataset"]["val_seed"],
        save_path=split_indices_path,
    )

    full_train_dataset = HER2BaselineDataset(root_dir=train_dir, transform=train_transform)

    if val_fraction > 0.0:
        # Split a validation holdout from the training set
        val_dataset = HER2BaselineDataset(root_dir=train_dir, transform=test_transform)
        train_dataset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
    else:
        # val_fraction == 0.0 → validate on the official TEST set (as in the notebook)
        train_dataset = full_train_dataset
        val_dataset = HER2BaselineDataset(root_dir=test_dir, transform=test_transform)
        val_subset = val_dataset
        val_indices = np.arange(len(val_dataset))
        logger.info("Validation on the official TEST set (val_fraction=0.0)")

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_indices)} | "
        f"Batch (per GPU): {batch_size}"
    )

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    amp_config = config["training"].get("amp", True)
    use_amp = amp_config and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

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

    for stage, stage_epochs in stages:
        if stage < start_stage:
            logger.info(f"Skipping stage {stage} (resumed at stage {start_stage})")
            continue

        set_stage_requires_grad(model, stage)
        optimizer = build_optimizer(model, config, stage)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)

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
                model=model, dataloader=train_loader, optimizer=optimizer,
                scaler=scaler, device=device, config=config,
                logger=logger, rank=rank, debug=debug,
            )

            scheduler.step()

            if rank == 0:
                val_loss, val_acc, preds, labels = validate_one_epoch(
                    model=model, dataloader=val_loader, device=device,
                    config=config, logger=logger, rank=rank, debug=debug,
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
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch + 1, stage=stage, metrics=metrics,
                        config=config, split_indices_path=split_indices_path,
                        save_path=os.path.join(checkpoint_dir, checkpoint_file),
                        rank=rank,
                    )
                    logger.info(f"  ✅ New best stage {stage} (val acc {val_acc:.2f}%)")

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch + 1, stage=stage, metrics=metrics,
                    config=config, split_indices_path=split_indices_path,
                    save_path=os.path.join(checkpoint_dir, "last.pt"),
                    rank=rank,
                )
            else:
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
                    f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}%"
                )

        if rank == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=stage_epochs, stage=stage, metrics=best_metrics,
                config=config, split_indices_path=split_indices_path,
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
    parser.add_argument("--config", type=str, default="configs/plain_vit_baseline_config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.debug:
        config["training"]["debug"] = True
    debug = config["training"]["debug"]
    run_training(config, resume=args.resume, debug=debug, force_distributed=args.distributed)


if __name__ == "__main__":
    main()