# ============================================================
# UNI-Regularized — Training Script (CLI, HPC/DDP-ready)
# ============================================================
#
# Same 2-stage structure as the UNI-baseline, with two
# deliberate changes:
#   1. AdamW instead of Adam
#   2. Patience-based early stopping on validation loss
#
#   Stage 1 (max 30 ep): frozen UNI backbone, train head only,
#       AdamW lr=1e-4, wd=0.0 (both head groups).
#   Stage 2 (max 30 ep): full fine-tuning,
#       AdamW backbone 1e-5 (wd 0.05 non-1D / 0.0 1D),
#             head 1e-4 (wd 0.05 non-1D / 0.0 1D).
#
#   - Optimizer : AdamW (decay/no-decay split by param.ndim <= 1)
#   - Loss      : CrossEntropyLoss (no label smoothing)
#   - Batch size: 32 (per GPU)
#   - AMP: disabled
#   - Gradient clipping: none
#   - Scheduler : CosineAnnealingLR(T_max=stage_epochs), recreated per stage
#   - Early stopping: patience 7, min_delta 1e-4 on VAL LOSS
#       (counters reset at the Stage 1 -> Stage 2 transition)
#   - Checkpoint selection: best by VAL ACCURACY (unchanged)
#
# Stage 2 is initialized from best_stage1.pt (NOT last.pt).
# The original UNI-baseline continues Stage 2 from the final
# Stage-1 epoch weights; this build corrects that.
#
# The split file is LOAD-ONLY (never regenerated):
#   .../plain_vit_baseline_001/split_indices_wsi.npz
#   (train 7283 | val 810 | test 1847, val_fraction=0.10, seed=42)
#
# Usage:
#   uv run python UNI-Regularized/train_uni_regularized.py \
#       --config configs/uni_regularized_config.yaml
#   ... --resume
#   ... --debug
# ============================================================

from __future__ import annotations

import os

# Force any accidental Hugging Face network access to fail loudly.
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import logging
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# Add project root to path so `from baseline import ...` works.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from baseline import HER2BaselineDataset, compute_metrics
except ImportError:
    # Self-contained fallback (keeps the hyphen folder usable standalone).
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
        cohen_kappa_score,
    )

    CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]

    class HER2BaselineDataset(Dataset):
        """Fallsback class-folder dataset (identical layout to baseline)."""

        def __init__(self, root_dir: str, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = CLASSES
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.image_paths: List[Path] = []
            self.labels: List[int] = []
            _exts = {".png", ".jpg", ".jpeg"}
            if not self.root_dir.exists():
                raise ValueError(f"Dataset root not found: {self.root_dir}")
            for cls in self.classes:
                cls_dir = self.root_dir / cls
                if not cls_dir.exists():
                    raise ValueError(f"Missing class dir: {cls_dir}")
                label = self.class_to_idx[cls]
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in _exts:
                        self.image_paths.append(p)
                        self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

        def get_class_names(self):
            return self.classes

    def compute_metrics(y_true, y_pred, class_names=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if class_names is None:
            num_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
            class_names = [f"class_{i}" for i in range(num_classes)]
        accuracy = float((y_true == y_pred).mean())
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
        )
        per_class = {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(len(class_names))
        }
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        return {
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "qwk": qwk,
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_names": class_names,
        }


# Sibling import (hyphen folder, run by path).
from uni_regularized_model import UNIRegularizedModel


# ------------------------------------------------------------
# Transforms — NO ImageNet normalization in the dataloader.
# The model applies normalization internally.
# ------------------------------------------------------------
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=10,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),  # raw RGB [0, 1]
    ])


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # raw RGB [0, 1]
    ])


# ------------------------------------------------------------
# Split — LOAD ONLY, never regenerate.
# ------------------------------------------------------------
def load_split_indices(split_indices_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(split_indices_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found at '{path}'. This file must already exist; "
            "it is never regenerated by the UNI-Regularized."
        )
    data = np.load(path)
    for key in ("train_indices", "val_indices", "test_indices"):
        if key not in data:
            raise KeyError(f"Split file '{path}' is missing key '{key}'.")
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]
    val_fraction = float(data.get("val_fraction", -1.0))
    print(f"Loaded split from '{path}'")
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | "
          f"Test: {len(test_indices)} | val_fraction: {val_fraction}")
    if val_fraction != 0.10:
        raise ValueError(
            f"Expected val_fraction=0.10 but split file contains {val_fraction}. "
            "Refusing to train on a different split."
        )
    return train_indices, val_indices, test_indices


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("uni_regularized")
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
def split_decay_params(module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split params into decay (ndim > 1) and no_decay (ndim <= 1).

    Skips params with requires_grad=False.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:  # LayerNorm / bias / 1D
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def set_stage_requires_grad(model: nn.Module, stage: int) -> None:
    """Stage 1: freeze backbone, train head only. Stage 2: train all."""
    if stage == 1:
        model.freeze_backbone()
        for p in model.head.parameters():
            p.requires_grad = True
    elif stage == 2:
        model.unfreeze_backbone()
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_optimizer(model: nn.Module, cfg: dict, stage: int) -> torch.optim.Optimizer:
    """AdamW with decay/no-decay split (weight decay excluded from 1D params)."""
    cfg_training = cfg["training"]
    if stage == 1:
        wd_s1 = cfg_training.get("weight_decay_stage1", 0.0)
        head_decay, head_no_decay = split_decay_params(model.head)
        groups = []
        if head_decay:
            groups.append({"params": head_decay, "lr": cfg_training["stage1_lr"], "weight_decay": wd_s1})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": cfg_training["stage1_lr"], "weight_decay": 0.0})
        return torch.optim.AdamW(groups)

    if stage == 2:
        wd_s2 = cfg_training.get("weight_decay_stage2", 0.05)
        backbone_decay, backbone_no_decay = split_decay_params(model.backbone)
        head_decay, head_no_decay = split_decay_params(model.head)
        groups = []
        if backbone_decay:
            groups.append({"params": backbone_decay, "lr": cfg_training["stage2_backbone_lr"], "weight_decay": wd_s2})
        if backbone_no_decay:
            groups.append({"params": backbone_no_decay, "lr": cfg_training["stage2_backbone_lr"], "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "lr": cfg_training["stage2_head_lr"], "weight_decay": wd_s2})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": cfg_training["stage2_head_lr"], "weight_decay": 0.0})
        return torch.optim.AdamW(groups)

    raise ValueError(f"Unknown stage: {stage}")


# ------------------------------------------------------------
# Epoch loops
# ------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, device, logger, debug=False
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss = F.cross_entropy(pred["logits"], labels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        if debug and batch_idx >= 2:
            logger.info(f"[DEBUG] Train stopped after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return running_loss / n, 100.0 * correct / n


@torch.no_grad()
def validate_one_epoch(
    model, dataloader, device, logger, debug=False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images)
        loss = F.cross_entropy(pred["logits"], labels)

        bs = images.size(0)
        running_loss += loss.item() * bs
        preds = pred["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if debug and batch_idx >= 2:
            logger.warning(f"[DEBUG] Val stopped after {batch_idx + 1} batches.")
            break

    n = max(total, 1)
    return (
        running_loss / n,
        100.0 * correct / n,
        np.array(all_preds),
        np.array(all_labels),
    )


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------
def save_checkpoint(
    model, optimizer, scheduler, epoch, stage, metrics, config,
    split_path, save_path, rank,
) -> None:
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "model_state_dict": state_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "config": config,
        "split_indices_path": split_path,
    }
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to '{save_path}'")


def load_model_state(model: nn.Module, checkpoint_path: str, device) -> None:
    """Loads only the model state from a checkpoint (Stage-2 init)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_model = model.module if hasattr(model, "module") else model
    state_model.load_state_dict(ckpt["model_state_dict"])


# ------------------------------------------------------------
# Static forbidden-import scan (debug)
# ------------------------------------------------------------
def static_check() -> None:
    forbidden = [
        "hf_hub_download",
        "huggingface_hub.login",
        "HF_TOKEN",
        "from_pretrained",
    ]
    target_dir = Path(__file__).resolve().parent
    # Skip this file — it merely DECLARES the forbidden strings
    # for the scan and must not flag itself.
    self_name = Path(__file__).name
    for f in target_dir.glob("*.py"):
        if f.name == self_name:
            continue
        text = f.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise RuntimeError(
                    f"Forbidden token '{token}' found in {f.name}. "
                    "The UNI-Regularized must be fully offline."
                )
    print("[DEBUG] Static check passed: no forbidden network/download tokens.")


# ------------------------------------------------------------
# Sanity check (debug)
# ------------------------------------------------------------
def run_sanity_check(
    model, device, num_classes=4, bs=2, image_size=224, device_name="cpu",
) -> None:
    print("=" * 60)
    print("SANITY CHECK — UNI-Regularized")
    print("=" * 60)

    def _assert_no_naninf(t, name):
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise RuntimeError(f"NaN/Inf found in {name}")

    x = torch.rand(bs, 3, image_size, image_size, device=device)  # raw RGB [0,1]
    with torch.no_grad():
        pred = model(x)
        print(f"  Logits             : {tuple(pred['logits'].shape)}")
        print(f"  Probs              : {tuple(pred['probs'].shape)}")
        assert tuple(pred["logits"].shape) == (bs, num_classes), "logits shape!"
        assert tuple(pred["probs"].shape) == (bs, num_classes), "probs shape!"
        _assert_no_naninf(pred["logits"], "logits")
        _assert_no_naninf(pred["probs"], "probs")

    x2 = torch.rand(bs, 3, image_size, image_size, device=device)
    pred2 = model(x2)
    loss = F.cross_entropy(
        pred2["logits"],
        torch.zeros(bs, dtype=torch.long, device=device),
    )
    loss.backward()
    print(f"  loss.backward()    : OK (loss={loss.item():.4f})")

    # Assert freeze behavior
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters()), \
        "freeze_backbone() left some backbone params trainable!"
    assert all(p.requires_grad for p in model.head.parameters()), \
        "freeze_backbone() froze head params!"
    print("  freeze_backbone()  : OK (backbone frozen, head trainable)")

    # Assert unfreeze behavior
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters()), \
        "unfreeze_backbone() left some backbone params frozen!"
    print("  unfreeze_backbone(): OK (all backbone trainable)")

    counts = model.count_parameters()
    print("-" * 60)
    print("Parameter Summary")
    print("-" * 60)
    for name, count in counts.items():
        print(f"  {name:<12} : {count:>12,}")
    print("=" * 60)
    print("✅ Sanity check passed.")
    print(f"  Device: {device_name}")


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def run_training(config: dict, resume: bool, debug: bool, force_distributed: bool = False) -> None:
    rank, world_size, local_rank = setup_ddp(force_distributed=force_distributed)
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu"

    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_dir = config["paths"]["log_dir"]
    logger = setup_logging(log_dir, rank)
    logger.info(f"Rank {rank}/{world_size} | Device: {device}")

    train_dir = config["paths"]["train_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    split_path = config["paths"]["split_indices_path"]
    experiment_dir = config["paths"]["experiment_dir"]
    results_dir = config["paths"]["results_dir"]
    uni_checkpoint_path = config["paths"]["uni_checkpoint_path"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Static scan for forbidden network tokens (before model creation).
    if debug and rank == 0:
        static_check()

    # Split file — LOAD ONLY.
    if rank == 0:
        logger.info(f"Loading WSI split from '{split_path}'")
    train_indices, val_indices, test_indices = load_split_indices(split_path)

    # Model.
    model = UNIRegularizedModel(
        checkpoint_path=uni_checkpoint_path,
        num_classes=config["model"]["num_classes"],
        verbose=(rank == 0),
    ).to(device)

    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("UNI-Regularized Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<12} : {count:>12,}")
        logger.info("=" * 60)

    if debug and rank == 0:
        run_sanity_check(
            model, device,
            num_classes=config["model"]["num_classes"],
            image_size=config["model"]["image_size"],
            device_name=device_name,
        )
        _debug_stage2_init_check(model, device, checkpoint_dir, config)

    # Data.
    image_size = config["model"]["image_size"]
    train_transform = get_train_transform(image_size=image_size)
    test_transform = get_test_transform(image_size=image_size)

    train_set = HER2BaselineDataset(root_dir=train_dir, transform=train_transform)
    val_set = HER2BaselineDataset(root_dir=train_dir, transform=test_transform)
    train_dataset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)

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
        f"Test (held out): {len(test_indices)} | Batch (per GPU): {batch_size}"
    )

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    # AMP disabled.
    use_amp = False
    scaler = None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    stage1_epochs = config["training"]["stage1_epochs"]
    stage2_epochs = config["training"]["stage2_epochs"]

    # Early stopping config
    patience = config["training"].get("early_stopping_patience", 7)
    min_delta = config["training"].get("early_stopping_min_delta", 1e-4)

    logger.info("=" * 60)
    logger.info(f"Early stopping: patience={patience}, min_delta={min_delta} (on val loss)")
    logger.info("=" * 60)

    # Stage 1
    logger.info("=" * 60)
    logger.info(f"Stage 1 — up to {stage1_epochs} epochs (UNI frozen, head only)")
    logger.info("=" * 60)

    set_stage_requires_grad(model, 1)
    optimizer_s1 = build_optimizer(model, config, 1)
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=stage1_epochs)

    start_epoch_s1 = 0
    best_val_acc_s1 = 0.0
    best_metrics_s1 = {}
    best_val_loss_s1 = float("inf")
    epochs_no_improve_s1 = 0
    stage1_stopped_reason = f"reached max_epochs cap"
    stage1_stopped_epoch = stage1_epochs

    if resume:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from '{resume_path}' (Stage 1)")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            ckpt_stage = ckpt.get("stage", 1)
            ckpt_epoch = ckpt.get("epoch", 0)
            if ckpt_stage != 1:
                logger.warning(
                    f"Checkpoint stage {ckpt_stage} != 1; starting Stage 1 from scratch."
                )
            else:
                start_epoch_s1 = ckpt_epoch
                if "optimizer_state_dict" in ckpt:
                    optimizer_s1.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                    scheduler_s1.load_state_dict(ckpt["scheduler_state_dict"])
                state_model = model.module if hasattr(model, "module") else model
                state_model.load_state_dict(ckpt["model_state_dict"])
                best_val_acc_s1 = ckpt.get("metrics", {}).get("accuracy", 0.0)
                best_metrics_s1 = ckpt.get("metrics", {})
                best_val_loss_s1 = ckpt.get("metrics", {}).get("val_loss", float("inf"))
                logger.info(f"Resumed Stage 1 at epoch {ckpt_epoch}")

    for epoch in range(start_epoch_s1, stage1_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer_s1,
            device=device, logger=logger, debug=debug,
        )

        scheduler_s1.step()

        if rank == 0:
            val_loss, val_acc, preds, labels = validate_one_epoch(
                model=model, dataloader=val_loader, device=device,
                logger=logger, debug=debug,
            )

            class_names = val_set.get_class_names()
            metrics = compute_metrics(labels, preds, class_names)
            metrics["val_loss"] = val_loss
            metrics["train_loss"] = train_loss
            metrics["train_acc"] = train_acc
            metrics["epoch"] = epoch + 1
            metrics["stage"] = 1

            elapsed = time.time() - epoch_start
            logger.info(
                f"Stage 1 | Epoch [{epoch+1:02d}/{stage1_epochs}] | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.2f}% | QWK {metrics['qwk']:.4f} | {elapsed:.1f}s"
            )

            json_metrics = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }
            metrics_path = os.path.join(log_dir, "metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(json_metrics) + "\n")

            # Checkpoint selection by VAL ACCURACY (unchanged).
            if val_acc > best_val_acc_s1:
                best_val_acc_s1 = val_acc
                best_metrics_s1 = metrics
                save_checkpoint(
                    model=model, optimizer=optimizer_s1, scheduler=scheduler_s1,
                    epoch=epoch + 1, stage=1, metrics=metrics,
                    config=config, split_path=split_path,
                    save_path=os.path.join(checkpoint_dir, "best_stage1.pt"),
                    rank=rank,
                )
                logger.info(f"  New best stage 1 (val acc {val_acc:.2f}%)")

            save_checkpoint(
                model=model, optimizer=optimizer_s1, scheduler=scheduler_s1,
                epoch=epoch + 1, stage=1, metrics=metrics,
                config=config, split_path=split_path,
                save_path=os.path.join(checkpoint_dir, "last.pt"),
                rank=rank,
            )

            # Early stopping on VAL LOSS only.
            if val_loss < best_val_loss_s1 - min_delta:
                best_val_loss_s1 = val_loss
                epochs_no_improve_s1 = 0
            else:
                epochs_no_improve_s1 += 1

            if epochs_no_improve_s1 >= patience:
                stage1_stopped_reason = f"early stopping triggered at epoch {epoch + 1}"
                stage1_stopped_epoch = epoch + 1
                logger.info(
                    f"Early stopping: no val_loss improvement for {patience} epochs "
                    f"-> {stage1_stopped_reason}"
                )
                break
        else:
            logger.info(
                f"Stage 1 | Epoch [{epoch+1:02d}/{stage1_epochs}] | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}%"
            )

    else:
        # Loop completed without break -> reached max epochs
        stage1_stopped_reason = f"reached max_epochs cap"
        stage1_stopped_epoch = stage1_epochs

    if rank == 0:
        logger.info(f"Stage 1 stopped: {stage1_stopped_reason}")

    if rank == 0 and os.path.exists(os.path.join(checkpoint_dir, "best_stage1.pt")):
        save_checkpoint(
            model=model, optimizer=optimizer_s1, scheduler=scheduler_s1,
            epoch=stage1_stopped_epoch, stage=1, metrics=best_metrics_s1,
            config=config, split_path=split_path,
            save_path=os.path.join(checkpoint_dir, "stage1_end.pt"),
            rank=rank,
        )

    # ------------------------------------------------------------
    # Stage 2 — initialized from best_stage1.pt (NOT last.pt)
    # ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Stage 2 — up to {stage2_epochs} epochs (full fine-tuning)")
    logger.info("=" * 60)

    set_stage_requires_grad(model, 2)
    optimizer_s2 = build_optimizer(model, config, 2)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs)

    if rank == 0:
        best_stage1_path = os.path.join(checkpoint_dir, "best_stage1.pt")
        if not os.path.exists(best_stage1_path):
            raise FileNotFoundError(
                f"best_stage1.pt not found at '{best_stage1_path}'. "
                "Stage 2 requires the Stage 1 best-accuracy checkpoint."
            )
        logger.info("Stage 2 initialized from best_stage1.pt")
        load_model_state(model, best_stage1_path, device)

    start_epoch_s2 = 0
    best_val_acc_s2 = 0.0
    best_metrics_s2 = {}
    best_val_loss_s2 = float("inf")
    epochs_no_improve_s2 = 0
    stage2_stopped_reason = f"reached max_epochs cap"
    stage2_stopped_epoch = stage2_epochs

    if resume:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from '{resume_path}' (Stage 2)")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            ckpt_stage = ckpt.get("stage", 2)
            if ckpt_stage == 2:
                start_epoch_s2 = ckpt.get("epoch", 0)
                if "optimizer_state_dict" in ckpt:
                    optimizer_s2.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                    scheduler_s2.load_state_dict(ckpt["scheduler_state_dict"])
                state_model = model.module if hasattr(model, "module") else model
                state_model.load_state_dict(ckpt["model_state_dict"])
                best_val_acc_s2 = ckpt.get("metrics", {}).get("accuracy", 0.0)
                best_metrics_s2 = ckpt.get("metrics", {})
                best_val_loss_s2 = ckpt.get("metrics", {}).get("val_loss", float("inf"))
                logger.info(f"Resumed Stage 2 at epoch {start_epoch_s2}")

    for epoch in range(start_epoch_s2, stage2_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer_s2,
            device=device, logger=logger, debug=debug,
        )

        scheduler_s2.step()

        if rank == 0:
            val_loss, val_acc, preds, labels = validate_one_epoch(
                model=model, dataloader=val_loader, device=device,
                logger=logger, debug=debug,
            )

            class_names = val_set.get_class_names()
            metrics = compute_metrics(labels, preds, class_names)
            metrics["val_loss"] = val_loss
            metrics["train_loss"] = train_loss
            metrics["train_acc"] = train_acc
            metrics["epoch"] = epoch + 1
            metrics["stage"] = 2

            elapsed = time.time() - epoch_start
            logger.info(
                f"Stage 2 | Epoch [{epoch+1:02d}/{stage2_epochs}] | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.2f}% | QWK {metrics['qwk']:.4f} | {elapsed:.1f}s"
            )

            json_metrics = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }
            metrics_path = os.path.join(log_dir, "metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(json_metrics) + "\n")

            if val_acc > best_val_acc_s2:
                best_val_acc_s2 = val_acc
                best_metrics_s2 = metrics
                save_checkpoint(
                    model=model, optimizer=optimizer_s2, scheduler=scheduler_s2,
                    epoch=epoch + 1, stage=2, metrics=metrics,
                    config=config, split_path=split_path,
                    save_path=os.path.join(checkpoint_dir, "best_stage2.pt"),
                    rank=rank,
                )
                logger.info(f"  New best stage 2 (val acc {val_acc:.2f}%)")

            save_checkpoint(
                model=model, optimizer=optimizer_s2, scheduler=scheduler_s2,
                epoch=epoch + 1, stage=2, metrics=metrics,
                config=config, split_path=split_path,
                save_path=os.path.join(checkpoint_dir, "last.pt"),
                rank=rank,
            )

            if val_loss < best_val_loss_s2 - min_delta:
                best_val_loss_s2 = val_loss
                epochs_no_improve_s2 = 0
            else:
                epochs_no_improve_s2 += 1

            if epochs_no_improve_s2 >= patience:
                stage2_stopped_reason = f"early stopping triggered at epoch {epoch + 1}"
                stage2_stopped_epoch = epoch + 1
                logger.info(
                    f"Early stopping: no val_loss improvement for {patience} epochs "
                    f"-> {stage2_stopped_reason}"
                )
                break
        else:
            logger.info(
                f"Stage 2 | Epoch [{epoch+1:02d}/{stage2_epochs}] | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}%"
            )

    else:
        stage2_stopped_reason = f"reached max_epochs cap"
        stage2_stopped_epoch = stage2_epochs

    if rank == 0:
        logger.info(f"Stage 2 stopped: {stage2_stopped_reason}")
        save_checkpoint(
            model=model, optimizer=optimizer_s2, scheduler=scheduler_s2,
            epoch=stage2_stopped_epoch, stage=2, metrics=best_metrics_s2,
            config=config, split_path=split_path,
            save_path=os.path.join(checkpoint_dir, "stage2_end.pt"),
            rank=rank,
        )

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"Stage 1 stopped: {stage1_stopped_reason}")
        logger.info(f"Stage 2 stopped: {stage2_stopped_reason}")
        logger.info(f"Best Stage 1 val accuracy: {best_val_acc_s1:.4f}%")
        logger.info(f"Best Stage 2 val accuracy: {best_val_acc_s2:.4f}%")
        logger.info("=" * 60)

    cleanup_ddp()


# ------------------------------------------------------------
# Debug: mock Stage-2 initialization check (fast, no epochs)
# ------------------------------------------------------------
def _debug_stage2_init_check(model, device, checkpoint_dir, config) -> None:
    print("=" * 60)
    print("DEBUG: Stage 2 initialization check (mocked)")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_stage1_path = os.path.join(checkpoint_dir, "best_stage1.pt")
    save_checkpoint(
        model=model, optimizer=None, scheduler=None,
        epoch=1, stage=1, metrics={}, config=config,
        split_path="mock", save_path=best_stage1_path, rank=0,
    )

    # Save current head weights
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}

    # Zero out head weights to prove the load actually restores them.
    for p in model.head.parameters():
        p.data.zero_()

    ckpt = torch.load(best_stage1_path, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt, "best_stage1.pt missing 'model_state_dict'!"
    state_model = model.module if hasattr(model, "module") else model
    state_model.load_state_dict(ckpt["model_state_dict"])
    print("  ... zeroed head weights, then loaded best_stage1.pt ...")

    # Verify restoration by comparing a few head tensors.
    for name in head_before:
        assert torch.allclose(
            getattr(model.head, name), head_before[name]
        ), f"Head tensor '{name}' was not restored exactly from best_stage1.pt!"
    print("  Head weight restoration (exact tensor comparison) : OK")
    print("=" * 60)
    print("✅ Stage 2 init check passed (uses best_stage1.pt, key 'model_state_dict').")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNI-Regularized (HER2-IHC-40x)")
    parser.add_argument("--config", type=str, default="configs/uni_regularized_config.yaml")
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
    debug = config["training"].get("debug", False)
    run_training(config, resume=args.resume, debug=debug, force_distributed=args.distributed)


if __name__ == "__main__":
    main()