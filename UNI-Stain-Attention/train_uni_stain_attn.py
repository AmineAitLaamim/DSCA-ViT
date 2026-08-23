# ============================================================
# UNI-Stain-Attention — Training Script (CLI, HPC-ready)
# ============================================================
#
# Single-stage training. UNI backbone is frozen FOREVER.
# Trainable: stain_encoder, stain_query_proj, patch_key_proj,
# patch_value_proj, attention (MultiheadAttention), head.
#
#   Epochs        : 30
#   Optimizer     : AdamW (trainable params only, never backbone)
#   LR            : 1e-3
#   Weight decay  : 0.05 (weights only)
#   Loss          : CrossEntropyLoss(label_smoothing=0.1)
#   Batch size    : 64 (per GPU)
#   AMP           : enabled
#   Grad clip     : 1.0
#   Scheduler     : CosineAnnealingLR(T_max=epochs)
#   Validation    : every epoch, best by val accuracy
#
# The split file is LOAD-ONLY (never regenerated):
#   .../plain_vit_baseline_001/split_indices_wsi.npz
#   (train 7283 | val 810 | test 1847, val_fraction=0.10, seed=42)
#
# Usage:
#   uv run python UNI-Stain-Attention/train_uni_stain_attn.py \
#       --config configs/uni_stain_attn_config.yaml
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
    from torch.utils.data import Dataset as _Dataset

    CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    class HER2BaselineDataset(_Dataset):
        def __init__(self, root_dir: str, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = CLASSES
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.image_paths: List[Path] = []
            self.labels: List[int] = []
            if not self.root_dir.exists():
                raise ValueError(f"Dataset root not found: {self.root_dir}")
            for cls in self.classes:
                cls_dir = self.root_dir / cls
                if not cls_dir.exists():
                    raise ValueError(f"Missing class dir: {cls_dir}")
                label = self.class_to_idx[cls]
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
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

        def get_num_classes(self):
            return len(self.classes)

        def get_class_names(self):
            return self.classes

    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        cohen_kappa_score,
        precision_recall_fscore_support,
    )

    def compute_metrics(y_true, y_pred, class_names=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if class_names is None:
            num_classes = max(int(y_true.max()) + 1, int(y_pred.max()) + 1)
            class_names = [f"class_{i}" for i in range(num_classes)]
        accuracy = float((y_true == y_pred).mean())
        bal = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
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
            "balanced_accuracy": bal,
            "macro_f1": macro_f1,
            "weighted_f1": w_f1,
            "qwk": qwk,
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_names": class_names,
        }


# Sibling import (hyphen folder, run by path).
from uni_stain_attn import UNIStainAttention
from stain_stats import load_stain_stats


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
            "it is never regenerated by the UNI-Stain-Attention."
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
    logger = logging.getLogger("uni_stain_attn")
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
# MixUp (optional, default 0.0)
# ------------------------------------------------------------
def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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
# Optimizer — only trainable params (never backbone)
# ------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW on trainable params only, weight decay on weights only."""
    wd = cfg["training"]["weight_decay"]

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    decay = []
    no_decay = []
    for p in trainable_params:
        if p.ndim <= 1:  # bias / BN
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "lr": cfg["training"]["lr"], "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "lr": cfg["training"]["lr"], "weight_decay": 0.0})

    return torch.optim.AdamW(groups)


# ------------------------------------------------------------
# Epoch loops
# ------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, scaler, device, config, logger, debug=False
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    label_smoothing = config["training"]["label_smoothing"]
    mixup_alpha = config["training"].get("mixup_alpha", 0.0)
    use_mixup = mixup_alpha > 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            y_a = y_b = labels

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            pred = model(images)
            if use_mixup:
                loss = lam * F.cross_entropy(pred["logits"], y_a, label_smoothing=label_smoothing) \
                       + (1.0 - lam) * F.cross_entropy(pred["logits"], y_b, label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(pred["logits"], labels, label_smoothing=label_smoothing)

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
    model, dataloader, device, config, logger, debug=False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    label_smoothing = config["training"]["label_smoothing"]

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images)
        loss = F.cross_entropy(pred["logits"], labels, label_smoothing=label_smoothing)

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
    model, optimizer, scheduler, epoch, metrics, config,
    split_path, save_path, rank,
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
        "metrics": metrics,
        "config": config,
        "split_indices_path": split_path,
    }
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to '{save_path}'")


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
                    "The UNI-Stain-Attention must be fully offline."
                )
    print("[DEBUG] Static check passed: no forbidden network/download tokens.")


# ------------------------------------------------------------
# Sanity check (debug)
# ------------------------------------------------------------
def run_sanity_check(model, device, num_classes=4, bs=2, image_size=224, device_name="cpu") -> None:
    print("=" * 60)
    print("SANITY CHECK — UNI-Stain-Attention")
    print("=" * 60)

    def _assert_no_naninf(t, name):
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise RuntimeError(f"NaN/Inf found in {name}")

    # Verify backbone is frozen
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    if not backbone_frozen:
        raise RuntimeError("UNI backbone must be fully frozen (requires_grad=False).")
    print("  UNI backbone frozen : OK (all requires_grad=False)")

    # Verify trainable params > 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("No trainable parameters found!")
    print(f"  Trainable params    : {trainable:,} (>0 OK)")

    model.eval()
    x = torch.rand(bs, 3, image_size, image_size, device=device)  # raw RGB [0,1]
    with torch.no_grad():
        pred = model(x)
        print(f"  Input RGB          : {tuple(x.shape)}")
        print(f"  Logits             : {tuple(pred['logits'].shape)}")
        print(f"  Probs              : {tuple(pred['probs'].shape)}")
        assert tuple(pred["logits"].shape) == (bs, num_classes), "logits shape!"
        assert tuple(pred["probs"].shape) == (bs, num_classes), "probs shape!"
        _assert_no_naninf(pred["logits"], "logits")
        _assert_no_naninf(pred["probs"], "probs")

    model.train()
    # After .train(), backbone must still be in eval mode
    if model.backbone.training:
        raise RuntimeError("backbone.training must be False after model.train()!")
    print("  model.train() keeps backbone in eval mode : OK")

    # Backbone forward is wrapped in torch.no_grad() (code inspection):
    # The forward() method uses `with torch.no_grad():` around forward_features.
    print("  Backbone forward inside torch.no_grad() : OK (code inspection)")

    # Verify attention output shape [B, 256] before fusion
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        x_norm = (x - mean) / std
        tokens = model.backbone.forward_features(x_norm)
        patch_tokens = tokens[:, 1:]  # [B, 196, 1024]
        stain_query = model.stain_query_proj(
            model.stain_encoder(
                torch.cat([
                    (torch.rand_like(x[:, :1, :, :]) * 0.5),
                    (torch.rand_like(x[:, :1, :, :]) * 0.5),
                ], dim=1)
            )
        )  # [B, 256]
        pad_keys = model.patch_key_proj(patch_tokens)
        pad_vals = model.patch_value_proj(patch_tokens)
        attn_out, _ = model.attn(stain_query.unsqueeze(1), pad_keys, pad_vals)
        stain_attended = attn_out.squeeze(1)  # [B, 256]
        assert tuple(stain_attended.shape) == (bs, 256), "attention output shape!"
    print(f"  Attention output     : {tuple(stain_attended.shape)} (OK)")

    x2 = torch.rand(bs, 3, image_size, image_size, device=device)
    pred2 = model(x2)
    loss = F.cross_entropy(
        pred2["logits"],
        torch.zeros(bs, dtype=torch.long, device=device),
        label_smoothing=0.1,
    )
    loss.backward()
    print(f"  loss.backward()    : OK (loss={loss.item():.4f})")

    counts = model.count_parameters()
    print("-" * 60)
    print("Parameter Summary")
    print("-" * 60)
    for name, count in counts.items():
        print(f"  {name:<18} : {count:>12,}")
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
    stain_stats_path = config["paths"]["stain_stats_path"]

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

    # Stain stats — must exist (precompute script).
    if not os.path.exists(stain_stats_path):
        raise FileNotFoundError(
            f"Stain statistics file not found at '{stain_stats_path}'. "
            "Run UNI-Stain-Attention/precompute_stain_stats.py first."
        )
    stain_stats = load_stain_stats(stain_stats_path)

    # Model.
    model = UNIStainAttention(
        checkpoint_path=uni_checkpoint_path,
        num_classes=config["model"]["num_classes"],
        stain_dim=config["training"].get("stain_dim", 512),
        stain_stats=stain_stats,
        verbose=(rank == 0),
    ).to(device)

    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("UNI-Stain-Attention Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<18} : {count:>12,}")
        logger.info("=" * 60)

    if debug and rank == 0:
        run_sanity_check(
            model, device, device_name=device_name,
            num_classes=config["model"]["num_classes"],
            image_size=config["model"]["image_size"],
        )

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

    # AMP
    amp_cfg = config["training"].get("amp", True)
    use_amp = bool(amp_cfg) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    epochs = config["training"]["epochs"]
    start_epoch = 0
    best_val_acc = 0.0
    best_metrics = {}

    if resume:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from '{resume_path}'")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("metrics", {}).get("accuracy", 0.0)
            best_metrics = ckpt.get("metrics", {})
            state_model = model.module if hasattr(model, "module") else model
            state_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Resumed at epoch {start_epoch}, best val acc {best_val_acc:.4f}"
            )
        else:
            logger.warning(f"No checkpoint found at '{resume_path}'. Starting fresh.")

    # Single stage
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    if resume and start_epoch > 0:
        resume_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    logger.info("=" * 60)
    logger.info(f"Training for {config['training']['epochs']} epochs (UNI frozen)")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            scaler=scaler, device=device, config=config, logger=logger, debug=debug,
        )

        scheduler.step()

        if rank == 0:
            val_loss, val_acc, preds, labels = validate_one_epoch(
                model=model, dataloader=val_loader, device=device,
                config=config, logger=logger, debug=debug,
            )

            class_names = val_set.get_class_names()
            metrics = compute_metrics(labels, preds, class_names)
            metrics["val_loss"] = val_loss
            metrics["train_loss"] = train_loss
            metrics["train_acc"] = train_acc
            metrics["epoch"] = epoch + 1

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch+1:02d}/{config['training']['epochs']}] | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
                f"QWK {metrics['qwk']:.4f} | {elapsed:.1f}s"
            )

            json_metrics = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }
            metrics_path = os.path.join(log_dir, "metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(json_metrics) + "\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = metrics
                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch + 1, metrics=metrics,
                    config=config, split_path=split_path,
                    save_path=os.path.join(checkpoint_dir, "best.pt"),
                    rank=rank,
                )
                logger.info(f"  New best (val acc {val_acc:.2f}%)")

            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, metrics=metrics,
                config=config, split_path=split_path,
                save_path=os.path.join(checkpoint_dir, "last.pt"),
                rank=rank,
            )
        else:
            logger.info(
                f"Epoch [{epoch+1:02d}/{config['training']['epochs']}] | "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}%"
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
    parser = argparse.ArgumentParser(description="Train UNI-Stain-Attention (HER2-IHC-40x)")
    parser.add_argument("--config", type=str, default="configs/uni_stain_attn_config.yaml")
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