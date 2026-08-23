# ============================================================
# UNI-CORN — Training Script (CLI, HPC/DDP-ready)
# ============================================================
#
# Combines the UNI backbone with CORN ordinal regression loss.
# Uses the ORIGINAL UNI-baseline recipe exactly (Adam, no weight
# decay, no early stopping), except loss and label conversion.
#
#   Stage 1 (30 ep): frozen UNI backbone, train CORN head only,
#       Adam lr=1e-4.
#   Stage 2 (30 ep): full fine-tuning, Adam backbone 1e-5,
#       head 1e-4.
#
#   - Optimizer : Adam (NOT AdamW)
#   - Weight decay: 0.0
#   - Loss      : corn_loss(logits, labels, num_classes=4)
#   - Batch size: 32 (per GPU)
#   - AMP: disabled
#   - Gradient clipping: none
#   - Scheduler : CosineAnnealingLR(T_max=stage_epochs), recreated per stage
#   - Validation: every epoch, best checkpoint by val accuracy
#   - No early stopping (fixed 30+30 epoch budget)
#
# Predictions are ALWAYS corn_label_from_logits(logits).
# torch.argmax must NOT appear anywhere in this package.
#
# Stage 2 is initialized from best_stage1.pt (NOT last.pt).
#
# The split file is LOAD-ONLY (never regenerated):
#   .../plain_vit_baseline_001/split_indices_wsi.npz
#   (train 7283 | val 810 | test 1847, val_fraction=0.10, seed=42)
#
# Usage:
#   uv run python UNI-CORN/train_uni_corn.py \
#       --config configs/uni_corn_config.yaml
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
import yaml
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss as corn_loss_fn
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
from uni_corn_model import UNICORN


# ------------------------------------------------------------
# Transforms — NO ImageNet normalization in the dataloader.
# The model applies normalization internally (UNI-baseline
# convention).
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
            "it is never regenerated by the UNI-CORN."
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
    logger = logging.getLogger("uni_corn")
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
        for p in model.head.parameters():
            p.requires_grad = True
    elif stage == 2:
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def build_optimizer(model: nn.Module, cfg: dict, stage: int) -> torch.optim.Optimizer:
    """Adam (weight decay 0.0) matching the original UNI-baseline."""
    wd = cfg["training"]["weight_decay"]
    if stage == 1:
        head_params = [p for p in model.head.parameters()]
        groups = [{"params": head_params, "lr": cfg["training"]["stage1_lr"], "weight_decay": wd}]
    elif stage == 2:
        groups = [
            {"params": [p for p in model.backbone.parameters()],
             "lr": cfg["training"]["stage2_backbone_lr"], "weight_decay": wd},
            {"params": [p for p in model.head.parameters()],
             "lr": cfg["training"]["stage2_head_lr"], "weight_decay": wd},
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return torch.optim.Adam(groups)


# ------------------------------------------------------------
# Epoch loops
# ------------------------------------------------------------
def train_one_epoch(
    model, dataloader, optimizer, device, config, logger, debug=False
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
        loss = corn_loss_fn(pred["logits"], labels, num_classes=config["model"]["num_classes"])
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        preds = corn_label_from_logits(pred["logits"])
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

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images)
        loss = corn_loss_fn(pred["logits"], labels, num_classes=config["model"]["num_classes"])

        bs = images.size(0)
        running_loss += loss.item() * bs
        preds = corn_label_from_logits(pred["logits"])
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
        "optimizer_state_dict": optimizer.state_dict(),
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
        "argmax",  # must not appear anywhere in the package
    ]
    target_dir = Path(__file__).resolve().parent
    self_name = Path(__file__).name
    for f in target_dir.glob("*.py"):
        if f.name == self_name:
            continue
        text = f.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise RuntimeError(
                    f"Forbidden token '{token}' found in {f.name}. "
                    "The UNI-CORN must be fully offline and must not "
                    "use argmax for predictions."
                )
    print("[DEBUG] Static check passed: no forbidden tokens (includes argmax).")


# ------------------------------------------------------------
# Sanity check (debug)
# ------------------------------------------------------------
def run_sanity_check(
    model, device, config, bs=2, image_size=224, device_name="cpu", stage=1,
) -> None:
    print("=" * 60)
    print("SANITY CHECK — UNI-CORN")
    print("=" * 60)

    def _assert_no_naninf(t, name):
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise RuntimeError(f"NaN/Inf found in {name}")

    # Assert class-to-index mapping.
    cls_to_idx = {"class_0": 0, "class_1+": 1, "class_2+": 2, "class_3+": 3}
    print(f"  Class mapping : {cls_to_idx}")
    assert cls_to_idx == {"class_0": 0, "class_1+": 1, "class_2+": 2, "class_3+": 3}, \
        "class mapping mismatch!"

    x = torch.rand(bs, 3, image_size, image_size, device=device)  # raw RGB [0,1]
    with torch.no_grad():
        pred = model(x)
        print(f"  Logits             : {tuple(pred['logits'].shape)}")
        print(f"  Probs              : {tuple(pred['probs'].shape)}")
        assert tuple(pred["logits"].shape) == (bs, config["model"]["num_classes"] - 1), \
            "logits shape (must be [B, 3])!"
        assert tuple(pred["probs"].shape) == (bs, config["model"]["num_classes"]), \
            "probs shape (must be [B, 4])!"
        probs_sum = pred["probs"].sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4), \
            "probs do not sum to 1 per row!"
        labels_pred = corn_label_from_logits(pred["logits"])
        assert set(labels_pred.cpu().tolist()).issubset({0, 1, 2, 3}), \
            "corn_label_from_logits returned labels outside {0,1,2,3}!"
        _assert_no_naninf(pred["logits"], "logits")
        _assert_no_naninf(pred["probs"], "probs")

    x2 = torch.rand(bs, 3, image_size, image_size, device=device)
    pred2 = model(x2)
    loss = corn_loss_fn(
        pred2["logits"],
        torch.zeros(bs, dtype=torch.long, device=device),
        num_classes=config["model"]["num_classes"],
    )
    loss.backward()
    print(f"  loss.backward()    : OK (loss={loss.item():.4f})")

    if stage == 1:
        backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
        if not backbone_frozen:
            raise RuntimeError("Stage 1: backbone must be frozen (requires_grad=False).")
        print("  Stage 1 backbone frozen : OK")

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

    # Static scan for forbidden tokens (debug).
    if debug and rank == 0:
        static_check()

    # Split file — LOAD ONLY.
    if rank == 0:
        logger.info(f"Loading WSI split from '{split_path}'")
    train_indices, val_indices, test_indices = load_split_indices(split_path)

    # Model.
    model = UNICORN(
        checkpoint_path=uni_checkpoint_path,
        num_classes=config["model"]["num_classes"],
        verbose=(rank == 0),
    ).to(device)

    if rank == 0:
        counts = model.count_parameters()
        logger.info("=" * 60)
        logger.info("UNI-CORN Parameter Summary")
        logger.info("=" * 60)
        for name, count in counts.items():
            logger.info(f"  {name:<12} : {count:>12,}")
        logger.info("=" * 60)

    if debug and rank == 0:
        run_sanity_check(
            model, device, config,
            image_size=config["model"]["image_size"], device_name=device_name,
            stage=1,
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

    # AMP disabled (matches original UNI-baseline).
    use_amp = False
    scaler = None
    logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

    stages = [
        (1, config["training"]["stage1_epochs"]),
        (2, config["training"]["stage2_epochs"]),
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

        # Stage 2 initialization from best_stage1.pt (NOT last.pt).
        if stage == 2 and rank == 0:
            best_stage1_path = os.path.join(checkpoint_dir, "best_stage1.pt")
            if not os.path.exists(best_stage1_path):
                raise FileNotFoundError(
                    f"best_stage1.pt not found at '{best_stage1_path}'. "
                    "Stage 2 requires the Stage 1 best-accuracy checkpoint."
                )
            logger.info("Stage 2 initialized from best_stage1.pt")
            load_model_state(model, best_stage1_path, device)

        logger.info("=" * 60)
        logger.info(f"Stage {stage} — {stage_epochs} epochs")
        logger.info("=" * 60)

        for epoch in range(start_epoch if stage == start_stage else 0, stage_epochs):
            if is_distributed:
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model=model, dataloader=train_loader, optimizer=optimizer,
                device=device, config=config, logger=logger, debug=debug,
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
                metrics["stage"] = stage

                elapsed = time.time() - epoch_start
                logger.info(
                    f"Stage {stage} | Epoch [{epoch+1:02d}/{stage_epochs}] | "
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

                checkpoint_file = "best_stage1.pt" if stage == 1 else "best_stage2.pt"
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = metrics
                    save_checkpoint(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        stage=stage, epoch=epoch + 1, metrics=metrics,
                        config=config, split_path=split_path,
                        save_path=os.path.join(checkpoint_dir, checkpoint_file),
                        rank=rank,
                    )
                    logger.info(f"  New best stage {stage} (val acc {val_acc:.2f}%)")

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    stage=stage, epoch=epoch + 1, metrics=metrics,
                    config=config, split_path=split_path,
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
                stage=stage, epoch=stage_epochs, metrics=best_metrics,
                config=config, split_path=split_path,
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
    parser = argparse.ArgumentParser(description="Train UNI-CORN (HER2-IHC-40x)")
    parser.add_argument("--config", type=str, default="configs/uni_corn_config.yaml")
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