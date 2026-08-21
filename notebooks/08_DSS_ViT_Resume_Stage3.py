# DSS-ViT Resume Stage 3 Notebook
# ============================================================
# This file is written as a Python script with cell markers.
# Paste each section into a Google Colab cell.
#
# Loads the BEST Stage 3 checkpoint (best_stage3.pt) and
# CONTINUES training Stage 3 from the recorded epoch until the
# full 40 epochs are complete. Reuses the same training logic
# as 07_DSS_ViT_Training.py (utils/train_dss_vit.py helpers).
#
#   - Loads best_stage3.pt (model + optimizer + scheduler)
#   - Resumes the cosine LR schedule from the checkpoint epoch
#   - Trains until STAGE3_EPOCHS (40) total epochs
#   - Saves new best_stage3.pt / last.pt on improvement
#   - One final official-test evaluation
# ============================================================


# ============================================================
# Cell 1 — Environment / Drive / Repository
# ============================================================

from google.colab import drive
import os

drive.mount("/content/drive")

if os.path.exists("/content/drive/MyDrive"):
    print("✅ Google Drive mounted successfully.")
else:
    raise RuntimeError("❌ Google Drive was not mounted correctly.")

import subprocess

REPO_URL = "https://github.com/AmineAitLaamim/DSCA-ViT.git"
REPO_DIR = "/content/DSCA-ViT"

if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("✅ Repository cloned.")
else:
    print("Pulling latest changes...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print("✅ Repository updated.")

import sys
sys.path.insert(0, REPO_DIR)

print(f"REPO_DIR: {REPO_DIR}")


# ============================================================
# Cell 2 — Dependencies
# ============================================================
# scikit-learn is required by utils/split_utils.py and
# utils/metrics_dss_vit.py.

subprocess.run(
    [
        "pip", "install",
        "timm", "pyyaml", "scipy", "scikit-learn", "seaborn",
        "--quiet",
    ],
    check=True
)
print("✅ Dependencies installed.")


# ============================================================
# Cell 3 — Imports + Seed
# ============================================================

import random
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use "cuda:0" (not bare "cuda") so str(device) == 'cuda:0' matches
# str(param.device) for GPU tensors in the device check.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Environment Information")
print("=" * 60)
print(f"Python version  : {platform.python_version()}")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Device          : {device}")
print(f"Random seed     : {SEED}")
print("=" * 60)


# ============================================================
# Cell 4 — Dataset Preparation
# ============================================================
# Same proven download+extract logic as the v3 training notebook.

import zipfile
from pathlib import Path

DATA_ROOT = Path("/content/HER2_Dataset")
DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), URL], check=True)
else:
    print("Dataset archive already exists.")

WSI_DIR = DATA_ROOT / "WSI-based-dataset"
if not WSI_DIR.exists():
    print("Extracting main archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Main archive extracted.")
else:
    print("Main archive already extracted.")

nested_archives = [
    WSI_DIR / "train_data_wsi.zip",
    WSI_DIR / "test_data_wsi.zip",
]
for archive in nested_archives:
    extract_folder = archive.parent / archive.stem.replace("_data_wsi", "")
    if extract_folder.exists():
        print(f"{extract_folder.name} already extracted.")
        continue
    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(extract_folder)

for archive in [ZIP_PATH] + nested_archives:
    if archive.exists():
        archive.unlink()
print("ZIP files removed.")

TRAIN_DIR = WSI_DIR / "train"
TEST_DIR = WSI_DIR / "test"

assert TRAIN_DIR.exists(), "Train directory not found."
assert TEST_DIR.exists(), "Test directory not found."

print("\nDataset location:")
print(TRAIN_DIR)
print(TEST_DIR)


# ============================================================
# Cell 5 — Config, Colab Paths, Split Loaders, Stain Stats
# ============================================================
# Loads the DSS-ViT config, overrides the Toubkal-absolute paths
# with Colab-local ones, creates the deterministic stratified 10%
# validation holdout, and loads (or recomputes) the global H/DAB
# stain statistics.
#
# IMPORTANT: /content/ is wiped on every fresh Colab session —
# only Drive persists. The stain stats file created by the training
# notebook lives in /content/DSS_ViT_experiment/, so if it is gone
# (fresh session), we recompute it inline on the 90% train subset
# (deterministic, same logic as the training notebook).
#
# CRITICAL: the split and stain stats must match the original
# training run exactly (same seed 42, same 90% train subset).

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from datasets import HER2Dataset, get_train_transform, get_test_transform
from models_v2_1 import load_stain_stats
from utils.split_utils import get_or_create_split_indices

# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
with open(os.path.join(REPO_DIR, "configs", "dss_vit_config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

IMAGE_SIZE = CONFIG["dataset"]["image_size"]
VAL_FRACTION = CONFIG["dataset"]["val_fraction"]
VAL_SEED = CONFIG["dataset"]["val_seed"]
# num_workers=0 avoids the harmless "_MultiProcessingDataLoaderIter.__del__"
# AssertionError warnings that Colab prints when a DataLoader with workers
# is garbage-collected at cell completion.
NUM_WORKERS = 0

# Batch size: config says 64 (Toubkal A100 default). A T4 has
# 15 GB VRAM — cap at 32 for Colab; drop to 16 if OOM.
BATCH_SIZE = CONFIG["training"]["batch_size"]
if BATCH_SIZE >= 64:
    print(f"⚠️  Config batch_size={BATCH_SIZE} -> capped to 32 for Colab T4.")
    print("   (use 16 if OOM, 64 on Toubkal A100)")
    BATCH_SIZE = 32

# ------------------------------------------------------------
# Colab-local paths (must match the original training run)
# ------------------------------------------------------------
EXPERIMENT_DIR = "/content/DSS_ViT_experiment"
CHECKPOINT_DIR = "/content/drive/MyDrive/HER2_Checkpoints/DSS-ViT"
LOG_DIR = os.path.join(EXPERIMENT_DIR, "logs")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
SPLIT_INDICES_PATH = os.path.join(EXPERIMENT_DIR, "split_indices.npz")
STAIN_STATS_PATH = os.path.join(EXPERIMENT_DIR, "stain_stats.json")

CONFIG["paths"] = {
    "experiment_name": "dss_vit_colab",
    "train_dir": str(TRAIN_DIR),
    "test_dir": str(TEST_DIR),
    "experiment_dir": EXPERIMENT_DIR,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir": LOG_DIR,
    "results_dir": RESULTS_DIR,
    "stain_stats_path": STAIN_STATS_PATH,
    "split_indices_path": SPLIT_INDICES_PATH,
}

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# Deterministic stratified split (seed 42) — same as training
# ------------------------------------------------------------
train_indices, val_indices = get_or_create_split_indices(
    train_dir=str(TRAIN_DIR),
    val_fraction=VAL_FRACTION,
    seed=VAL_SEED,
    save_path=SPLIT_INDICES_PATH,
)

train_transform = get_train_transform(image_size=IMAGE_SIZE)
test_transform = get_test_transform(image_size=IMAGE_SIZE)

full_train_dataset = HER2Dataset(root_dir=str(TRAIN_DIR), transform=train_transform)
train_dataset = Subset(full_train_dataset, train_indices)

val_dataset = HER2Dataset(root_dir=str(TRAIN_DIR), transform=test_transform)
val_subset = Subset(val_dataset, val_indices)

test_dataset = HER2Dataset(root_dir=str(TEST_DIR), transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("=" * 60)
print("Dataset Split")
print("=" * 60)
print(f"  Full train      : {len(full_train_dataset)}")
print(f"  Train (90%)     : {len(train_dataset)}")
print(f"  Validation (10%): {len(val_subset)}")
print(f"  Test (official) : {len(test_dataset)}")
print("=" * 60)

# ------------------------------------------------------------
# Load precomputed stain stats (or recompute if missing)
# ------------------------------------------------------------
from models_v2_1 import ColorDeconvolution, save_stain_stats

def _compute_stain_stats(train_dir, train_indices, image_size):
    """Recomputes global H/DAB mean/std on the 90% train subset."""
    stats_dataset = HER2Dataset(root_dir=train_dir, transform=get_test_transform(image_size=image_size))
    stats_subset = Subset(stats_dataset, train_indices)
    print(f"Recomputing stain stats on {len(stats_subset)} training images...")

    _deconv = ColorDeconvolution().to(device)
    _deconv.eval()

    _h_sum = 0.0
    _h_sq_sum = 0.0
    _dab_sum = 0.0
    _dab_sq_sum = 0.0
    _total_pixels = 0

    _stats_loader = DataLoader(
        stats_subset,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    with torch.no_grad():
        for images, _ in _stats_loader:
            images = images.to(device)
            h_channel, dab_channel = _deconv(images)

            h_flat = h_channel.flatten()
            dab_flat = dab_channel.flatten()

            _h_sum += h_flat.sum().item()
            _h_sq_sum += (h_flat ** 2).sum().item()
            _dab_sum += dab_flat.sum().item()
            _dab_sq_sum += (dab_flat ** 2).sum().item()
            _total_pixels += h_flat.numel()

    _h_mean = _h_sum / _total_pixels
    _h_var = max(_h_sq_sum / _total_pixels - _h_mean ** 2, 0.0)
    _h_std = max(float(np.sqrt(_h_var)), 1e-6)

    _dab_mean = _dab_sum / _total_pixels
    _dab_var = max(_dab_sq_sum / _total_pixels - _dab_mean ** 2, 0.0)
    _dab_std = max(float(np.sqrt(_dab_var)), 1e-6)

    return {
        "h_mean": float(_h_mean),
        "h_std": float(_h_std),
        "dab_mean": float(_dab_mean),
        "dab_std": float(_dab_std),
    }

if os.path.exists(STAIN_STATS_PATH):
    stain_stats = load_stain_stats(STAIN_STATS_PATH)
    print("=" * 60)
    print("Stain Statistics (loaded from training run)")
    print("=" * 60)
else:
    print("⚠️  stain_stats.json not found in /content (fresh session).")
    print("   Recomputed it from the 90% train subset (deterministic, seed 42).")
    stain_stats = _compute_stain_stats(str(TRAIN_DIR), train_indices, IMAGE_SIZE)
    save_stain_stats(
        h_mean=stain_stats["h_mean"],
        h_std=stain_stats["h_std"],
        dab_mean=stain_stats["dab_mean"],
        dab_std=stain_stats["dab_std"],
        path=STAIN_STATS_PATH,
    )

print(f"  H mean   : {stain_stats['h_mean']:.6f}")
print(f"  H std    : {stain_stats['h_std']:.6f}")
print(f"  DAB mean : {stain_stats['dab_mean']:.6f}")
print(f"  DAB std  : {stain_stats['dab_std']:.6f}")
print("=" * 60)


# ============================================================
# Cell 6 — Build Model + Load Best Stage 3 Checkpoint
# ============================================================
# Builds the DSSViT model, sets Stage 3 freeze/LR config, builds
# the Stage 3 optimizer + cosine scheduler, then loads the BEST
# Stage 3 checkpoint (model + optimizer + scheduler state).
#
# The scheduler state is restored so the cosine LR schedule
# continues from the checkpoint epoch, not from epoch 0.

from models_v2_1 import DSSViT
from utils.train_dss_vit import (
    setup_logging,
    build_optimizer,
    set_stage_requires_grad,
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    load_checkpoint,
)
from utils.metrics_dss_vit import compute_metrics, print_metrics

STAGE3_EPOCHS = CONFIG["stage3"]["epochs"]
BEST_S3_CKPT = os.path.join(CHECKPOINT_DIR, "best_stage3.pt")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "last.pt")

logger = setup_logging(LOG_DIR, rank=0)
logger.info(f"Device: {device}")

# AMP scaler
use_amp = CONFIG["training"].get("amp", True) and torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
logger.info(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")

# Build model
model = DSSViT(
    num_classes=CONFIG["model"]["num_classes"],
    pretrained=CONFIG["model"]["pretrained"],
    num_stain_tokens=CONFIG["model"]["num_stain_tokens"],
    stain_bottleneck_dim=CONFIG["model"]["stain_bottleneck_dim"],
    stain_stats=stain_stats,
    image_size=CONFIG["model"]["image_size"],
).to(device)

# Stage 3 freeze/LR config
set_stage_requires_grad(model, stage=3, config=CONFIG)
optimizer = build_optimizer(model, CONFIG, stage=3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE3_EPOCHS)

assert os.path.exists(BEST_S3_CKPT), f"Best Stage 3 checkpoint not found: {BEST_S3_CKPT}"

# Load model + optimizer + scheduler state from best_stage3.pt
ckpt = load_checkpoint(
    path=BEST_S3_CKPT,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)
model.eval()

# Resume state: number of completed epochs + best val acc so far
start_epoch = ckpt.get("epoch", 0)  # 1-based count of completed epochs
best_acc = ckpt.get("metrics", {}).get("accuracy", 0.0) * 100  # fraction -> %
best_metrics = ckpt.get("metrics", {})

print("=" * 60)
print("Best Stage 3 Checkpoint Loaded (resume)")
print("=" * 60)
print(f"  Checkpoint    : {BEST_S3_CKPT}")
print(f"  Stage         : {ckpt.get('stage', 'N/A')}")
print(f"  Completed epochs : {start_epoch} / {STAGE3_EPOCHS}")
print(f"  Best val acc  : {best_acc:.2f}%")
print(f"  Resuming from epoch {start_epoch} to {STAGE3_EPOCHS}")
print("=" * 60)


# ============================================================
# Cell 7 — Continue Stage 3 Training
# ============================================================
# Continues Stage 3 from the checkpoint epoch until the full
# STAGE3_EPOCHS (40) are complete. The cosine scheduler resumes
# from the restored state. New best models overwrite
# best_stage3.pt; the latest state is saved to last.pt.

print("=" * 60)
print("Stage 3 — Continue (Joint Optimization, ViT unfrozen)")
print(f"  vit new_lr : {CONFIG['stage3']['vit_lr']}")
print(f"  new lr     : {CONFIG['stage3']['new_lr']}")
print(f"  resuming   : epoch {start_epoch} -> {STAGE3_EPOCHS}")
print("=" * 60)

for epoch in range(start_epoch, STAGE3_EPOCHS):
    train_total, train_ce, train_ord_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        config=CONFIG,
        epoch=epoch,
        logger=logger,
        rank=0,
        debug=CONFIG["training"]["debug"],
    )

    scheduler.step()

    val_total, val_ce, val_ord_loss, val_acc, val_preds, val_labels = validate_one_epoch(
        model=model,
        dataloader=val_loader,
        device=device,
        config=CONFIG,
        logger=logger,
        rank=0,
        debug=CONFIG["training"]["debug"],
    )

    metrics = compute_metrics(val_labels, val_preds, full_train_dataset.get_class_names())
    metrics.update({
        "val_total_loss": val_total,
        "val_ce_loss": val_ce,
        "val_ord_loss": val_ord_loss,
        "train_total_loss": train_total,
        "train_ce_loss": train_ce,
        "train_ord_loss": train_ord_loss,
        "train_acc": train_acc,
        "epoch": epoch + 1,
        "stage": 3,
    })

    print(
        f"Stage 3 | Epoch [{epoch+1:02d}/{STAGE3_EPOCHS}] | "
        f"Train Loss {train_total:.4f} (CE {train_ce:.4f}, Ord {train_ord_loss:.4f}) | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_total:.4f} | Val Acc {val_acc:.2f}% | "
        f"QWK {metrics['qwk']:.4f}"
    )

    # Best validation accuracy -> best_stage3.pt
    if val_acc > best_acc:
        best_acc = val_acc
        best_metrics = metrics
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            stage=3,
            metrics=metrics,
            config=CONFIG,
            split_indices_path=SPLIT_INDICES_PATH,
            save_path=BEST_S3_CKPT,
            rank=0,
        )
        print(f"  ✅ New best Stage 3 model saved (Epoch {epoch+1} | Val Acc: {val_acc:.2f}%)")

# Save the last checkpoint (used by --resume / continued runs)
save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=STAGE3_EPOCHS,
    stage=3,
    metrics=best_metrics,
    config=CONFIG,
    split_indices_path=SPLIT_INDICES_PATH,
    save_path=LAST_CKPT,
    rank=0,
)
print(f"✅ Stage 3 resumed training finished. Best val acc: {best_acc:.2f}%")
print(f"   Best checkpoint : {BEST_S3_CKPT}")
print(f"   Last checkpoint : {LAST_CKPT}")


# ============================================================
# Cell 8 — Load Final Best Stage 3 Checkpoint
# ============================================================
# Loads the (possibly updated) best Stage 3 checkpoint into the
# model for evaluation. Self-contained: works right after Cell 7,
# or in a fresh session (rebuilds model from CONFIG).

# Self-contained imports (in case this cell is run in a fresh session)
import os
import sys
import torch
import yaml

from models_v2_1 import DSSViT, load_stain_stats
from utils.train_dss_vit import load_checkpoint

if "REPO_DIR" not in globals():
    REPO_DIR = "/content/DSCA-ViT"
if "device" not in globals():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, REPO_DIR)

# Rebuild CONFIG / paths if not already defined in this session
if "CONFIG" not in globals():
    with open(os.path.join(REPO_DIR, "configs", "dss_vit_config.yaml")) as f:
        CONFIG = yaml.safe_load(f)
    EXPERIMENT_DIR = "/content/DSS_ViT_experiment"
    CHECKPOINT_DIR = "/content/drive/MyDrive/HER2_Checkpoints/DSS-ViT"
    STAIN_STATS_PATH = os.path.join(EXPERIMENT_DIR, "stain_stats.json")

BEST_S3_CKPT = os.path.join(CHECKPOINT_DIR, "best_stage3.pt")

# Rebuild model if not already defined in this session
if "model" not in globals():
    stain_stats = load_stain_stats(STAIN_STATS_PATH)
    model = DSSViT(
        num_classes=CONFIG["model"]["num_classes"],
        pretrained=CONFIG["model"]["pretrained"],
        num_stain_tokens=CONFIG["model"]["num_stain_tokens"],
        stain_bottleneck_dim=CONFIG["model"]["stain_bottleneck_dim"],
        stain_stats=stain_stats,
        image_size=CONFIG["model"]["image_size"],
    ).to(device)

assert os.path.exists(BEST_S3_CKPT), f"Best Stage 3 checkpoint not found: {BEST_S3_CKPT}"

ckpt = load_checkpoint(
    path=BEST_S3_CKPT,
    model=model,
    device=device,
)
model.eval()

best_val = ckpt.get("metrics", {}).get("accuracy", "N/A")
if isinstance(best_val, (int, float)):
    best_val_str = f"{best_val * 100:.2f}%"
else:
    best_val_str = str(best_val)

print("=" * 60)
print("Final Best Stage 3 Checkpoint Loaded")
print("=" * 60)
print(f"  Checkpoint    : {BEST_S3_CKPT}")
print(f"  Stage         : {ckpt.get('stage', 'N/A')}")
print(f"  Epoch         : {ckpt.get('epoch', 'N/A')}")
print(f"  Best val acc  : {best_val_str}")
print("=" * 60)


# ============================================================
# Cell 9 — Validation Metrics
# ============================================================

val_total, val_ce, val_ord_loss, val_acc_pct, val_preds, val_labels = validate_one_epoch(
    model=model,
    dataloader=val_loader,
    device=device,
    config=CONFIG,
    logger=logger,
    rank=0,
    debug=False,
)

class_names = full_train_dataset.get_class_names()
val_metrics = compute_metrics(val_labels, val_preds, class_names)
print("=" * 60)
print("VALIDATION METRICS (final best Stage 3 checkpoint)")
print("=" * 60)
print_metrics(val_metrics)


# ============================================================
# Cell 10 — One-Time Test Evaluation
# ============================================================

test_total, test_ce, test_ord_loss, test_acc_pct, test_preds, test_labels = validate_one_epoch(
    model=model,
    dataloader=test_loader,
    device=device,
    config=CONFIG,
    logger=logger,
    rank=0,
    debug=False,
)

test_metrics = compute_metrics(test_labels, test_preds, test_dataset.get_class_names())
print("=" * 60)
print("FINAL TEST EVALUATION (official test split, evaluated once)")
print("=" * 60)
print_metrics(test_metrics)

# Persist test results
results = {
    "checkpoint": os.path.join(CHECKPOINT_DIR, "best_stage3.pt"),
    "resumed_from": "best_stage3.pt (continued Stage 3 to completion)",
    "metrics": {
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "qwk": test_metrics["qwk"],
        "per_class": test_metrics["per_class"],
    },
    "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    "class_names": class_names,
}
results_path = os.path.join(RESULTS_DIR, "test_results_resume.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Test results saved: {results_path}")


# ============================================================
# Cell 11 — Baseline Comparison
# ============================================================

print("=" * 60)
print("BASELINE COMPARISON (official test split)")
print("=" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'Balanced Acc':>14} {'Macro-F1':>10}")
print("-" * 60)
print(f"{'ViT baseline':<22} {'95.02%':>10} {'—':>14} {'—':>10}")
print(f"{'DSCA-ViT v1':<22} {'~92.26%':>10} {'—':>14} {'—':>10}")
print(f"{'DSCA-ViT v2':<22} {'87.22%':>10} {'—':>14} {'—':>10}")
print(f"{'DSCA-ViT v3':<22} {'TBD':>10} {'—':>14} {'—':>10}")
print(
    f"{'DSS-ViT (resumed)':<22} "
    f"{test_metrics['accuracy'] * 100:>9.2f}% "
    f"{test_metrics['balanced_accuracy'] * 100:>13.2f}% "
    f"{test_metrics['macro_f1']:>10.4f}"
)
print("-" * 60)
print("\nPer-class recall (DSS-ViT, resumed):")
for i, cls in enumerate(class_names):
    print(f"  {cls}: recall={test_metrics['per_class'][cls]['recall']:.4f}")
print("=" * 60)

# Validation -> test gap (the key generalization metric)
gap = val_metrics["accuracy"] * 100 - test_metrics["accuracy"] * 100
print("=" * 60)
print("GENERALIZATION GAP (validation - test)")
print("=" * 60)
print(f"  Validation acc : {val_metrics['accuracy'] * 100:.2f}%")
print(f"  Test acc       : {test_metrics['accuracy'] * 100:.2f}%")
print(f"  Gap            : {gap:+.2f} pp")
print("=" * 60)


# ============================================================
# Cell 12 — Fusion Gate Telemetry
# ============================================================
# Lightweight telemetry: how much did the stain branch contribute
# via the gated residual, and does the gate correlate with model
# confidence on a validation batch?

from scipy.stats import pearsonr, spearmanr

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)

        # Manual forward to capture per-sample gates (B, 768)
        h_channel, dab_channel = model.color_deconv(images)
        h_norm = (h_channel - model.h_mean) / model.h_std
        d_norm = (dab_channel - model.dab_mean) / model.dab_std
        stain_tokens = model.stain_encoder(torch.cat([h_norm, d_norm], dim=1))

        rgb_norm = (images - model.imagenet_mean) / model.imagenet_std
        features = model.vit.forward_features(rgb_norm)
        x_cls = features[:, 0]  # (B, 768)

        attn_out, _ = model.cross_attn(
            query=x_cls.unsqueeze(1),
            key=stain_tokens,
            value=stain_tokens,
        )
        attn_out = attn_out.squeeze(1)
        concat = torch.cat([x_cls, attn_out], dim=-1)
        gate = torch.sigmoid(model.gate_mlp(concat))  # (B, 768)
        fused_cls_val = x_cls + gate * attn_out        # (B, 768)

        # Full forward for probabilities + confidence
        pred = model(images)
        probs = pred["probs"]

        sample_gate = gate.mean(dim=1).cpu().numpy()        # (B,)
        confidence = probs.max(dim=1).values.cpu().numpy()  # (B,)
        break

torch.cuda.empty_cache()

g = gate.cpu().numpy()
print("=" * 60)
print("FUSION GATE TELEMETRY (validation batch)")
print("=" * 60)
print(f"  mean   : {g.mean():.4f}")
print(f"  std    : {g.std():.4f}")
print(f"  min    : {g.min():.4f}")
print(f"  max    : {g.max():.4f}")
print(f"  median : {np.median(g):.4f}")
print("=" * 60)

# Stain branch contribution: mean |fused - x_cls|
additive = (fused_cls_val - x_cls).abs().mean().item()
print("=" * 60)
print("STAIN BRANCH CONTRIBUTION")
print("=" * 60)
print(f"  mean |fused_cls - x_cls| : {additive:.4f}")
print("=" * 60)

# Gate/confidence correlation
pearson_r, _ = pearsonr(sample_gate, confidence)
spearman_r, _ = spearmanr(sample_gate, confidence)
print("=" * 60)
print("GATE / CONFIDENCE CORRELATION")
print("=" * 60)
print(f"  Pearson  : {pearson_r:.4f}")
print(f"  Spearman : {spearman_r:.4f}")
print("=" * 60)

print("\n✅ DSS-ViT resumed Stage 3 training complete.")


# ============================================================
# Cell 13 — Final Report
# ============================================================

print("=" * 60)
print("DSS-ViT RESUMED STAGE 3 — FINAL REPORT")
print("=" * 60)
print(f"  Validation accuracy : {val_metrics['accuracy'] * 100:.2f}%")
print(f"  Test accuracy       : {test_metrics['accuracy'] * 100:.2f}%")
print(f"  Generalization gap  : {gap:+.2f} pp")
print()
print("  Baseline comparison (official test):")
print(f"    ViT baseline       : 95.02%")
print(f"    Original DSCA-ViT  : ~92.26%")
print(f"    DSCA-ViT v2        : 87.22%")
print(f"    DSCA-ViT v3        : TBD")
print(f"    DSS-ViT (resumed)  : {test_metrics['accuracy'] * 100:.2f}%")
print()
print("  Success criteria:")
print("    - Beat the plain ViT baseline on official test (95.02%)")
print("    - Improve on v2 (87.22%) with a smaller val->test gap")
print("=" * 60)