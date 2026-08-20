# ============================================================
# DSS-ViT — Training Notebook (Colab-ready)
# ============================================================
# End-to-end training: download data → Stage 1 → Stage 2 →
# Stage 3 → evaluate official test → compare vs ViT baseline.
#
# Each cell below is separated by the marker:
#   # ====...==== / # Cell N
# ============================================================

# ------------------------------------------------------------
# Cell 1 — Environment & Mount Drive
# ------------------------------------------------------------
from google.colab import drive
import os

drive.mount("/content/drive")

# ------------------------------------------------------------
# Cell 2 — Clone / Pull Repository
# ------------------------------------------------------------
import subprocess

REPO_URL = "https://github.com/AmineAitLaamim/DSS-ViT.git"
REPO_DIR = "/content/DSS-ViT"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

import sys
sys.path.insert(0, REPO_DIR)

# ------------------------------------------------------------
# Cell 3 — Install Dependencies
# ------------------------------------------------------------
subprocess.run(["pip", "install", "timm", "pyyaml", "--quiet"], check=True)

# ------------------------------------------------------------
# Cell 4 — Imports & Reproducibility
# ------------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
import yaml

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"Device   : {device}")
print("=" * 60)

# ------------------------------------------------------------
# Cell 5 — Download & Extract Dataset
# ------------------------------------------------------------
import zipfile
from pathlib import Path

DATA_ROOT = Path("/content/HER2_Dataset")
ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

if not ZIP_PATH.exists():
    print("Downloading dataset...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), URL], check=True)
else:
    print("Dataset archive already exists.")

WSI_DIR = DATA_ROOT / "WSI-based-dataset"

if not WSI_DIR.exists():
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Dataset extracted.")

nested = [WSI_DIR / "train_data_wsi.zip", WSI_DIR / "test_data_wsi.zip"]
for archive in nested:
    folder = archive.parent / archive.stem.replace("_data_wsi", "")
    if not folder.exists():
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(folder)
        archive.unlink()

print("\nDataset location:")
print(WSI_DIR)
assert WSI_DIR.exists(), "Train directory not found."
print("Dataset ready!")

# ------------------------------------------------------------
# Cell 5 — Show directory structure
# ------------------------------------------------------------
import os
print("\nFolder structure:")
for folder in [WSI_DIR]:
    print(f"\n{folder.name}/")
    for cls in sorted(os.listdir(folder)):
        p = folder / cls
        if p.is_dir():
            n = len(os.listdir(p))
            print(f"   {cls:<10} {n:5d} images")

# ------------------------------------------------------------
# Cell 5b — Configuration
# ------------------------------------------------------------
BACKBONE_NAME = "DSS_ViT"
MODEL_ID = "dss_vit_b16"
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 32

CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
EXPERIMENT_DIR = os.path.join(CHECKPOINT_ROOT, BACKBONE_NAME)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

print("=" * 60)
print("Experiment Configuration")
print("=" * 60)
print(f"Model           : {BACKBONE_NAME}")
print(f"Experiment Dir  : {EXPERIMENT_DIR}")
print("=" * 60)

# ------------------------------------------------------------
# Cell 6 — Datasets & DataLoaders
# ------------------------------------------------------------
from datasets import HER2Dataset, get_train_transform, get_test_transform
from torch.utils.data import DataLoader

train_transform = get_train_transform(image_size=IMAGE_SIZE)
test_transform  = get_test_transform(image_size=IMAGE_SIZE)

train_dataset = HER2Dataset(root_dir=WSI_DIR, transform=train_transform)
test_dataset   = HER2Dataset(root_dir=TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train images : {len(train_dataset)}")
print(f"Test images  : {len(test_dataset)}")
print(f"Train batches: {len(train_loader)}")
print(f"Test batch counts: {len(test_loader)}")

# ------------------------------------------------------------
# Cell 7 — Build Model
# ------------------------------------------------------------
from models import DSSViT

model = DSSViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.1,
    classifier_dropout=0.1,
)
model = model.to(device)

counts = model.count_parameters()
print("=" * 60)
print("DSS-ViT Parameter Summary")
print("=" * 60)
for name, count in counts.items():
    print(f"  {name:<20} : {count:>12,}")
print("=" * 60)

# ------------------------------------------------------------
# Cell 8 — Stage 1: Train New Components (Encoder Frozen)
# ------------------------------------------------------------
from utils import train_one_epoch, validate_one_epoch, save_checkpoint
import torch.optim as optim

for param in model.encoder.parameters():
    param.requires_grad = False

param_groups = model.get_parameter_groups()
optimizer = optim.Adam(
    param_groups["new"],
    lr=1e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

criterion = nn.CrossEntropyLoss()

STAGE1_EPOCHS = 30
BEST_S1_PATH = f"/content/best_stage1_{BACKBONE_NAME}.pth"
best_acc = 0.0
best_epoch = 0

print("=" * 60)
print(f"Stage 1 — {BACKBONE_NAME}")
print("Encoder: Frozen | New components: Trainable")
print("=" * 60)

for epoch in range(STAGE1_EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, _, _ = validate_one_epoch(
        model, test_loader, criterion, device
    )

    scheduler.step()

    print(
        f"Epoch [{epoch+1:02d}/{STAGE1_EPOCHS}] | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.2f}%"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            stage=1,
            metrics={"val_acc": best_acc},
            config=config,
            split_indices_path=SPLIT_INDICES_PATH,
            save_path=BEST1_PATH,
        )
        print(f"  ✅ New best model saved (Epoch {best_epoch} | Val Acc {best_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"Stage 1 Finished | Best: {best_acc:.2f}% @ Epoch {best_epoch}")
print(f"{'='*60}")

# ------------------------------------------------------------
# Cell 9 — Save Stage 1 Checkpoint to Drive
# ------------------------------------------------------------
import shutil

SAVE_DIR_S1 = os.path.join(EXPERIMENT_DIR, "Stage1")
os.makedirs(SAVE_DIR_S1, exist_ok=True)

DEST_S1 = os.path.join(SAVE_DIR_S1, f"best_stage1_{BACKBONE_NAME}.pth")
shutil.copy2(BEST1_PATH, DEST_S1)

size_mb = os.path.getsize(DEST_S1) / 1024 / 1024
print(f"✅ Stage 1 checkpoint saved to Google Drive.")
print(f"   Path : {DEST_S1}")
print(f"   Size : {size_mb:.2f} MB")

# ============================================================
# Cell 10 — Stage 2: Full Fine-tuning
# =============================================================
# Load Stage 1 best checkpoint
from utils import load_checkpoint

BEST_S1 = f"/content/best_stage1_{BACKBONE_NAME}.pth"
if not os.path.exists(DEST_S1):
    print(f"⚠️ Drive checkpoint not found at:\n    {DEST_S1}")
    print(f"    Falling back to local: {BEST_S1_PATH}")
    DEST_S1 = BEST_S1_PATH

assert os.path.exists(DEST_S1), (
    f"Stage 1 checkpoint not found.\n"
    f"    Checked Drive : {os.path.join(EXPERIMENT_DIR, 'Stage1', f'best_stage1_{BACKBONE_NAME}.pth')}\n"
    f"    Checked local : {BEST_S1_PATH}\n"
    f"    Run Cell 8 (Stage 1) or Cell 9 to copy to Drive."
)
print(f"✅ Loading Stage 1 checkpoint:\n    {DEST_S1}")
checkpoint_s1 = load_checkpoint(path=DEST_S1, model=model, device=device)
print("✅ Stage 1 weights loaded.")

# ----------
# Unfreeze entire model
# ----------
for param in model.parameters():
    param.requires_grad = True

param_groups = model.get_parameter_groups()
optimizer_all = optim.Adam(
    [
        {"params": param_groups["encoder"], "lr": 1e-5},
        {"params": param_groups["new"],     "lr": 1e-4},
    ]
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_all, T_max=30)
criterion = nn.CrossEntropyLoss()

STAGE2_EPOCHS = 30
BEST2_PATH = f"/content/best_stage2_{BACKBONE_NAME}.pth"
best2_acc = checkpoint_s1.get("best_val_accuracy", 0.0)
best2_epoch = checkpoint_s1.get("epoch", 0)

print("=" * 60)
print(f"Stage 2 — {BACKBONE_NAME}")
print("Full fine-tuning")
print(f"Encoder LR: {ENCODER_LR} | New components LR: {NEW_COMP_LR}")
print(f"Starting from Stage 1 best: {best2_acc:.2f}%")
print("=" * 60)

for epoch in range(STAGE2_EPOCHS):
    train_total, train_ce, train_ord, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_total, val_ce, val_ord, val_acc, preds, labels = validate_one_epoch(
        model, test_loader, criterion, device
    )

    scheduler.step()

    print(
        f"Epoch [{epoch+1:02d}/{STAGE2_EPOCHS}] | "
        f"Train Loss {train_total:.4f} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_total:.4f} | "
        f"Val Acc {val_acc:.2f}%"
    )

    if val_acc > best2_acc:
        best2_acc = val_acc
        best2_epoch = epoch + 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best2_epoch,
            stage=2,
            metrics={"val_acc": best2_acc},
            config=CONFIG,
            split_indices_path=SPLIT_INDICES_PATH,
            save_path=BEST2_PATH,
        )
        print(f"  ✅ New best model saved (Epoch {best2_epoch} | Val Acc {best2_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"Stage 2 Finished | Best: {best2_acc:.2f}% @ Epoch {best2_epoch}")
print(f"{'='*60}")

# ============================================================
# Cell 12 — Save Stage 2 Checkpoint to Drive
# ============================================================
import shutil

S2_SAVE_DIR = os.path.join(EXPERIMENT_DIR, "Stage2")
os.makedirs(S2_SAVE_DIR, exist_ok=True)

S2_DEST = os.path.join(S2_SAVE_DIR, f"best_stage2_{BACKBONE_NAME}.pth")
shutil.copy2(BEST2_PATH, S2_DEST)

S2_SIZE_MB = os.path.getsize(S2_DEST) / 1024 / 1024
print(f"✅ Stage 2 checkpoint saved to Google Drive.")
print(f"   Path : {S2_DEST}")
print(f"   Size : {S2_SIZE_MB:.2f} MB")

# ------------------------------------------------------------
# Cell 13 — Final Evaluation on Official Test Set
# ------------------------------------------------------------
# Locate best Stage2 checkpoint (prefer Drive, fallback local).
DEST_S2 = os.path.join(EXPERIMENT_DIR, "Stage2", f"best_stage2_{BACKBONE_NAME}.pth")
if not os.path.exists(DEST_S2):
    print(f"⚠️ Drive checkpoint not found:\n    {DEST_S2}")
    print(f"    Falling back to local: {BEST_S2_PATH}")
    DEST_S2 = BEST_S2_PATH

assert os.path.exists(DEST_S2), (
    f"Stage 2 checkpoint not found.\n"
    f"    Checked Drive : {os.path.join(EXPERIMENT_DIR, 'Stage2', f'best_stage2_{BACKBONE_NAME}.pth')}\n"
    f"    Checked local : {BEST_S2_PATH}\n"
    f"    Run Cell 11 (Stage 2) or Cell 12 to copy to Drive."
)
print(f"✅ Loading Stage 2 checkpoint:\n    {EST_S2}")
checkpoint_s2 = load_checkpoint(path=DEST_S2, model=model, device=device)
print("✅ Stage 2 weights loaded.")

# ----------
# Unfreeze entire model + rebuild optimizer
# ----------
for param in model.parameters():
    param.requires_grad = True

param_groups = model.get_parameter_groups()
optimizer_all = optim.Adam(
    [
        {"params": param_groups["encet"], "lr": 1e-5},
        {"params": param_groups["new"],     "lr": 1e-4},
    ]
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_all, T_est=30)
criterion = nn.CrossEntropyLoss()

STAGE2_EPOCHS = 30
BEST2_PATH = f"/content/best_stage2_{BACKBONE_NAME}.pth"
STAGE2_BEST_ACC = checkpoint_s1.get("best_val_acc", 0.0)
STAGE2_BEST_EPOCH = checkpoint_s1.get("epoch", 0)

print("=" * 60)
print(f"Stage 2 — {BACKBONE_NAME}")
print("Full model fine-tuning")
print(f"Encoder LR: {ENCODER_LR} | New components LR: {NEW_COMP_LR}")
print(f"Starting from Stage 1 best: {STAGE2_BEST_ACC:.2f}%")
print("=" * 60)

for epoch in range(STAGE2_EPOCHS):
    train_total, train_ce, train_ord, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_total, val_ce, val_ord, val_acc, _, _ = validate_one_epoch(model, test_loader, criterion, device)

    scheduler.step()

    print(f"Epoch [{epoch+1:02d}/{STAGE2_EPOCHS}] | Train Loss {train_total:.4f} | Train Acc {train_acc:.2f}% | Val Loss {val_total:.4f} | Val Acc {val_acc:.2f}%")

    if val_acc > STAGE2_BEST_ACC:
        STAGE2_BEST_ACC = val_acc
        STAGE2_BEST_EPOCH = epoch + 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best2_epoch,
            stage=2,
            metrics={"val_acc": best2_acc},
            config=CONFIG,
            split_indices_path=SPLIT_INDICES_PATH,
            save_path=BEST2_PATH,
        )
        print(f"  ✅ New best model saved (Epoch {best2_epoch} | Val Acc {best2_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"Stage 2 Finished | Best: {best2_acc:.2f}% @ Epoch {best2_epoch}")
print(f"{'='*60}")

# ============================================================
# Cell 12b — Save Stage2 Checkpoint to Drive
# ============================================================
import shutil

S2_SAVE_DIR = os.path.join(EXPERIMENT_DIR, "Stage2")
os.makedirs(S2_SAVE_DIR, exist_ok=True)

S2_DEST = os.path.join(S2_SAVE_DIR, f"best_stage2_{BACKBONE_NAME}.pth")
shutil.copy2(BEST_S2_PATH etxek, S2_DEST)

S2_SIZE_MB = os.path.getsize(S2_DEST) / 1024 / 1024
print(f"✅ Stage 2 checkpoint saved to Google Drive.")
print(f"   Path : {S2_DEST}")
print(f"   Size : {S2_SIZE_MB:.2f} MB")