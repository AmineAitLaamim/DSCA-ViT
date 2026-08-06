# DSCA-ViT Training Notebook
# This file is written as a Python script with cell markers.
# Paste each section into a Google Colab cell.

# ============================================================
# Cell 1 — Environment Setup
# ============================================================

# Mount Google Drive
from google.colab import drive
import os

drive.mount("/content/drive")

if os.path.exists("/content/drive/MyDrive"):
    print("✅ Google Drive mounted successfully.")
else:
    raise RuntimeError("❌ Google Drive was not mounted correctly.")


# ============================================================
# Cell 2 — Clone / Pull Repository
# ============================================================

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

# Add repository to Python path
import sys
sys.path.insert(0, REPO_DIR)

print(f"REPO_DIR: {REPO_DIR}")


# ============================================================
# Cell 3 — Install Dependencies
# ============================================================

subprocess.run(
    ["pip", "install", "timm", "pyyaml", "--quiet"],
    check=True
)
print("✅ Dependencies installed.")


# ============================================================
# Cell 4 — Imports & Reproducibility
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"PyTorch Version : {torch.__version__}")
print(f"Device          : {device}")
print("=" * 60)


# ============================================================
# Cell 5 — Download & Prepare Dataset
# ============================================================

import zipfile

DATA_ROOT = Path("/content/HER2_Dataset")
DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run(
        ["wget", "-O", str(ZIP_PATH), URL],
        check=True
    )
else:
    print("Dataset archive already exists.")

WSI_DIR = DATA_ROOT / "WSI-based-dataset"

if not WSI_DIR.exists():
    print("Extracting main archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Main archive extracted.")

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

TRAIN_DIR = WSI_DIR / "train"
TEST_DIR = WSI_DIR / "test"

assert TRAIN_DIR.exists(), f"Train directory not found: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Test directory not found: {TEST_DIR}"

print("\nDataset location:")
print(TRAIN_DIR)
print(TEST_DIR)

print("\nFolder structure:")
for folder in [TRAIN_DIR, TEST_DIR]:
    print(f"\n{folder.name}/")
    for cls in sorted(os.listdir(folder)):
        cls_path = folder / cls
        if cls_path.is_dir():
            n = len(list(cls_path.iterdir()))
            print(f"   {cls:<10} {n:5d} images")


# ============================================================
# Cell 6 — Configuration
# ============================================================

# Experiment identity
BACKBONE_NAME   = "DSCA_ViT"
MODEL_ID        = "dsca_vit_b16"
NUM_CLASSES     = 4
IMAGE_SIZE      = 224
BATCH_SIZE      = 32

# Checkpoint paths
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
EXPERIMENT_DIR  = os.path.join(CHECKPOINT_ROOT, BACKBONE_NAME)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

print("=" * 60)
print("Experiment Configuration")
print("=" * 60)
print(f"Model           : {BACKBONE_NAME}")
print(f"Experiment Dir  : {EXPERIMENT_DIR}")
print("=" * 60)


# ============================================================
# Cell 7 — Datasets & Dataloaders
# ============================================================

from datasets import HER2Dataset, get_train_transform, get_test_transform
from torch.utils.data import DataLoader

train_transform = get_train_transform(image_size=IMAGE_SIZE)
test_transform  = get_test_transform(image_size=IMAGE_SIZE)

train_dataset = HER2Dataset(root_dir=TRAIN_DIR, transform=train_transform)
val_dataset   = HER2Dataset(root_dir=TEST_DIR,  transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Train images    : {len(train_dataset)}")
print(f"Val images      : {len(val_dataset)}")
print(f"Train batches   : {len(train_loader)}")
print(f"Val batches     : {len(val_loader)}")


# ============================================================
# Cell 8 — Build Model
# ============================================================

from models import DSCAViT

model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.5,
    classifier_dropout=0.1,
)

model = model.to(device)

# Parameter summary
counts = model.count_parameters()
print("=" * 60)
print("DSCA-ViT Parameter Summary")
print("=" * 60)
for name, count in counts.items():
    print(f"  {name:<20} : {count:>12,}")
print("=" * 60)


# ============================================================
# Cell 9 — Stage 1: Train New Components (Encoder Frozen)
# ============================================================

from utils import train_one_epoch, validate_one_epoch, save_checkpoint

# ----------------------------------------------------------
# Freeze shared encoder; train only new components
# ----------------------------------------------------------

for param in model.encoder.parameters():
    param.requires_grad = False

# Everything else is trainable
param_groups = model.get_parameter_groups()

optimizer = optim.Adam(
    param_groups["new"],
    lr=1e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
)

criterion = nn.CrossEntropyLoss()

STAGE1_EPOCHS = 30
BEST_S1_PATH  = f"/content/best_stage1_{BACKBONE_NAME}.pth"

best_acc   = 0.0
best_epoch = 0

print("=" * 60)
print(f"Stage 1 — {BACKBONE_NAME}")
print("Encoder: Frozen | New components: Trainable")
print("=" * 60)

for epoch in range(STAGE1_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, preds, labels = validate_one_epoch(
        model, val_loader, criterion, device
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
        best_acc   = val_acc
        best_epoch = epoch + 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            val_acc=best_acc,
            save_path=BEST_S1_PATH,
            stage=1,
            model_name=BACKBONE_NAME,
            model_id=MODEL_ID,
        )
        print(f"  ✅ New best model saved (Epoch {best_epoch} | Val Acc: {best_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"Stage 1 Finished | Best: {best_acc:.2f}% @ Epoch {best_epoch}")
print(f"{'='*60}")


# ============================================================
# Cell 10 — Save Stage 1 Checkpoint to Google Drive
# ============================================================

import shutil
from datetime import datetime

SAVE_DIR_S1 = os.path.join(EXPERIMENT_DIR, "Stage1")
os.makedirs(SAVE_DIR_S1, exist_ok=True)

DEST_S1 = os.path.join(SAVE_DIR_S1, f"best_stage1_{BACKBONE_NAME}.pth")
shutil.copy2(BEST_S1_PATH, DEST_S1)

size_mb = os.path.getsize(DEST_S1) / 1024 / 1024
print(f"✅ Stage 1 checkpoint saved to Google Drive.")
print(f"   Path : {DEST_S1}")
print(f"   Size : {size_mb:.2f} MB")


# ============================================================
# Cell 11 — Stage 2: Full Fine-tuning
# ============================================================

from utils import load_checkpoint

# ----------------------------------------------------------
# Load best Stage 1 weights
# ----------------------------------------------------------

load_checkpoint(
    path=DEST_S1,
    model=model,
    device=device,
)
print("✅ Stage 1 weights loaded.")

# ----------------------------------------------------------
# Unfreeze entire model
# ----------------------------------------------------------

for param in model.parameters():
    param.requires_grad = True

param_groups = model.get_parameter_groups()

ENCODER_LR   = 1e-5
NEW_COMP_LR  = 1e-4

optimizer = optim.Adam(
    [
        {"params": param_groups["encoder"], "lr": ENCODER_LR},
        {"params": param_groups["new"],     "lr": NEW_COMP_LR},
    ]
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
)

STAGE2_EPOCHS = 30
BEST_S2_PATH  = f"/content/best_stage2_{BACKBONE_NAME}.pth"

best_s2_acc   = best_acc      # Start from Stage 1 best
best_s2_epoch = best_epoch

print("=" * 60)
print(f"Stage 2 — {BACKBONE_NAME}")
print("Full model fine-tuning")
print(f"Encoder LR: {ENCODER_LR} | New components LR: {NEW_COMP_LR}")
print(f"Starting from Stage 1 best: {best_acc:.2f}%")
print("=" * 60)

for epoch in range(STAGE2_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, preds, labels = validate_one_epoch(
        model, val_loader, criterion, device
    )

    scheduler.step()

    print(
        f"Epoch [{epoch+1:02d}/{STAGE2_EPOCHS}] | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.2f}%"
    )

    if val_acc > best_s2_acc:
        best_s2_acc   = val_acc
        best_s2_epoch = epoch + 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_s2_epoch,
            val_acc=best_s2_acc,
            save_path=BEST_S2_PATH,
            stage=2,
            model_name=BACKBONE_NAME,
            model_id=MODEL_ID,
            config={
                "encoder_lr":  ENCODER_LR,
                "new_comp_lr": NEW_COMP_LR,
                "batch_size":  BATCH_SIZE,
                "epochs":      STAGE2_EPOCHS,
            },
        )
        print(f"  ✅ New best model saved (Epoch {best_s2_epoch} | Val Acc: {best_s2_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"Stage 2 Finished | Best: {best_s2_acc:.2f}% @ Epoch {best_s2_epoch}")
print(f"{'='*60}")


# ============================================================
# Cell 12 — Save Stage 2 Checkpoint to Google Drive
# ============================================================

SAVE_DIR_S2 = os.path.join(EXPERIMENT_DIR, "Stage2")
os.makedirs(SAVE_DIR_S2, exist_ok=True)

DEST_S2 = os.path.join(SAVE_DIR_S2, f"best_stage2_{BACKBONE_NAME}.pth")
shutil.copy2(BEST_S2_PATH, DEST_S2)

# Save weights-only file
weights_path = os.path.join(SAVE_DIR_S2, f"weights_{BACKBONE_NAME}.pth")
torch.save(model.state_dict(), weights_path)

size_mb = os.path.getsize(DEST_S2) / 1024 / 1024
print(f"✅ Stage 2 checkpoint saved.")
print(f"   Checkpoint : {DEST_S2}")
print(f"   Weights    : {weights_path}")
print(f"   Size       : {size_mb:.2f} MB")


# ============================================================
# Cell 13 — Final Evaluation
# ============================================================

from utils import compute_metrics, print_metrics
import numpy as np

# ----------------------------------------------------------
# Load best Stage 2 model
# ----------------------------------------------------------

load_checkpoint(path=DEST_S2, model=model, device=device)
model.eval()

print("=" * 60)
print(f"Final Evaluation — {BACKBONE_NAME}")
print("=" * 60)

all_labels      = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds   = outputs.argmax(dim=1)

        all_predictions.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

all_labels      = np.array(all_labels)
all_predictions = np.array(all_predictions)

class_names = train_dataset.get_class_names()

metrics = compute_metrics(all_labels, all_predictions, class_names)
print_metrics(metrics)

# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------

import csv

RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# TXT report
txt_path = os.path.join(RESULTS_DIR, f"evaluation_{BACKBONE_NAME}.txt")
with open(txt_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Model    : {BACKBONE_NAME}\n")
    f.write(f"Model ID : {MODEL_ID}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy  : {metrics['accuracy']:.4f}%\n")
    f.write(f"Precision : {metrics['precision']:.4f}\n")
    f.write(f"Recall    : {metrics['recall']:.4f}\n")
    f.write(f"F1        : {metrics['f1']:.4f}\n\n")
    f.write("Classification Report\n")
    f.write("-" * 40 + "\n")
    f.write(metrics["classification_report"])

# CSV comparison table (append row)
csv_path = os.path.join(CHECKPOINT_ROOT, "experiment_results.csv")
file_exists = os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "Best Epoch"])
    writer.writerow([
        BACKBONE_NAME,
        round(metrics["accuracy"], 4),
        round(metrics["precision"], 4),
        round(metrics["recall"], 4),
        round(metrics["f1"], 4),
        best_s2_epoch,
    ])

print(f"\n✅ Results saved:")
print(f"   TXT : {txt_path}")
print(f"   CSV : {csv_path}")
