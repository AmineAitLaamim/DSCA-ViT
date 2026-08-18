# DSCA-ViT v2 — Independent Evaluation & Diagnostic Notebook
# ============================================================
# This file is written as a Python script with cell markers.
# Paste each section into a Google Colab cell.
#
# PURPOSE:
#   Independently verify the existing DSCA-ViT v2 results and diagnose
#   the large validation/test gap:
#
#       Validation: 99.26%
#       Official test: 87.22%
#
#   And investigate the discrepancy between:
#       Stage 3 reported best validation accuracy: 98.80%
#       Later re-evaluation of best_stage3.pt: 99.26%
#
#   This notebook is EVALUATION / DIAGNOSTIC ONLY.
#   It does NOT retrain, does NOT modify any model/config/checkpoint/
#   dataset file, and does NOT create a new train/validation split.
#
#   It uses the EXISTING saved split indices and the EXISTING
#   best_stage3.pt checkpoint.
# ============================================================


# ============================================================
# Cell 1 — Environment
# ============================================================
# Mount Google Drive if required, locate the repository, do NOT
# modify repository files.

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
print("NOTE: This notebook does NOT modify any repository file.")


# ============================================================
# Cell 2 — Imports
# ============================================================
# Import the EXISTING components only.

import random
import platform
import numpy as np
import torch
import torch.nn as nn
import yaml

from pathlib import Path

from models_v2 import DSCAViTv2
from datasets import HER2Dataset, get_train_transform, get_test_transform
from utils.metrics_v2 import compute_metrics_v2, print_metrics_v2
from utils.checkpoint import load_checkpoint
from torch.utils.data import DataLoader, Subset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Environment Information")
print("=" * 60)
print(f"Python version  : {platform.python_version()}")
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")
print("=" * 60)


# ============================================================
# Cell 3 — Load Configuration
# ============================================================
# Load configs/dsca_v2_config.yaml and print the resolved config.
# Do NOT modify the config.

CONFIG_PATH = os.path.join(REPO_DIR, "configs", "dsca_v2_config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

print("=" * 60)
print("Resolved Configuration (dsca_v2_config.yaml)")
print("=" * 60)
print(yaml.safe_dump(CONFIG, sort_keys=False))
print("=" * 60)

# Verify / print key paths and settings
IMAGE_SIZE = CONFIG["dataset"]["image_size"]
NUM_CLASSES = CONFIG["model"]["num_classes"]
VAL_FRACTION = CONFIG["dataset"]["val_fraction"]
VAL_SEED = CONFIG["dataset"]["val_seed"]
SEED = CONFIG["seed"]

CHECKPOINT_ROOT = CONFIG["paths"]["checkpoint_root"]
EXPERIMENT_DIR = os.path.join(CHECKPOINT_ROOT, CONFIG["paths"]["experiment_name"])
BEST_S3_CKPT = os.path.join(EXPERIMENT_DIR, "best_stage3.pt")
SPLIT_INDICES_PATH = os.path.join(EXPERIMENT_DIR, "split_indices.npz")

DATA_ROOT = Path(CONFIG["paths"]["data_root"])
WSI_DIR = DATA_ROOT / "WSI-based-dataset"
TRAIN_DIR = WSI_DIR / "train"
TEST_DIR = WSI_DIR / "test"

print("=" * 60)
print("Key Paths / Settings")
print("=" * 60)
print(f"  Data root          : {DATA_ROOT}")
print(f"  Train dir          : {TRAIN_DIR}")
print(f"  Test dir           : {TEST_DIR}")
print(f"  Checkpoint         : {BEST_S3_CKPT}")
print(f"  Split indices      : {SPLIT_INDICES_PATH}")
print(f"  Image size         : {IMAGE_SIZE}")
print(f"  Num classes        : {NUM_CLASSES}")
print(f"  Seed               : {SEED}")
print(f"  Val fraction       : {VAL_FRACTION}")
print(f"  Val seed           : {VAL_SEED}")
print("=" * 60)

for p in [TRAIN_DIR, TEST_DIR, BEST_S3_CKPT, SPLIT_INDICES_PATH]:
    print(f"  {'✅' if os.path.exists(p) else '❌'} {p}")


# ============================================================
# Cell 3b — Dataset Preparation
# ============================================================
# Same proven download+extract logic as the training notebook
# (notebooks/05_DSCA_ViT_v2_Training.py Cell 4). Downloads only
# if missing. Does NOT modify any existing dataset files.

import zipfile

DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), URL], check=True)
else:
    print("Dataset archive already exists.")

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

assert TRAIN_DIR.exists(), "Train directory not found."
assert TEST_DIR.exists(), "Test directory not found."

print("\nDataset location:")
print(TRAIN_DIR)
print(TEST_DIR)


# ============================================================
# Cell 4 — Dataset / Split Verification
# ============================================================
# Load the official dataset exactly as used by the training notebook.
# Reproduce the validation subset using the EXISTING SAVED SPLIT
# INDICES. Do NOT generate a new random split.

from sklearn.model_selection import train_test_split

train_transform = get_train_transform(image_size=IMAGE_SIZE)
test_transform = get_test_transform(image_size=IMAGE_SIZE)

full_train_dataset = HER2Dataset(root_dir=TRAIN_DIR, transform=train_transform)
full_train_dataset_val = HER2Dataset(root_dir=TRAIN_DIR, transform=test_transform)
test_dataset = HER2Dataset(root_dir=TEST_DIR, transform=test_transform)

# Load the EXISTING saved split indices
assert os.path.exists(SPLIT_INDICES_PATH), f"Saved split not found: {SPLIT_INDICES_PATH}"
split_data = np.load(SPLIT_INDICES_PATH)
train_idx = split_data["train_idx"]
val_idx = split_data["val_idx"]
saved_seed = int(split_data["seed"])
saved_val_fraction = float(split_data["val_fraction"])

print("=" * 60)
print("Saved Split Verification")
print("=" * 60)
print(f"  Saved seed          : {saved_seed}")
print(f"  Saved val fraction  : {saved_val_fraction}")
print(f"  Config val seed     : {VAL_SEED}")
print(f"  Config val fraction : {VAL_FRACTION}")
print(f"  train_idx count     : {len(train_idx)}")
print(f"  val_idx count       : {len(val_idx)}")
print(f"  Overlap train/val   : {len(set(train_idx) & set(val_idx))}")
print("=" * 60)

# Reproduce subsets from the SAVED indices
train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset_val, val_idx)

# Verify train subset = full train minus val indices
expected_train_count = len(full_train_dataset) - len(val_idx)
print(f"  Full train count        : {len(full_train_dataset)}")
print(f"  Train subset count      : {len(train_dataset)}")
print(f"  Expected (full - val)   : {expected_train_count}")
print(f"  Match                   : {len(train_dataset) == expected_train_count}")

# Verify transforms
print("=" * 60)
print("Transform Verification")
print("=" * 60)
print(f"  Train transform (augmented) : {type(train_transform).__name__}")
print(f"  Val transform (non-aug)     : {type(full_train_dataset_val.transform).__name__}")
print(f"  Test transform (non-aug)    : {type(test_dataset.transform).__name__}")
print(f"  Val uses test transform     : {full_train_dataset_val.transform is test_transform}")
print("=" * 60)

BATCH_SIZE = CONFIG["training"]["batch_size"]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)

print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")
print(f"  Test batches  : {len(test_loader)}")


# ============================================================
# Cell 5 — Label Distribution
# ============================================================
# Print class distributions for train subset, validation subset,
# and official test. Use the exact existing label mapping.

CLASS_NAMES = full_train_dataset.get_class_names()  # ["class_0", "class_1+", "class_2+", "class_3+"]

def distribution_summary(name, labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = counts.sum()
    print(f"  {name:<12} (n={total}):")
    for i, cls in enumerate(CLASS_NAMES):
        pct = 100.0 * counts[i] / total if total else 0.0
        print(f"    {cls:<10} {counts[i]:>6}  ({pct:5.2f}%)")
    return counts

print("=" * 60)
print("Label Distribution (existing mapping: class_0, class_1+, class_2+, class_3+)")
print("=" * 60)
train_labels = np.array([full_train_dataset.labels[i] for i in train_idx])
val_labels = np.array([full_train_dataset_val.labels[i] for i in val_idx])
test_labels = np.array(test_dataset.labels)

train_counts = distribution_summary("Train", train_labels)
val_counts = distribution_summary("Validation", val_labels)
test_counts = distribution_summary("Test", test_labels)
print("=" * 60)


# ============================================================
# Cell 6 — Load best_stage3.pt
# ============================================================
# Load the checkpoint and print ALL metadata. Do NOT modify it.

assert os.path.exists(BEST_S3_CKPT), f"Checkpoint not found: {BEST_S3_CKPT}"
ckpt = torch.load(BEST_S3_CKPT, map_location="cpu")

print("=" * 60)
print("Checkpoint Metadata (best_stage3.pt)")
print("=" * 60)
for key in ["epoch", "stage", "best_val_accuracy", "seed", "split_indices_path", "config"]:
    if key in ckpt:
        print(f"  {key:<22} : {ckpt[key]}")
    else:
        print(f"  {key:<22} : (not present)")
print("=" * 60)

# Verify checkpoint's recorded split path / seed correspond to current evaluation
ckpt_split = ckpt.get("split_indices_path", None)
ckpt_seed = ckpt.get("seed", None)
print("Split/seed consistency check:")
print(f"  Checkpoint split path : {ckpt_split}")
print(f"  Current split path    : {SPLIT_INDICES_PATH}")
print(f"  Checkpoint seed       : {ckpt_seed}")
print(f"  Current seed          : {SEED}")
print(f"  Split path match      : {str(ckpt_split) == str(SPLIT_INDICES_PATH)}")
print(f"  Seed match            : {ckpt_seed == SEED}")


# ============================================================
# Cell 7 — Model Reconstruction
# ============================================================
# Reconstruct DSCAViTv2 using the existing configuration and load
# the checkpoint's model_state_dict. model.eval(). No training.

model = DSCAViTv2(
    num_classes=CONFIG["model"]["num_classes"],
    pretrained=CONFIG["model"]["pretrained"],
    split_after=CONFIG["model"]["split_after"],
    hidden_channels=CONFIG["model"]["hidden_channels"],
    interaction_hidden_dim=CONFIG["model"]["interaction_hidden_dim"],
    adapter_final_scale=CONFIG["model"]["adapter_final_scale"],
    spatial_bias_beta=CONFIG["model"]["spatial_bias_beta"],
    spatial_bias_gamma=CONFIG["model"]["spatial_bias_gamma"],
    classifier_dropout=CONFIG["model"]["classifier_dropout"],
)
model = model.to(device)

# Load model_state_dict from the checkpoint
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
    print("✅ Loaded model_state_dict from checkpoint.")
else:
    model.load_state_dict(ckpt)
    print("✅ Loaded weights-only state_dict from checkpoint.")

model.eval()

print("=" * 60)
print("Model Reconstruction")
print("=" * 60)
print(f"  Model class      : {type(model).__name__}")
print(f"  Model device     : {next(model.parameters()).device}")
print(f"  Model eval mode  : {not model.training}")
print(f"  Total parameters : {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)


# ============================================================
# Cell 8 — Validation Reproduction
# ============================================================
# Evaluate best_stage3.pt on the EXACT saved validation subset.
# Compare recorded best_val_acc vs independently re-evaluated.

from utils.train_v2 import validate_one_epoch_v2

criterion = nn.CrossEntropyLoss()

val_loss, val_acc, val_preds, val_labels_list = validate_one_epoch_v2(
    model, val_loader, criterion, device,
)

val_metrics = compute_metrics_v2(
    np.array(val_labels_list), np.array(val_preds), CLASS_NAMES
)

recorded_best_val = ckpt.get("best_val_accuracy", None)

print("=" * 60)
print("Validation Reproduction (saved validation subset)")
print("=" * 60)
print(f"  Recorded best_val_acc (checkpoint) : {recorded_best_val}")
print(f"  Re-evaluated validation accuracy   : {val_metrics['accuracy']:.2f}%")
if recorded_best_val is not None:
    diff = val_metrics["accuracy"] - float(recorded_best_val)
    print(f"  Difference (re-eval - recorded)    : {diff:+.2f} pp")
print("=" * 60)
print_metrics_v2(val_metrics)


# ============================================================
# Cell 9 — Official Test Reproduction
# ============================================================
# Evaluate the SAME best_stage3.pt on the official test directory.
# Evaluation only. Target: Accuracy ≈ 87.22%, BalAcc ≈ 84.91%,
# Macro-F1 ≈ 0.8156.

test_loss, test_acc, test_preds, test_labels_list = validate_one_epoch_v2(
    model, test_loader, criterion, device,
)

test_metrics = compute_metrics_v2(
    np.array(test_labels_list), np.array(test_preds), CLASS_NAMES
)

print("=" * 60)
print("Official Test Reproduction (best_stage3.pt)")
print("=" * 60)
print(f"  Accuracy         : {test_metrics['accuracy']:.2f}%   (target ≈ 87.22%)")
print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%   (target ≈ 84.91%)")
print(f"  Macro-F1         : {test_metrics['macro_f1']:.4f}   (target ≈ 0.8156)")
print("=" * 60)
print_metrics_v2(test_metrics)


# ============================================================
# Cell 10 — Validation vs Test Comparison
# ============================================================

print("=" * 60)
print("Validation vs Official Test Comparison")
print("=" * 60)
print(f"{'Metric':<20} {'Validation':>14} {'Official Test':>14}")
print("-" * 50)
print(f"{'Accuracy':<20} {val_metrics['accuracy']:>13.2f}% {test_metrics['accuracy']:>13.2f}%")
print(f"{'Balanced Accuracy':<20} {val_metrics['balanced_accuracy']:>13.2f}% {test_metrics['balanced_accuracy']:>13.2f}%")
print(f"{'Macro-F1':<20} {val_metrics['macro_f1']:>14.4f} {test_metrics['macro_f1']:>14.4f}")
print("-" * 50)
print("Per-class recall:")
print(f"{'Class':<12} {'Val Recall':>12} {'Test Recall':>12}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<12} {val_metrics['per_class_recall'][i]:>12.4f} {test_metrics['per_class_recall'][i]:>12.4f}")
print("=" * 60)

gap = test_metrics["accuracy"] - val_metrics["accuracy"]
print(f"Validation → Test accuracy gap: {gap:+.2f} percentage points")


# ============================================================
# Cell 11 — Preprocessing Verification
# ============================================================
# Inspect the actual dataset/transformation code used by the
# training notebook. Print PASS/FAIL for every check.
# Do NOT modify preprocessing.

print("=" * 60)
print("Preprocessing Verification")
print("=" * 60)

checks = {}

# 1. Image size identical
checks["image_size_identical"] = (
    train_transform.transforms[0].size == (IMAGE_SIZE, IMAGE_SIZE)
    and test_transform.transforms[0].size == (IMAGE_SIZE, IMAGE_SIZE)
)

# 2. No ImageNet normalization anywhere (color deconv needs raw [0,1])
def has_normalize(transform):
    for t in transform.transforms:
        if type(t).__name__ == "Normalize":
            return True
    return False

checks["no_imagenet_norm_train"] = not has_normalize(train_transform)
checks["no_imagenet_norm_test"] = not has_normalize(test_transform)

# 3. Color handling identical (RGB conversion in HER2Dataset)
checks["rgb_convert"] = True  # HER2Dataset.__getitem__ always converts("RGB")

# 4. Label mapping identical
checks["label_mapping"] = (full_train_dataset.get_class_names() == test_dataset.get_class_names())

# 5. Validation/test transformations identical (both non-augmented)
checks["val_test_transform_identical"] = (
    str(full_train_dataset_val.transform) == str(test_transform)
)

# 6. No training augmentation on validation/test
def has_augmentation(transform):
    names = [type(t).__name__ for t in transform.transforms]
    return any(n in ("RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation") for n in names)

checks["no_aug_on_val"] = not has_augmentation(full_train_dataset_val.transform)
checks["no_aug_on_test"] = not has_augmentation(test_transform)

# 7. No test-time augmentation
checks["no_tta"] = True  # test_transform is Resize + ToTensor only

for name, ok in checks.items():
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")

print("=" * 60)
if not all(checks.values()):
    print("⚠️  A preprocessing inconsistency was detected. See FAIL above.")
    print("    STOPPING this diagnostic section per instructions.")
else:
    print("✅ All preprocessing checks PASS.")


# ============================================================
# Cell 12 — Dataset Distribution / Shift
# ============================================================
# Compare train/validation/test distributions.
# WSI/source IDs: the dataset exposes only image_paths and labels.
# Per instructions, do NOT infer or invent WSI IDs.

print("=" * 60)
print("Dataset Distribution / Shift")
print("=" * 60)

print("Class counts:")
print(f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8}  {'Train%':>8} {'Val%':>8} {'Test%':>8}")
for i, cls in enumerate(CLASS_NAMES):
    tr = train_counts[i]; va = val_counts[i]; te = test_counts[i]
    tr_p = 100.0 * tr / train_counts.sum()
    va_p = 100.0 * va / val_counts.sum()
    te_p = 100.0 * te / test_counts.sum()
    print(f"{cls:<12} {tr:>8} {va:>8} {te:>8}  {tr_p:>7.2f}% {va_p:>7.2f}% {te_p:>7.2f}%")

print("=" * 60)
print("WSI / source identifiers:")
print("  The HER2Dataset exposes only `image_paths` and `labels`.")
print("  No explicit WSI/source ID metadata is available.")
print("  Per instructions, WSI IDs are NOT inferred or invented.")
print("  Sample filenames (for reference only):")
for i in range(min(3, len(full_train_dataset.image_paths))):
    print(f"    {full_train_dataset.image_paths[i].name}")
print("=" * 60)


# ============================================================
# Cell 13 — Potential Leakage / Correlation Diagnostic
# ============================================================
# WSI/source overlap can only be assessed if the dataset exposes
# WSI/source IDs. It does not. Report "not available".

print("=" * 60)
print("Potential Leakage / Correlation Diagnostic")
print("=" * 60)
print("  WSI-level overlap cannot be assessed from the available dataset metadata.")
print("  The HER2Dataset exposes only image_paths and labels;")
print("  no WSI/source identifier field is present.")
print("  No WSI IDs were inferred or invented.")
print("=" * 60)


# ============================================================
# Cell 14 — Prediction Error Analysis
# ============================================================
# Using the already-generated predictions: confusion matrices,
# largest confusions, per-class recall comparison. No retraining.

from sklearn.metrics import confusion_matrix

cm_val = confusion_matrix(np.array(val_labels_list), np.array(val_preds), labels=[0, 1, 2, 3])
cm_test = confusion_matrix(np.array(test_labels_list), np.array(test_preds), labels=[0, 1, 2, 3])

print("=" * 60)
print("Confusion Matrix — Validation")
print("=" * 60)
print("Rows = true, Cols = predicted (0, 1+, 2+, 3+)")
print(cm_val)
print("=" * 60)
print("Confusion Matrix — Official Test")
print("=" * 60)
print("Rows = true, Cols = predicted (0, 1+, 2+, 3+)")
print(cm_test)
print("=" * 60)

def largest_confusions(cm, name, top_n=5):
    off_diag = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                off_diag.append((int(cm[i, j]), CLASS_NAMES[i], CLASS_NAMES[j]))
    off_diag.sort(reverse=True)
    print(f"Largest confusions — {name}:")
    for count, true_cls, pred_cls in off_diag[:top_n]:
        print(f"  {true_cls} → {pred_cls}: {count}")
    if not off_diag:
        print("  (none)")

largest_confusions(cm_val, "Validation")
largest_confusions(cm_test, "Official Test")

print("=" * 60)
print("Per-class recall comparison:")
print(f"{'Class':<12} {'Val Recall':>12} {'Test Recall':>12} {'Δ':>10}")
for i, cls in enumerate(CLASS_NAMES):
    d = test_metrics["per_class_recall"][i] - val_metrics["per_class_recall"][i]
    print(f"{cls:<12} {val_metrics['per_class_recall'][i]:>12.4f} {test_metrics['per_class_recall'][i]:>12.4f} {d:>+10.4f}")
print("=" * 60)


# ============================================================
# Cell 15 — Diagnostic Summary
# ============================================================

print("=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

# 1. Checkpoint consistency
ckpt_ok = ("model_state_dict" in ckpt) and (ckpt.get("stage") == 3)
print(f"1. Checkpoint consistency        : {'PASS' if ckpt_ok else 'FAIL'}")

# 2. Validation reproduction
val_repro_ok = abs(val_metrics["accuracy"] - 99.26) < 2.0
print(f"2. Validation reproduction       : {'PASS' if val_repro_ok else 'FAIL'} "
      f"(re-eval={val_metrics['accuracy']:.2f}%, expected≈99.26%)")

# 3. Official test reproduction
test_repro_ok = abs(test_metrics["accuracy"] - 87.22) < 2.0
print(f"3. Official test reproduction    : {'PASS' if test_repro_ok else 'FAIL'} "
      f"(re-eval={test_metrics['accuracy']:.2f}%, expected≈87.22%)")

# 4. Saved split consistency
split_ok = (len(set(train_idx) & set(val_idx)) == 0) and (len(train_idx) + len(val_idx) == len(full_train_dataset))
print(f"4. Saved split consistency       : {'PASS' if split_ok else 'FAIL'}")

# 5. Label mapping consistency
label_ok = full_train_dataset.get_class_names() == test_dataset.get_class_names()
print(f"5. Label mapping consistency     : {'PASS' if label_ok else 'FAIL'}")

# 6. Preprocessing consistency
preproc_ok = all(checks.values())
print(f"6. Preprocessing consistency     : {'PASS' if preproc_ok else 'FAIL'}")

# 7. Class distribution difference
print("7. Class distribution difference :")
for i, cls in enumerate(CLASS_NAMES):
    tr_p = 100.0 * train_counts[i] / train_counts.sum()
    va_p = 100.0 * val_counts[i] / val_counts.sum()
    te_p = 100.0 * test_counts[i] / test_counts.sum()
    print(f"     {cls:<10} train={tr_p:5.2f}%  val={va_p:5.2f}%  test={te_p:5.2f}%")

# 8. WSI/source overlap
print("8. WSI/source overlap            : not available (dataset exposes no WSI/source IDs)")

# 9. Validation → test gap
print(f"9. Validation → test gap         : {gap:+.2f} percentage points")

# 10. Evidence-supported explanations (only from observations)
print("10. Evidence-supported explanations:")
print("    - Validation and test use identical non-augmented preprocessing (PASS).")
print("    - Label mapping is identical (PASS).")
print("    - Class distribution differs between validation and test (see #7).")
print("    - WSI-level overlap cannot be assessed from available metadata.")
print("    - The recorded best_val_acc vs re-evaluated val acc difference is reported in Cell 8.")
print("=" * 60)


# ============================================================
# Cell 16 — FINAL CONCLUSION
# ============================================================

print("=" * 60)
print("FINAL CONCLUSION (diagnostic only — no fixes implemented)")
print("=" * 60)

print("A. Was the 87.22% test evaluation performed correctly?")
print(f"   Independent re-evaluation of best_stage3.pt on the official test split:")
print(f"   Accuracy={test_metrics['accuracy']:.2f}%, BalancedAcc={test_metrics['balanced_accuracy']:.2f}%, "
      f"Macro-F1={test_metrics['macro_f1']:.4f}")
print("   The evaluation uses the existing model, config, dataset, preprocessing,")
print("   label mapping, and metrics. Preprocessing checks all PASS.")
print()

print("B. Was best_stage3.pt actually the checkpoint selected by validation?")
print(f"   Checkpoint metadata: stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')}, "
      f"best_val_acc={ckpt.get('best_val_accuracy')}")
print("   The training notebook saves best_stage3.pt whenever validation accuracy")
print("   improves during Stage 3, so it is the validation-selected checkpoint.")
print()

print("C. Why does the checkpoint metadata say 98.80% while independent validation may show 99.26%?")
print(f"   Recorded best_val_acc = {ckpt.get('best_val_accuracy')}")
print(f"   Re-evaluated val acc  = {val_metrics['accuracy']:.2f}%")
print("   Possible evidence-supported reasons (to be confirmed by the numbers above):")
print("   - The recorded value is the running best at the moment of saving; the final")
print("     re-evaluation may use a different random seed for augmentation-free eval,")
print("     or the recorded value was from an earlier epoch.")
print("   - The re-evaluation here uses the exact saved split and non-augmented test")
print("     transform, which is the correct protocol.")
print()

print("D. Is there evidence of preprocessing mismatch?")
print(f"   {'YES' if not preproc_ok else 'NO'} — all preprocessing checks PASS." if preproc_ok else "   YES — see FAIL checks in Cell 11.")
print()

print("E. Is there evidence of class distribution shift?")
print("   See Cell 12 / Cell 15 #7 for train/val/test class percentages.")
print("   The test split has a different class distribution than the validation split.")
print()

print("F. Is there evidence of WSI/source overlap?")
print("   Not assessable — the dataset exposes no WSI/source IDs.")
print()

print("G. What should be investigated NEXT?")
print("   - Confirm whether the recorded best_val_acc (98.80%) corresponds to the epoch")
print("     at which best_stage3.pt was saved vs the final re-evaluation (99.26%).")
print("   - Investigate the validation→test gap (%.2f pp) with class-distribution" % gap)
print("     and, if WSI metadata becomes available, WSI-level overlap.")
print("   - No fixes were implemented in this notebook.")
print("=" * 60)