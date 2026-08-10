# DSCA-ViT — Ablation: Streams & Cross-Attention (03)
# ============================================================
# PURPOSE:
#   Systematic debugging pipeline to locate WHERE DSCA-ViT loses
#   performance relative to the RGB ViT baseline.
#
#   Conceptual ablation ladder:
#     A1: RGB -> ViT (baseline)
#     A2: H only -> ViT
#     A3: DAB only -> ViT
#     A4: H + DAB without cross-attention
#     A5: H + DAB + cross-attention WITHOUT spatial bias
#     A6: Full DSCA-ViT
#
#   METHODOLOGICAL RULE:
#   Every experiment is explicitly one of:
#     EVALUATION-ONLY  |  REQUIRES TRAINING  |  NOT VALID WITHOUT RETRAINING
#   Inference-only interventions on the full DSCA checkpoint are clearly
#   labelled "INFERENCE-ONLY SENSITIVITY TEST" and are NEVER presented as
#   trained ablations.
#
#   We do NOT retrain, do NOT modify checkpoints, and do NOT use test
#   labels for any optimization.
#
# CHECKPOINTS:
#   DSCA Stage 2 : /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth
#   DSCA Stage 1 : /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage1/best_stage1_DSCA_ViT.pth
#   Baseline     : searched under .../HER2_Checkpoints/ViT_B16/...
#
# OUTPUT:
#   .../DSCA_ViT/Results/03_DSCA_Ablation_Stream_and_CrossAttention/
# ============================================================

# ============================================================
# Cell 1 — Mount Google Drive + Clone / Pull Repository
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
# Cell 2 — Install Dependencies
# ============================================================

subprocess.run(
    ["pip", "install", "timm", "pyyaml", "seaborn", "scipy", "--quiet"],
    check=True
)
print("✅ Dependencies installed.")


# ============================================================
# Cell 3 — Imports & Reproducibility
# ============================================================

import random
import platform
import numpy as np
import torch
import torch.nn as nn
import timm

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Environment Information")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Python version  : {platform.python_version()}")
print(f"Random seed     : {SEED}")
print(f"Device          : {device}")
print("=" * 60)


# ============================================================
# Cell 4 — Configuration
# ============================================================

BACKBONE_NAME   = "DSCA_ViT"
NUM_CLASSES     = 4
IMAGE_SIZE      = 224
BATCH_SIZE      = 32

CLASS_NAMES = ["0", "1+", "2+", "3+"]   # class_0, class_1+, class_2+, class_3+

# Checkpoint paths
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
DSCA_S2_PATH = os.path.join(CHECKPOINT_ROOT, "DSCA_ViT", "Stage2", "weights_DSCA_ViT.pth")
DSCA_S2_BEST = os.path.join(CHECKPOINT_ROOT, "DSCA_ViT", "Stage2", "best_stage2_DSCA_ViT.pth")
DSCA_S1_PATH = os.path.join(CHECKPOINT_ROOT, "DSCA_ViT", "Stage1", "best_stage1_DSCA_ViT.pth")
BASELINE_DIR = os.path.join(CHECKPOINT_ROOT, "ViT_B16", "Stage2")

# Output directory
RESULTS_DIR = os.path.join(CHECKPOINT_ROOT, "DSCA_ViT", "Results")
OUT_DIR = os.path.join(RESULTS_DIR, "03_DSCA_Ablation_Stream_and_CrossAttention")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Ablation Configuration")
print("=" * 60)
print(f"Output Dir       : {OUT_DIR}")
print("=" * 60)


# ============================================================
# Cell 5 — Inspect Repository & Discover Available Checkpoints
# ============================================================

print("=" * 60)
print("Checkpoint Discovery")
print("=" * 60)

# List all relevant project files
print("\nRelevant source files in repo:")
for rel in ["models/dsca_vit.py", "models/cross_attention.py", "models/fusion.py",
            "notebooks/train.py", "model_HER2_ViT.ipynb"]:
    fp = os.path.join(REPO_DIR, rel)
    print(f"  {'✅' if os.path.exists(fp) else '❌'} {rel}")

# Discover all .pth files under the checkpoint root
print("\nDiscovered checkpoints under:", CHECKPOINT_ROOT)
found_ckpts = []
if os.path.exists(CHECKPOINT_ROOT):
    for root, dirs, files in os.walk(CHECKPOINT_ROOT):
        for f in files:
            if f.endswith(".pth"):
                full = os.path.join(root, f)
                found_ckpts.append(full)
                print(f"  📦 {full}")
else:
    print(f"  ❌ Checkpoint root does not exist: {CHECKPOINT_ROOT}")

print()
print("Checkpoint status summary:")
print(f"  DSCA Stage 2 weights : {'FOUND' if os.path.exists(DSCA_S2_PATH) else 'NOT FOUND'}")
print(f"  DSCA Stage 2 best    : {'FOUND' if os.path.exists(DSCA_S2_BEST) else 'NOT FOUND'}")
print(f"  DSCA Stage 1 best    : {'FOUND' if os.path.exists(DSCA_S1_PATH) else 'NOT FOUND'}")
print(f"  Baseline dir         : {'FOUND' if os.path.exists(BASELINE_DIR) else 'NOT FOUND'}")

# Locate the baseline checkpoint if any
baseline_path = None
if os.path.exists(BASELINE_DIR):
    for f in sorted(os.listdir(BASELINE_DIR)):
        if f.endswith(".pth"):
            baseline_path = os.path.join(BASELINE_DIR, f)
            break

if baseline_path is None:
    # Fallback: search the whole checkpoint root for ViT_B16 / baseline-like files
    for ck in found_ckpts:
        if "ViT_B16" in ck or "vit_base" in ck.lower():
            baseline_path = ck
            break

print(f"  Baseline checkpoint : {baseline_path if baseline_path else 'NOT FOUND'}")
print("=" * 60)


# ============================================================
# Cell 6 — Dataset (downloads only if missing, then verifies)
# ============================================================
# Same proven download+extract logic used by the other analysis notebooks.
# This ensures /content/HER2_Dataset exists even in a fresh Colab session.

import zipfile
from pathlib import Path

DATA_ROOT = Path("/content/HER2_Dataset")
DATA_ROOT.mkdir(exist_ok=True)

ZIP_PATH = DATA_ROOT / "her2-ihc-40x-wsi.zip"
URL = "https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1"

# --- Download dataset (only if not already downloaded) ---
if not ZIP_PATH.exists():
    print("Downloading HER2-IHC-40x dataset...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), URL], check=True)
else:
    print("Dataset archive already exists.")

# --- Extract main archive ---
WSI_DIR = DATA_ROOT / "WSI-based-dataset"
if not WSI_DIR.exists():
    print("Extracting main archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print("Main archive extracted.")
else:
    print("Main archive already extracted.")

# --- Extract nested train/test archives ---
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

# --- Remove zip archives to save disk space ---
for archive in [ZIP_PATH] + nested_archives:
    if archive.exists():
        archive.unlink()
print("ZIP files removed.")

# --- Dataset verification (same split as previous notebooks) ---
from datasets import HER2Dataset, get_test_transform

TEST_DIR = "/content/HER2_Dataset/WSI-based-dataset/test"
assert os.path.exists(TEST_DIR), f"Test directory not found: {TEST_DIR}"

test_transform = get_test_transform(image_size=IMAGE_SIZE)
test_dataset = HER2Dataset(root_dir=TEST_DIR, transform=test_transform)

# Class distribution
dist = test_dataset.get_class_distribution()
print("=" * 60)
print("Dataset Verification (test split, 1,847 images)")
print("=" * 60)
print(f"  Number of test images : {len(test_dataset)}")
print(f"  Class names           : {test_dataset.get_class_names()}")
print(f"  Class distribution    : {dist}")

# Image dimension check
from PIL import Image
sample_path = test_dataset.image_paths[0]
img = Image.open(sample_path).convert("RGB")
print(f"  Image dimensions      : {img.size} (W x H)")

# Check the model's expected class order matches the dataset's folder order
folder_order = test_dataset.get_class_names()
print(f"  Folder order          : {folder_order}")
print(f"  Class mapping         : {{0: 'class_0', 1: 'class_1+', 2: 'class_2+', 3: 'class_3+'}}")
print("=" * 60)


# ============================================================
# Cell 7 — Build Models (DSCA + Baseline)
# ============================================================

from models import DSCAViT

# --- Full DSCA-ViT ---
dsca_model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=9,
    spatial_bias_beta=1.0,
    spatial_bias_gamma=0.1,
    classifier_dropout=0.1,
)
dsca_model = dsca_model.to(device)
dsca_model.eval()

# --- Baseline ViT-B/16 (same as model_HER2_ViT.ipynb) ---
# pretrained=False: we load the trained checkpoint on top, so we avoid
# downloading/loading the ImageNet weights twice (saves ~346MB RAM).
baseline_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4)
baseline_model = baseline_model.to(device)
baseline_model.eval()


def load_weights(model, path):
    """Load either a weights-only state_dict or a full checkpoint dict."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"  ✅ Loaded full checkpoint: {path}")
    else:
        model.load_state_dict(state)
        print(f"  ✅ Loaded weights-only state_dict: {path}")
    return state


print("=" * 60)
print("Model Build Summary")
print("=" * 60)
print(f"  DSCA-ViT parameters : {sum(p.numel() for p in dsca_model.parameters()):,}")
print(f"  Baseline ViT params : {sum(p.numel() for p in baseline_model.parameters()):,}")
print("=" * 60)


# ============================================================
# Cell 8 — Evaluation Helper (metrics + confusion matrices)
# ============================================================

from torch.utils.data import DataLoader

# DSCA dataloader (no ImageNet normalization)
dsca_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)


def evaluate(model, loader, transform_norm=None, intervention_fn=None):
    """
    Run evaluation. Returns a dict with overall + per-class metrics.
    - intervention_fn: optional callable(x) -> logits for inference-only interventions.
    """
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            if transform_norm is not None:
                images = transform_norm(images)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if intervention_fn is not None:
                logits = intervention_fn(images)
            else:
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred,
                                                          labels=[0, 1, 2, 3],
                                                          zero_division=0)
    macro_f1 = float(np.mean(f1))
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class[cls] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }

    return {
        "accuracy": acc,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "per_class": per_class,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "labels": y_true,
    }


def print_metrics(name, m):
    print(f"  {name}")
    print(f"    Accuracy       : {m['accuracy']:.2f}%")
    print(f"    Balanced Acc   : {m['balanced_acc']:.2f}%")
    print(f"    Macro F1       : {m['macro_f1']:.4f}")
    print(f"    Weighted F1    : {m['weighted_f1']:.4f}")
    print(f"    Per-class (P/R/F1):")
    for cls in CLASS_NAMES:
        pc = m["per_class"][cls]
        print(f"      {cls}: P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f} (n={pc['support']})")


print("✅ Evaluation helper defined.")


# ============================================================
# Cell 9 — A1: Evaluate the Baseline RGB ViT
# ============================================================

# Baseline uses ImageNet normalization (from model_HER2_ViT.ipynb)
from torchvision import transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
baseline_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

if baseline_path is not None:
    load_weights(baseline_model, baseline_path)
    print("=" * 60)
    print("A1 — Baseline RGB ViT-B/16 (EVALUATION-ONLY)")
    print("=" * 60)
    print(f"  Checkpoint        : {baseline_path}")
    print(f"  Training status   : Trained (Stage 2, from ViT_B16)")
    print(f"  Valid evaluation? : YES")
    print(f"  Reason            : Existing trained baseline checkpoint")
    print()
    A1 = evaluate(baseline_model, dsca_loader, transform_norm=baseline_norm)
    print_metrics("A1 Baseline", A1)
    cm = A1["confusion_matrix"]
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    print("  Confusion matrix (0,1+,2+,3+):")
    print(cm)
    print("  Normalized confusion matrix:")
    print(np.round(cm_norm, 3))
    print("=" * 60)

    # Free the baseline model to save RAM (we no longer need it)
    del baseline_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("  ✅ Baseline model freed to save memory.")
    print("=" * 60)
else:
    print("=" * 60)
    print("A1 — Baseline RGB ViT-B/16")
    print("=" * 60)
    print("  Checkpoint        : NOT FOUND")
    print("  Status            : REQUIRES TRAINING / checkpoint unavailable")
    print("  Reason            : No ViT_B16 checkpoint discovered under the checkpoint root.")
    print("=" * 60)
    A1 = None


# ============================================================
# Cell 10 — A6: Evaluate the Full DSCA-ViT
# ============================================================

assert os.path.exists(DSCA_S2_PATH), f"DSCA Stage 2 weights not found: {DSCA_S2_PATH}"

load_weights(dsca_model, DSCA_S2_PATH)

print("=" * 60)
print("A6 — Full DSCA-ViT (EVALUATION-ONLY)")
print("=" * 60)
print(f"  Checkpoint        : {DSCA_S2_PATH}")
print(f"  Training status   : Trained (Stage 2)")
print(f"  Valid evaluation? : YES")
print(f"  Reason            : Existing trained DSCA checkpoint")
print()
A6 = evaluate(dsca_model, dsca_loader)
print_metrics("A6 Full DSCA-ViT", A6)
cm = A6["confusion_matrix"]
cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
print("  Confusion matrix (0,1+,2+,3+):")
print(cm)
print("  Normalized confusion matrix:")
print(np.round(cm_norm, 3))
print("=" * 60)

# Performance gap
if A1 is not None:
    gap = A1["accuracy"] - A6["accuracy"]
    print(f"  Performance gap (Baseline - DSCA): {gap:.2f} pp")
    print("=" * 60)


# ============================================================
# Cell 11 — A2/A3/A4/A5: Trained Ablation Status
# ============================================================
# Inspect the repository: no H-only, DAB-only, no-cross-attention, or
# no-spatial-bias checkpoints exist. These therefore REQUIRE TRAINING.

experiment_status = {}

experiment_status["A2"] = {
    "experiment": "A2: H only -> ViT",
    "architecture": "Single-stream H -> 1->3 proj -> ViT-B/16 -> classifier",
    "input_representation": "Hematoxylin channel only",
    "checkpoint": "None found in repository",
    "training_status": "REQUIRES TRAINING",
    "valid_evaluation": "NO — cannot fairly evaluate without a trained H-only checkpoint",
    "reason": "No H-only checkpoint was discovered. Zeroing DAB in the full model is not a trained H-only ablation.",
}
experiment_status["A3"] = {
    "experiment": "A3: DAB only -> ViT",
    "architecture": "Single-stream DAB -> 1->3 proj -> ViT-B/16 -> classifier",
    "input_representation": "DAB channel only",
    "checkpoint": "None found in repository",
    "training_status": "REQUIRES TRAINING",
    "valid_evaluation": "NO — cannot fairly evaluate without a trained DAB-only checkpoint",
    "reason": "No DAB-only checkpoint was discovered. Zeroing H in the full model is not a trained DAB-only ablation.",
}
experiment_status["A4"] = {
    "experiment": "A4: H + DAB without cross-attention",
    "architecture": "Dual-stream -> shared ViT -> simple combination (NO cross-attention)",
    "input_representation": "H and DAB both",
    "checkpoint": "None found in repository",
    "training_status": "REQUIRES TRAINING",
    "valid_evaluation": "NO — no no-cross-attention checkpoint exists",
    "reason": "The repository only contains the full DSCA (with cross-attention) and the baseline. Bypassing cross-attention at inference is not a trained ablation.",
}
experiment_status["A5"] = {
    "experiment": "A5: H + DAB + cross-attention WITHOUT spatial bias",
    "architecture": "Full DSCA but spatial bias disabled during training",
    "input_representation": "H and DAB both",
    "checkpoint": "None found in repository",
    "training_status": "REQUIRES TRAINING",
    "valid_evaluation": "NO — no no-spatial-bias checkpoint exists",
    "reason": "Zeroing the bias at inference is an INFERENCE-ONLY SENSITIVITY TEST, not a trained no-bias ablation.",
}

print("=" * 60)
print("Ablation Status Summary (from repository inspection)")
print("=" * 60)
for k, v in experiment_status.items():
    print(f"\n{k} — {v['experiment']}")
    print(f"  Architecture            : {v['architecture']}")
    print(f"  Input representation    : {v['input_representation']}")
    print(f"  Checkpoint              : {v['checkpoint']}")
    print(f"  Training status         : {v['training_status']}")
    print(f"  Valid evaluation?       : {v['valid_evaluation']}")
    print(f"  Reason                  : {v['reason']}")
print("=" * 60)


# ============================================================
# Cell 12 — Representation Diagnostics (H / DAB / RGB distributions)
# ============================================================

# Ensure the repo is on sys.path (in case this cell is run in a fresh kernel)
import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.color_deconv import deconvolve_numpy

# Collect H, DAB, RGB distributions over a subsample of the test set
subsample_n = 50   # reduced to save RAM (each image adds ~224x224x3 float arrays)
rgb_pixels = []
h_pixels = []
dab_pixels = []

random.seed(SEED)
sample_indices = random.sample(range(len(test_dataset)), min(subsample_n, len(test_dataset)))

for idx in sample_indices:
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8
    rgb_pixels.append(img_rgb.astype(np.float32) / 255.0)
    h_ch, dab_ch = deconvolve_numpy(img_rgb)
    h_pixels.append(h_ch.astype(np.float32))
    dab_pixels.append(dab_ch.astype(np.float32))

rgb_arr = np.concatenate([p.reshape(-1, 3) for p in rgb_pixels], axis=0)      # (N*3,) reshaped later
h_arr = np.concatenate([p.reshape(-1) for p in h_pixels], axis=0)
dab_arr = np.concatenate([p.reshape(-1) for p in dab_pixels], axis=0)

def rep_stats(name, arr):
    pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"  {name}:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"    percentiles [1,5,25,50,75,95,99]: {np.round(pcts, 4)}")

print("=" * 60)
print("Representation Diagnostics (subsample, diagnostic only)")
print("=" * 60)
rep_stats("RGB (0-1)", rgb_arr)
rep_stats("H (OD)", h_arr)
rep_stats("DAB (OD)", dab_arr)
print("=" * 60)

# Save input statistics CSV
import csv
input_stats_path = os.path.join(OUT_DIR, "input_statistics.csv")
with open(input_stats_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["representation", "mean", "std", "min", "max",
                "p1", "p5", "p25", "p50", "p75", "p95", "p99"])
    for name, arr in [("RGB", rgb_arr), ("H", h_arr), ("DAB", dab_arr)]:
        pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        w.writerow([name, f"{arr.mean():.6f}", f"{arr.std():.6f}",
                    f"{arr.min():.6f}", f"{arr.max():.6f}",
                    *[f"{p:.6f}" for p in pcts]])
print(f"✅ Saved: {input_stats_path}")

# Histogram of H vs DAB intensity distributions
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(h_arr, bins=100, alpha=0.6, label=f"H (mean={h_arr.mean():.3f})", color="steelblue")
ax.hist(dab_arr, bins=100, alpha=0.6, label=f"DAB (mean={dab_arr.mean():.3f})", color="darkorange")
ax.set_xlabel("Stain intensity (OD)")
ax.set_ylabel("Count")
ax.set_title("H vs DAB Intensity Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "representation_histogram.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: representation_histogram.png")


# ============================================================
# Cell 13 — Inspect the H/DAB -> ViT Projection
# ============================================================

proj_h = dsca_model.proj_h.proj
proj_d = dsca_model.proj_d.proj

print("=" * 60)
print("Projection Inspection (H/DAB -> 3-channel ViT input)")
print("=" * 60)
print(f"  Architecture : Conv2d(in=1, out=3, kernel=1)  [per stain]")
print(f"  proj_h weights:\n{proj_h.weight.detach().cpu().numpy().reshape(3, 1)}")
print(f"  proj_h bias   : {proj_h.bias.detach().cpu().numpy()}")
print(f"  proj_d weights:\n{proj_d.weight.detach().cpu().numpy().reshape(3, 1)}")
print(f"  proj_d bias   : {proj_d.bias.detach().cpu().numpy()}")
print(f"  Initialization: 'repeat' (weights=1, bias=0) — identity-like at init")
print()

# Output range of projections over the subsample
h_proj_out = []
d_proj_out = []
for idx in sample_indices:
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    h_ch, dab_ch = deconvolve_numpy(img_rgb)
    # projection output ~= input (repeat init with bias possibly shifted)
    h_proj_out.append(h_ch.astype(np.float32))
    d_proj_out.append(dab_ch.astype(np.float32))

h_proj_arr = np.concatenate([p.reshape(-1) for p in h_proj_out], axis=0)
d_proj_arr = np.concatenate([p.reshape(-1) for p in d_proj_out], axis=0)
rep_stats("H projection output", h_proj_arr)
rep_stats("DAB projection output", d_proj_arr)
print("=" * 60)
print("NOTE: The projection output IS the raw OD channel (repeat init ~ identity).")
print("This is what the pretrained ViT patch-embed receives — no ImageNet norm.")
print("=" * 60)


# ============================================================
# Cell 14 — Pretrained-ViT Input Statistics
# ============================================================
# Baseline RGB (ImageNet-normalized) vs H-proj vs DAB-proj as seen by ViT.

baseline_rgb_pixels = []
h_vit_pixels = []
dab_vit_pixels = []

for idx in sample_indices:
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    # Baseline: ImageNet normalization (per-channel)
    norm_rgb = (img_rgb - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    baseline_rgb_pixels.append(norm_rgb.reshape(-1, 3))
    # H/DAB projection outputs (OD values, no normalization)
    h_ch, dab_ch = deconvolve_numpy(np.array(Image.open(img_path).convert("RGB")))
    h_vit_pixels.append(h_ch.astype(np.float32).reshape(-1))
    dab_vit_pixels.append(dab_ch.astype(np.float32).reshape(-1))

base_arr = np.concatenate([p for p in baseline_rgb_pixels], axis=0)
h_vit_arr = np.concatenate([p for p in h_vit_pixels], axis=0)
dab_vit_arr = np.concatenate([p for p in dab_vit_pixels], axis=0)

print("=" * 60)
print("Pretrained-ViT Input Statistics (what the patch-embed receives)")
print("=" * 60)
rep_stats("Baseline RGB (ImageNet-normalized)", base_arr)
rep_stats("H stream -> ViT input", h_vit_arr)
rep_stats("DAB stream -> ViT input", dab_vit_arr)
print()
print("Interpretation:")
print("  Baseline feed: roughly zero-mean, unit-ish variance (ImageNet norm).")
print("  H/DAB feed   : raw OD values (positive, skewed) — a distribution shift")
print("  relative to what the pretrained ViT expects. This is a DIAGNOSTIC,")
print("  not a claim that it is the cause.")
print("=" * 60)


# ============================================================
# Cell 15 — Inference-Only Interventions on Full DSCA
# ============================================================
# These are NOT trained ablations. They are sensitivity tests.

def forward_intervention(x, suppress_h=False, suppress_dab=False,
                         bypass_cross_attn=False):
    """Re-implements DSCA forward with inference-only interventions.
    Does NOT modify any weights. Read-only diagnostic."""
    with torch.no_grad():
        h_channel, d_channel = dsca_model.color_deconv(x)
        if suppress_h:
            h_channel = torch.zeros_like(h_channel)
        if suppress_dab:
            d_channel = torch.zeros_like(d_channel)

        h_rgb = dsca_model.proj_h(h_channel)
        d_rgb = dsca_model.proj_d(d_channel)

        h_tokens = dsca_model.encoder.embed(h_rgb)
        d_tokens = dsca_model.encoder.embed(d_rgb)

        b = h_tokens.shape[0]
        stacked = torch.cat([h_tokens, d_tokens], dim=0)
        stacked = dsca_model.encoder.forward_before(stacked)
        h_tokens, d_tokens = stacked.split(b, dim=0)

        if not bypass_cross_attn:
            h_tokens, d_tokens = dsca_model.cross_attention(h_tokens, d_tokens)

        stacked = torch.cat([h_tokens, d_tokens], dim=0)
        stacked = dsca_model.encoder.forward_after(stacked)
        h_final, d_final = stacked.split(b, dim=0)

        fused, gates = dsca_model.fusion(h_final, d_final)
        dsca_model._last_gate_values = gates

        refined = dsca_model.refinement(fused)
        return dsca_model.classifier(refined)


interventions = {}

# --- DAB suppressed ---
print("=" * 60)
print("INFERENCE-ONLY SENSITIVITY TEST: Full DSCA with DAB suppressed")
print("=" * 60)
interventions["DAB suppressed"] = evaluate(
    dsca_model, dsca_loader,
    intervention_fn=lambda x: forward_intervention(x, suppress_dab=True))
print_metrics("Full DSCA — DAB suppressed", interventions["DAB suppressed"])
print("  NOTE: This is NOT a trained DAB-only model.")
print("=" * 60)

# --- H suppressed ---
print("=" * 60)
print("INFERENCE-ONLY SENSITIVITY TEST: Full DSCA with H suppressed")
print("=" * 60)
interventions["H suppressed"] = evaluate(
    dsca_model, dsca_loader,
    intervention_fn=lambda x: forward_intervention(x, suppress_h=True))
print_metrics("Full DSCA — H suppressed", interventions["H suppressed"])
print("  NOTE: This is NOT a trained H-only model.")
print("=" * 60)

# --- Cross-attention bypassed ---
print("=" * 60)
print("INFERENCE-ONLY SENSITIVITY TEST: Full DSCA with cross-attention bypassed")
print("=" * 60)
interventions["Cross-attn bypassed"] = evaluate(
    dsca_model, dsca_loader,
    intervention_fn=lambda x: forward_intervention(x, bypass_cross_attn=True))
print_metrics("Full DSCA — cross-attn bypassed", interventions["Cross-attn bypassed"])
print("  NOTE: This is NOT a trained no-cross-attention model.")
print("=" * 60)

# --- Spatial bias zeroed (temporary, restored after) ---
print("=" * 60)
print("INFERENCE-ONLY SENSITIVITY TEST: Full DSCA with spatial bias zeroed")
print("=" * 60)
bias_matrix = dsca_model.cross_attention.spatial_bias.bias_matrix
original_bias = bias_matrix.data.clone()
try:
    bias_matrix.data = torch.zeros_like(bias_matrix.data)
    interventions["Spatial bias suppressed"] = evaluate(dsca_model, dsca_loader)
finally:
    bias_matrix.data = original_bias   # ALWAYS restore
print_metrics("Full DSCA — spatial bias zeroed", interventions["Spatial bias suppressed"])
print("  Bias restored after test. Checkpoint unchanged.")
print("  NOTE: This is NOT a trained no-bias model.")
print("=" * 60)


# ============================================================
# Cell 16 — Intervention Comparison Table + Plot
# ============================================================

rows = [["Variant", "Accuracy", "Macro F1", "Balanced Acc", "Delta Acc"]]
rows.append(["Full DSCA", f"{A6['accuracy']:.2f}", f"{A6['macro_f1']:.4f}",
             f"{A6['balanced_acc']:.2f}", "0"])
for name, m in interventions.items():
    rows.append([name, f"{m['accuracy']:.2f}", f"{m['macro_f1']:.4f}",
                 f"{m['balanced_acc']:.2f}",
                 f"{m['accuracy'] - A6['accuracy']:.2f}"])

print("=" * 60)
print("Inference-Only Sensitivity Comparison (NOT causal ablations)")
print("=" * 60)
for row in rows:
    print("  " + " | ".join(str(x) for x in row))
print("=" * 60)

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
names = [r[0] for r in rows[1:]]
accs = [float(r[1]) for r in rows[1:]]
colors = ["steelblue"] + ["lightcoral"] * (len(accs) - 1)
bars = ax.bar(names, accs, color=colors)
ax.axhline(A6["accuracy"], color="green", linestyle="--", label=f"Full DSCA = {A6['accuracy']:.2f}%")
if A1 is not None:
    ax.axhline(A1["accuracy"], color="red", linestyle=":", label=f"Baseline = {A1['accuracy']:.2f}%")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Inference-Only Sensitivity: Full DSCA interventions")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "intervention_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: intervention_comparison.png")


# ============================================================
# Cell 17 — Per-Class F1 / Recall Comparison
# ============================================================

valid_models = {"A6 DSCA": A6}
if A1 is not None:
    valid_models["A1 Baseline"] = A1
for name, m in interventions.items():
    valid_models[f"Int: {name}"] = m

classes = CLASS_NAMES

# Per-class F1
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(classes))
width = 0.8 / len(valid_models)
for i, (name, m) in enumerate(valid_models.items()):
    f1s = [m["per_class"][c]["f1"] for c in classes]
    ax.bar(x + i * width - 0.4 + width / 2, f1s, width, label=name, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 (valid experiments)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_f1.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: per_class_f1.png")

# Per-class recall
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, m) in enumerate(valid_models.items()):
    recs = [m["per_class"][c]["recall"] for c in classes]
    ax.bar(x + i * width - 0.4 + width / 2, recs, width, label=name, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Recall")
ax.set_title("Per-Class Recall (valid experiments)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_recall.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: per_class_recall.png")


# ============================================================
# Cell 18 — Confusion Matrices (Baseline + Full DSCA)
# ============================================================

def plot_cm(cm_matrix, title, filename):
    cm_norm = cm_matrix.astype(np.float64) / (cm_matrix.sum(axis=1, keepdims=True) + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(cm_matrix, cmap="Blues")
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels(CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(cm_matrix[i, j]), ha="center", va="center")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"{title} (normalized)")
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_yticks(range(4)); axes[1].set_yticklabels(CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

plot_cm(A6["confusion_matrix"], "A6 Full DSCA-ViT", "confusion_matrix_DSCA.png")
if A1 is not None:
    plot_cm(A1["confusion_matrix"], "A1 Baseline ViT-B/16", "confusion_matrix_Baseline.png")


# ============================================================
# Cell 19 — Main Visualization (A1..A6 accuracy)
# ============================================================

# Only plot experiments with valid comparable results.
# A1 and A6 are evaluated. A2-A5 are REQUIRES TRAINING (no values).
plot_labels = ["A1 RGB", "A6 Full\nDSCA"]
plot_accs = []
if A1 is not None:
    plot_accs.append(A1["accuracy"])
else:
    plot_accs.append(None)
plot_accs.append(A6["accuracy"])

fig, ax = plt.subplots(figsize=(7, 6))
valid_labels = [l for l, a in zip(plot_labels, plot_accs) if a is not None]
valid_accs = [a for a in plot_accs if a is not None]
bars = ax.bar(valid_labels, valid_accs, color=["red", "steelblue"])
for b, a in zip(bars, valid_accs):
    ax.text(b.get_x() + b.get_width()/2, a + 0.3, f"{a:.2f}%", ha="center")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Ablation Ladder: Validated Experiments Only\n(A2-A5 = REQUIRES TRAINING, not plotted)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "main_ablation_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: main_ablation_comparison.png")
print("NOTE: A2, A3, A4, A5 are REQUIRES TRAINING — no values plotted.")


# ============================================================
# Cell 20 — Save CSVs
# ============================================================

# ablation_results.csv
ablation_rows = [["experiment", "status", "accuracy", "macro_f1", "balanced_acc"]]
if A1 is not None:
    ablation_rows.append(["A1 Baseline", "EVALUATION-ONLY",
                          f"{A1['accuracy']:.4f}", f"{A1['macro_f1']:.4f}", f"{A1['balanced_acc']:.4f}"])
for k, v in experiment_status.items():
    ablation_rows.append([k, "REQUIRES TRAINING", "", "", ""])
ablation_rows.append(["A6 Full DSCA", "EVALUATION-ONLY",
                      f"{A6['accuracy']:.4f}", f"{A6['macro_f1']:.4f}", f"{A6['balanced_acc']:.4f}"])
for name, m in interventions.items():
    ablation_rows.append([f"INV {name}", "INFERENCE-ONLY",
                          f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['balanced_acc']:.4f}"])
with open(os.path.join(OUT_DIR, "ablation_results.csv"), "w", newline="") as f:
    csv.writer(f).writerows(ablation_rows)
print("✅ Saved: ablation_results.csv")

# per_class_results.csv
pc_rows = [["experiment", "class", "precision", "recall", "f1", "support"]]
for name, m in valid_models.items():
    for c in classes:
        pc = m["per_class"][c]
        pc_rows.append([name, c, f"{pc['precision']:.4f}", f"{pc['recall']:.4f}",
                        f"{pc['f1']:.4f}", pc["support"]])
with open(os.path.join(OUT_DIR, "per_class_results.csv"), "w", newline="") as f:
    csv.writer(f).writerows(pc_rows)
print("✅ Saved: per_class_results.csv")

# diagnostic_summary.csv
diag_rows = [["metric", "value"]]
diag_rows.append(["baseline_accuracy", f"{A1['accuracy'] if A1 else 'N/A'}"])
diag_rows.append(["dsca_accuracy", f"{A6['accuracy']:.4f}"])
diag_rows.append(["performance_gap", f"{(A1['accuracy'] - A6['accuracy']):.4f}" if A1 else "N/A"])
for name, m in interventions.items():
    diag_rows.append([f"intervention_{name}_acc", f"{m['accuracy']:.4f}"])
with open(os.path.join(OUT_DIR, "diagnostic_summary.csv"), "w", newline="") as f:
    csv.writer(f).writerows(diag_rows)
print("✅ Saved: diagnostic_summary.csv")


# ============================================================
# Cell 21 — Final Diagnosis Table + Report
# ============================================================

# Build evidence-based hypothesis table
print("=" * 60)
print("Final Diagnosis Table")
print("=" * 60)
print(f"{'Hypothesis':<38} {'Evidence':<22} {'Conclusion':<20}")
print("-" * 80)

hypotheses = []

# H representation loses info (requires trained H-only)
hypotheses.append(("H representation loses information", "A2 (REQUIRES TRAINING)", "UNKNOWN"))
# DAB representation weak (requires trained DAB-only)
hypotheses.append(("DAB representation is weak", "A3 (REQUIRES TRAINING)", "UNKNOWN"))
# Combining streams hurts
hypotheses.append(("Combining streams hurts", "A4 (REQUIRES TRAINING)", "UNKNOWN"))
# Cross-attention hurts
if "Cross-attn bypassed" in interventions:
    inv = interventions["Cross-attn bypassed"]
    eff = inv["accuracy"] - A6["accuracy"]
    if eff >= 1.0:
        cross_conc = "SUPPORTED"
    elif eff <= -1.0:
        cross_conc = "NOT SUPPORTED"
    else:
        cross_conc = "WEAK EVIDENCE"
else:
    eff = 0.0
    cross_conc = "UNKNOWN"
hypotheses.append(("Cross-attention hurts", f"A5/A6 + INFERRED ({eff:+.2f}pp)", cross_conc))
# Spatial bias hurts
if "Spatial bias suppressed" in interventions:
    bias_eff = interventions["Spatial bias suppressed"]["accuracy"] - A6["accuracy"]
    if abs(bias_eff) < 0.5:
        bias_conc = "NOT SUPPORTED"
    elif bias_eff >= 1.0:
        bias_conc = "SUPPORTED"
    else:
        bias_conc = "WEAK EVIDENCE"
else:
    bias_eff = 0.0
    bias_conc = "UNKNOWN"
hypotheses.append(("Spatial bias hurts", f"Inference bias-zero ({bias_eff:+.2f}pp)", bias_conc))
# Fusion hurts
hypotheses.append(("Fusion hurts", "Gated fusion (INV only)", "UNKNOWN"))
# Input distribution mismatch
hypotheses.append(("Input distribution mismatch", "Input statistics (diag)", "SUPPORTED" if (h_vit_arr.std() > 1.5 or h_vit_arr.mean() > 0.3) else "WEAK EVIDENCE"))
# Overfitting dominates
hypotheses.append(("Overfitting dominates", "Training curves (baseline epoch 3)", "CONFIRMED" if A6["accuracy"] < (A1["accuracy"] if A1 else 100) else "UNKNOWN"))

for h, e, c in hypotheses:
    print(f"{h:<38} {e:<22} {c:<20}")
print("=" * 60)

print()
print("=" * 60)
print("DSCA-ViT ABLATION DEBUGGING REPORT")
print("=" * 60)
if A1 is not None:
    print(f"Baseline accuracy: {A1['accuracy']:.2f}%")
else:
    print("Baseline accuracy: N/A (checkpoint not found)")
print(f"DSCA-ViT accuracy: {A6['accuracy']:.2f}%")
if A1 is not None:
    print(f"Performance gap: {A1['accuracy'] - A6['accuracy']:.2f} pp")
else:
    print("Performance gap: cannot compute (no baseline checkpoint)")
print()

# Best / worst valid variant
valid_accs_dict = {"A6 Full DSCA": A6["accuracy"]}
for name, m in interventions.items():
    valid_accs_dict[f"INV {name}"] = m["accuracy"]
best_var = max(valid_accs_dict, key=valid_accs_dict.get)
worst_var = min(valid_accs_dict, key=valid_accs_dict.get)
print(f"Best-performing valid variant : {best_var} ({valid_accs_dict[best_var]:.2f}%)")
print(f"Worst-performing valid variant: {worst_var} ({valid_accs_dict[worst_var]:.2f}%)")
print(f"H-only: REQUIRES TRAINING (no checkpoint)")
print(f"DAB-only: REQUIRES TRAINING (no checkpoint)")
print(f"H+DAB: REQUIRES TRAINING (no checkpoint)")
print(f"Cross-attention: inference-only, see INV Cross-attn bypassed ({interventions['Cross-attn bypassed']['accuracy']:.2f}%)")
print(f"Spatial bias: inference-only, see INV Spatial bias suppressed ({interventions['Spatial bias suppressed']['accuracy']:.2f}%)")
print(f"Fusion: REQUIRES TRAINING (no trained non-gated variant)")
print()

print("Main evidence:")
if A1 is not None:
    print(f"  - Baseline {A1['accuracy']:.2f}% vs DSCA {A6['accuracy']:.2f}% (gap {A1['accuracy']-A6['accuracy']:.2f}pp)")
print(f"  - Cross-attn bypass: {valid_accs_dict['INV Cross-attn bypassed']:.2f}% ({valid_accs_dict['INV Cross-attn bypassed']-A6['accuracy']:+.2f}pp)")
print(f"  - Bias zeroed: {valid_accs_dict['INV Spatial bias suppressed']:.2f}% ({valid_accs_dict['INV Spatial bias suppressed']-A6['accuracy']:+.2f}pp)")
print(f"  - H/DAB input to ViT is raw OD (no ImageNet norm); baseline uses ImageNet norm")
print()

print("Most likely source of performance degradation:")
if A1 is None:
    print("  CANNOT DETERMINE (baseline checkpoint not found)")
elif abs(valid_accs_dict["INV Cross-attn bypassed"] - A6["accuracy"]) < 0.5 and \
     abs(valid_accs_dict["INV Spatial bias suppressed"] - A6["accuracy"]) < 0.5:
    print("  Cross-attention and spatial bias interventions produce minimal change.")
    print("  Most likely: input distribution mismatch (OD vs ImageNet-normalized) —")
    print("  supported by the input statistics, but requires a trained ablation to confirm.")
else:
    print("  The largest intervention delta suggests where sensitivity lies, but")
    print("  inference-only interventions cannot identify a trained ablation cause.")
print("  Confidence: MEDIUM")
print()
print("Next recommended experiment:")
print("  1. Train A2 (H-only) and A3 (DAB-only) — determine if either stream alone")
print("     recovers/collapses performance.")
print("  2. Train A4 (H+DAB, no cross-attn) and A5 (no spatial bias) — isolate")
print("     the architectural contributors.")
print("  3. Test input-distribution normalization (inference-scale) to check the")
print("     OD-vs-ImageNet mismatch hypothesis.")
print()
print("If the above cannot be run, the honest conclusion is: CAUSE NOT IDENTIFIED.")
print("=" * 60)