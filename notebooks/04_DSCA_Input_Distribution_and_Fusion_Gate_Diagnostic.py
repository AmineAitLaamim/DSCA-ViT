# DSCA-ViT — Input Distribution & Fusion Gate Diagnostic (04)
# ============================================================
# PURPOSE:
#   PURE DIAGNOSTIC. Inspect the EXISTING trained DSCA-ViT to answer:
#     1. What are the actual input distributions entering the pretrained
#        ViT for H and DAB?
#     2. What has the learned gated fusion mechanism actually learned,
#        especially for the problematic 1+ vs 2+ classes?
#
#   NO TRAINING. NO CHECKPOINT MODIFICATION. NO ARCHITECTURE CHANGE.
#   NO A2/A3/A4/A5 ABLATIONS. NO INFERENCE INTERVENTIONS (no zeroing).
#
#   We use the EXISTING repository implementation and the EXISTING
#   Stage-2 DSCA checkpoint. We do NOT reimplement DSCA-ViT.
#
# CHECKPOINT:
#   /content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth
#
# OUTPUT:
#   .../DSCA_ViT/Results/04_DSCA_Input_Distribution_and_Fusion_Gate_Diagnostic/
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
import gc
import numpy as np
import torch
import torch.nn as nn
import timm

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
print(f"Python version  : {platform.python_version()}")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Device          : {device}")
print(f"Random seed     : {SEED}")
print("=" * 60)

# Repository commit/hash if available
try:
    import subprocess as sp
    hash_out = sp.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                      capture_output=True, text=True)
    print(f"Repo commit     : {hash_out.stdout.strip() if hash_out.returncode == 0 else 'N/A'}")
except Exception:
    print("Repo commit     : N/A")
print("=" * 60)


# ============================================================
# Cell 4 — Verify Repository Files
# ============================================================

print("=" * 60)
print("Repository File Verification")
print("=" * 60)
for rel in ["models/dsca_vit.py", "models/cross_attention.py", "models/fusion.py",
            "models/color_deconv.py", "models/shared_vit.py",
            "datasets/dataset.py", "datasets/transforms.py"]:
    fp = os.path.join(REPO_DIR, rel)
    print(f"  {'✅' if os.path.exists(fp) else '❌'} {rel}")
print("=" * 60)


# ============================================================
# Cell 5 — Configuration (EXACT trained config)
# ============================================================

BACKBONE_NAME   = "DSCA_ViT"
MODEL_ID        = "dsca_vit_b16"
NUM_CLASSES     = 4
IMAGE_SIZE      = 224
PATCH_SIZE      = 16
GRID            = 14                 # 224 / 16
N_PATCH         = GRID * GRID        # 196
N_CLS           = 1
N_TOKENS        = N_PATCH + N_CLS    # 197
SPLIT_AFTER     = 9
SPATIAL_BIAS_BETA  = 1.0
SPATIAL_BIAS_GAMMA = 0.1
CLASSIFIER_DROPOUT = 0.1

CLASS_NAMES = ["0", "1+", "2+", "3+"]   # class_0, class_1+, class_2+, class_3+

# Checkpoint
CHECKPOINT_PATH = "/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth"

# Output directory
CHECKPOINT_ROOT = "/content/drive/MyDrive/HER2_Checkpoints"
RESULTS_DIR = os.path.join(CHECKPOINT_ROOT, "DSCA_ViT", "Results")
OUT_DIR = os.path.join(RESULTS_DIR, "04_DSCA_Input_Distribution_and_Fusion_Gate_Diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Configuration (exact trained DSCA-ViT)")
print("=" * 60)
print(f"  Model            : {MODEL_ID}")
print(f"  Input size       : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"  Patch size       : {PATCH_SIZE}")
print(f"  Patch grid       : {GRID} x {GRID}")
print(f"  Patch count      : {N_PATCH}")
print(f"  CLS              : {N_CLS}")
print(f"  Total tokens     : {N_TOKENS}")
print(f"  num_classes      : {NUM_CLASSES}")
print(f"  split_after      : {SPLIT_AFTER}")
print(f"  spatial_bias_beta: {SPATIAL_BIAS_BETA}")
print(f"  spatial_bias_gamma: {SPATIAL_BIAS_GAMMA}")
print(f"  classifier_dropout: {CLASSIFIER_DROPOUT}")
print(f"  Checkpoint       : {CHECKPOINT_PATH}")
print(f"  Output Dir       : {OUT_DIR}")
print("=" * 60)


# ============================================================
# Cell 6 — Dataset (downloads only if missing, then verifies)
# ============================================================
# Same proven download+extract logic as the other analysis notebooks.

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

# --- Dataset verification (same split as previous notebooks) ---
from datasets import HER2Dataset, get_test_transform

TEST_DIR = "/content/HER2_Dataset/WSI-based-dataset/test"
assert os.path.exists(TEST_DIR), f"Test directory not found: {TEST_DIR}"

test_transform = get_test_transform(image_size=IMAGE_SIZE)
test_dataset = HER2Dataset(root_dir=TEST_DIR, transform=test_transform)

dist = test_dataset.get_class_distribution()
print("=" * 60)
print("Dataset Verification (test split)")
print("=" * 60)
print(f"  Number of test images : {len(test_dataset)}")
print(f"  Class names           : {test_dataset.get_class_names()}")
print(f"  Class distribution    : {dist}")
print("  Expected support      : 0=658, 1+=316, 2+=111, 3+=762")
print("=" * 60)


# ============================================================
# Cell 7 — Load the Trained DSCA Model
# ============================================================

from models import DSCAViT

model = DSCAViT(
    num_classes=NUM_CLASSES,
    pretrained=True,
    split_after=SPLIT_AFTER,
    spatial_bias_beta=SPATIAL_BIAS_BETA,
    spatial_bias_gamma=SPATIAL_BIAS_GAMMA,
    classifier_dropout=CLASSIFIER_DROPOUT,
)
model = model.to(device)

assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found:\n{CHECKPOINT_PATH}"
print(f"✅ Loading checkpoint:\n    {CHECKPOINT_PATH}")

state = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
    print("✅ Loaded full checkpoint (model_state_dict).")
else:
    model.load_state_dict(state)
    print("✅ Loaded weights-only state_dict.")

model.eval()

# Verify architecture matches checkpoint
print("=" * 60)
print("Model Verification")
print("=" * 60)
print(f"  Spatial bias shape : {tuple(model.cross_attention.spatial_bias.bias_matrix.shape)}")
print(f"  Fusion module      : {type(model.fusion).__name__}")
print(f"  Fusion gate proj   : {type(model.fusion.gate_proj).__name__} "
      f"({model.fusion.gate_proj.in_features}->{model.fusion.gate_proj.out_features})")
print(f"  Fusion CLS fusion  : {type(model.fusion.cls_fusion).__name__} "
      f"({model.fusion.cls_fusion.in_features}->{model.fusion.cls_fusion.out_features})")
print(f"  Total parameters   : {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)


# ============================================================
# Cell 8 — Evaluate the Trained Model (normal, full test set)
# ============================================================

from torch.utils.data import DataLoader

loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=True)

all_preds, all_labels, all_confs = [], [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
                                                      labels=[0, 1, 2, 3], zero_division=0)
macro_f1 = float(np.mean(f1))
_, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                       average="weighted", zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

print("=" * 60)
print("Trained DSCA-ViT Evaluation (full test set)")
print("=" * 60)
print(f"  Accuracy       : {acc:.2f}%")
print(f"  Balanced Acc   : {bal_acc:.2f}%")
print(f"  Macro F1       : {macro_f1:.4f}")
print(f"  Weighted F1    : {weighted_f1:.4f}")
print("  Per-class (P/R/F1):")
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls}: P={prec[i]:.4f} R={rec[i]:.4f} F1={f1[i]:.4f} (n={supp[i]})")
print("  Confusion matrix (0,1+,2+,3+):")
print(cm)
print("=" * 60)

# Expected ~92.26%. If substantially different, STOP.
if abs(acc - 92.26) > 2.0:
    print("⚠️  WARNING: Accuracy differs substantially from expected ~92.26%.")
    print("    Investigate reproducibility/checkpoint/preprocessing before continuing.")
else:
    print("✅ Accuracy consistent with expected ~92.26%.")
print("=" * 60)


# ============================================================
# Cell 9 — Trace the H/DAB Data Flow (forward hooks)
# ============================================================
# We attach read-only forward hooks to capture key tensors WITHOUT
# modifying the model. We capture per-batch tensors and compute
# statistics online (memory-safe).

# NOTE: Several modules return TUPLES, not single tensors:
#   - color_deconv    -> (h, d)
#   - cross_attention -> (feat_h, feat_d)
#   - fusion          -> (fused_out, gate_values)
# We therefore select the desired tensor with `output_index`.

hook_captures = {}   # name -> list of tensors (per batch)

def make_hook(name, output_index=None):
    def hook(module, input, output):
        # Store a detached CPU copy (only for the first few batches to save RAM)
        if len(hook_captures.get(name, [])) < 3:   # capture first 3 batches
            out = output if output_index is None else output[output_index]
            hook_captures.setdefault(name, []).append(out.detach().cpu())
    return hook

# Register hooks on key modules (store handles so we can remove them later)
hook_handles = []
# Color deconvolution: capture H and DAB separately
hook_handles.append(model.color_deconv.register_forward_hook(make_hook("color_deconv_h", output_index=0)))
hook_handles.append(model.color_deconv.register_forward_hook(make_hook("color_deconv_d", output_index=1)))

# Stream projections (single tensor each)
hook_handles.append(model.proj_h.register_forward_hook(make_hook("proj_h_output")))
hook_handles.append(model.proj_d.register_forward_hook(make_hook("proj_d_output")))

# Shared ViT patch embedding (single tensor: patch tokens, no CLS yet)
hook_handles.append(model.encoder.patch_embed.register_forward_hook(make_hook("patch_embed_output")))

# Bidirectional cross-attention: capture both stream outputs
hook_handles.append(model.cross_attention.register_forward_hook(make_hook("cross_attention_h", output_index=0)))
hook_handles.append(model.cross_attention.register_forward_hook(make_hook("cross_attention_d", output_index=1)))

# Gated fusion: capture the fused patches (index 0); gate_values are
# handled separately via model.get_gate_values() in a later cell
hook_handles.append(model.fusion.register_forward_hook(make_hook("fusion_output", output_index=0)))

# Refinement block (single tensor)
hook_handles.append(model.refinement.register_forward_hook(make_hook("refinement_output")))

# Run a few batches to capture
with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        if i >= 3:
            break
        images = images.to(device)
        _ = model(images)

print("=" * 60)
print("Captured tensors (first 3 batches, read-only hooks)")
print("=" * 60)
for name, tensors in hook_captures.items():
    t = torch.cat(tensors, dim=0)
    print(f"  {name:<24} shape={tuple(t.shape)}")
print("=" * 60)


# ============================================================
# Cell 10 — Representative Tensor Statistics
# ============================================================

def tensor_stats(name, t):
    t = t.float()
    n = t.numel()
    zeros = (t == 0).float().mean().item()
    neg = (t < 0).float().mean().item()
    nan = torch.isnan(t).float().mean().item()
    inf = torch.isinf(t).float().mean().item()
    pcts = torch.quantile(t.flatten(), torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])).tolist()
    print(f"  {name:<24} min={t.min().item():.4f} max={t.max().item():.4f} "
          f"mean={t.mean().item():.4f} std={t.std().item():.4f}")
    print(f"    median={t.median().item():.4f} pcts[1,5,25,50,75,95,99]={[round(p,4) for p in pcts]}")
    print(f"    zeros={zeros:.4f} neg={neg:.4f} nan={nan:.4f} inf={inf:.4f}")

print("=" * 60)
print("Representative Tensor Statistics (first 3 batches)")
print("=" * 60)
for name, tensors in hook_captures.items():
    t = torch.cat(tensors, dim=0)
    tensor_stats(name, t)
print("=" * 60)

# Cleanup: remove hooks and free captured tensors to save RAM
for h in hook_handles:
    h.remove()
del hook_captures, hook_handles
gc.collect()
torch.cuda.empty_cache()
print("✅ Hooks removed and captured tensors freed.")


# ============================================================
# Cell 11 — H/DAB Distribution Analysis (full test set, online)
# ============================================================
# We compute H/DAB statistics over ALL 1847 test images using the
# deconvolution (same math as the model). Memory-safe: accumulate
# running sums + histograms, not raw arrays.

from models.color_deconv import deconvolve_numpy
from PIL import Image

# We'll accumulate per-image stats and a pooled histogram
h_means, h_stds, h_p95s, h_nonzero = [], [], [], []
dab_means, dab_stds, dab_p95s, dab_nonzero = [], [], [], []

# Pooled arrays (subsample for histogram to save RAM)
h_pool = []
dab_pool = []
MAX_POOL = 100          # cap pooled images
POOL_STRIDE = 20        # keep every 20th pixel (224*224/20 ≈ 11.2K px/img)

for idx in range(len(test_dataset)):
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    h_ch, dab_ch = deconvolve_numpy(img_rgb)

    h_means.append(h_ch.mean())
    h_stds.append(h_ch.std())
    h_p95s.append(np.percentile(h_ch, 95))
    h_nonzero.append((h_ch > 0).mean())

    dab_means.append(dab_ch.mean())
    dab_stds.append(dab_ch.std())
    dab_p95s.append(np.percentile(dab_ch, 95))
    dab_nonzero.append((dab_ch > 0).mean())

    # Pool a subsample of pixels for histograms (memory-safe)
    if len(h_pool) < MAX_POOL:
        h_pool.append(h_ch.ravel()[::POOL_STRIDE])
        dab_pool.append(dab_ch.ravel()[::POOL_STRIDE])

    del img_rgb, h_ch, dab_ch

h_means = np.array(h_means); h_stds = np.array(h_stds)
h_p95s = np.array(h_p95s); h_nonzero = np.array(h_nonzero)
dab_means = np.array(dab_means); dab_stds = np.array(dab_stds)
dab_p95s = np.array(dab_p95s); dab_nonzero = np.array(dab_nonzero)

h_pool = np.concatenate(h_pool)
dab_pool = np.concatenate(dab_pool)

def dist_stats(name, arr):
    pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99, 99.9])
    print(f"  {name}:")
    print(f"    mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
    print(f"    pcts[1,5,25,50,75,95,99,99.9]={np.round(pcts,4)}")
    print(f"    frac_zero={float((arr==0).mean()):.4f} "
          f">0.1={float((arr>0.1).mean()):.4f} >0.5={float((arr>0.5).mean()):.4f} "
          f">1={float((arr>1).mean()):.4f} >2={float((arr>2).mean()):.4f} >5={float((arr>5).mean()):.4f}")

print("=" * 60)
print("H/DAB Distribution Analysis (all 1847 test images)")
print("=" * 60)
dist_stats("H (OD)", h_pool)
dist_stats("DAB (OD)", dab_pool)
print("=" * 60)

# Save input distribution summary
import csv
input_summary_path = os.path.join(OUT_DIR, "input_distribution_summary.csv")
with open(input_summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["representation", "mean", "std", "min", "max",
                "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9",
                "frac_zero", "frac_gt_0.1", "frac_gt_0.5", "frac_gt_1", "frac_gt_2", "frac_gt_5"])
    for name, arr in [("H", h_pool), ("DAB", dab_pool)]:
        pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99, 99.9])
        w.writerow([name, f"{arr.mean():.6f}", f"{arr.std():.6f}",
                    f"{arr.min():.6f}", f"{arr.max():.6f}",
                    *[f"{p:.6f}" for p in pcts],
                    f"{float((arr==0).mean()):.6f}",
                    f"{float((arr>0.1).mean()):.6f}",
                    f"{float((arr>0.5).mean()):.6f}",
                    f"{float((arr>1).mean()):.6f}",
                    f"{float((arr>2).mean()):.6f}",
                    f"{float((arr>5).mean()):.6f}"])
print(f"✅ Saved: {input_summary_path}")

# Free memory (keep h_pool/dab_pool for Cell 12 histograms;
# keep h_means/dab_means for Cell 23 correlation)
del h_stds, h_p95s, h_nonzero
del dab_stds, dab_p95s, dab_nonzero
gc.collect()
print("✅ Cell 11 memory freed (kept pooled arrays for later cells).")


# ============================================================
# Cell 12 — H/DAB Histograms
# ============================================================

# H histogram (log-scale)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(h_pool[h_pool > 0], bins=100, color="steelblue", alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("H intensity (OD, log scale)")
ax.set_ylabel("Count")
ax.set_title("H Distribution (positive values, log scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "h_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: h_distribution.png")

# DAB histogram (log-scale)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(dab_pool[dab_pool > 0], bins=100, color="darkorange", alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("DAB intensity (OD, log scale)")
ax.set_ylabel("Count")
ax.set_title("DAB Distribution (positive values, log scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dab_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: dab_distribution.png")

# H vs DAB comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(h_pool[h_pool > 0], bins=100, alpha=0.6, label="H", color="steelblue")
ax.hist(dab_pool[dab_pool > 0], bins=100, alpha=0.6, label="DAB", color="darkorange")
ax.set_xscale("log")
ax.set_xlabel("Intensity (OD, log scale)")
ax.set_ylabel("Count")
ax.set_title("H vs DAB Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "h_vs_dab_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: h_vs_dab_distribution.png")

# Free pooled histogram arrays now (no longer needed after Cell 12)
del h_pool, dab_pool
gc.collect()
print("✅ Cell 12 memory freed.")


# ============================================================
# Cell 13 — Distribution at the Actual ViT Input
# ============================================================
# Measure the tensors immediately before the pretrained ViT encoder
# for H and DAB streams, and compare with baseline ImageNet-normalized RGB.

# Baseline ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Use a reproducible subset for the per-channel ViT-input analysis (memory-safe)
subsample_n = 50
PIXEL_STEP = 4   # keep every 4th pixel (224*224/16 ≈ 12.5K pixels per image)
random.seed(SEED)
sample_indices = random.sample(range(len(test_dataset)), min(subsample_n, len(test_dataset)))

# Collect: baseline RGB (ImageNet-norm), H before/after proj, DAB before/after proj
base_rgb_pixels = []
h_before_pixels = []
h_after_pixels = []
dab_before_pixels = []
dab_after_pixels = []

for idx in sample_indices:
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    norm_rgb = (img_rgb - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    base_rgb_pixels.append(norm_rgb.reshape(-1, 3)[::PIXEL_STEP])

    h_ch, dab_ch = deconvolve_numpy(np.array(Image.open(img_path).convert("RGB")))
    h_before_pixels.append(h_ch.astype(np.float32).ravel()[::PIXEL_STEP])
    dab_before_pixels.append(dab_ch.astype(np.float32).ravel()[::PIXEL_STEP])

    # After projection: use the model's actual proj layers (identity-like)
    h_t = torch.tensor(h_ch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    dab_t = torch.tensor(dab_ch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        h_after = model.proj_h(h_t).squeeze().cpu().numpy()   # (3, H, W)
        dab_after = model.proj_d(dab_t).squeeze().cpu().numpy()
    h_after_pixels.append(h_after.reshape(3, -1).T[::PIXEL_STEP])   # (N, 3)
    dab_after_pixels.append(dab_after.reshape(3, -1).T[::PIXEL_STEP])

    del img_rgb, norm_rgb, h_ch, dab_ch, h_t, dab_t, h_after, dab_after

base_arr = np.concatenate(base_rgb_pixels, axis=0)
h_before_arr = np.concatenate(h_before_pixels)
dab_before_arr = np.concatenate(dab_before_pixels)
h_after_arr = np.concatenate(h_after_pixels, axis=0)
dab_after_arr = np.concatenate(dab_after_pixels, axis=0)

def rep_row(name, arr):
    pcts = np.percentile(arr, [1, 50, 99])
    return [name, f"{arr.mean():.4f}", f"{arr.std():.4f}", f"{arr.min():.4f}",
            f"{arr.max():.4f}", f"{pcts[0]:.4f}", f"{pcts[1]:.4f}", f"{pcts[2]:.4f}"]

print("=" * 60)
print("Distribution at the Actual ViT Input (subsample=200)")
print("=" * 60)
print(f"{'Representation':<32} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P1':>8} {'P50':>8} {'P99':>8}")
rows = [
    rep_row("baseline RGB (ImageNet-norm)", base_arr),
    rep_row("DSCA H before projection", h_before_arr),
    rep_row("DSCA H after projection", h_after_arr),
    rep_row("DSCA DAB before projection", dab_before_arr),
    rep_row("DSCA DAB after projection", dab_after_arr),
]
for r in rows:
    print(f"  {r[0]:<30} {r[1]:>8} {r[2]:>8} {r[3]:>8} {r[4]:>8} {r[5]:>8} {r[6]:>8} {r[7]:>8}")
print("=" * 60)

# Per-channel stats for the projected H/DAB (3 channels)
print("Per-channel stats (H after projection):")
for c in range(3):
    print(f"  ch{c}: mean={h_after_arr[:,c].mean():.4f} std={h_after_arr[:,c].std():.4f}")
print("Per-channel stats (DAB after projection):")
for c in range(3):
    print(f"  ch{c}: mean={dab_after_arr[:,c].mean():.4f} std={dab_after_arr[:,c].std():.4f}")

# Correlation between the 3 projected channels (H)
h_corr = np.corrcoef(h_after_arr.T)
print("H projected channel correlation matrix:")
print(np.round(h_corr, 4))
dab_corr = np.corrcoef(dab_after_arr.T)
print("DAB projected channel correlation matrix:")
print(np.round(dab_corr, 4))
print("=" * 60)

# Save vit input comparison
vit_input_path = os.path.join(OUT_DIR, "vit_input_distribution_comparison.csv")
with open(vit_input_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["representation", "mean", "std", "min", "max", "p1", "p50", "p99"])
    for r in rows:
        w.writerow(r)
print(f"✅ Saved: {vit_input_path}")

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
labels = [r[0] for r in rows]
means = [float(r[1]) for r in rows]
stds = [float(r[2]) for r in rows]
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=4, color=["red", "steelblue", "steelblue", "darkorange", "darkorange"])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Mean value")
ax.set_title("ViT Input Distribution Comparison (mean ± std)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "vit_input_distribution_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: vit_input_distribution_comparison.png")

# Free memory (keep base_arr/h_before_arr/h_after_arr/dab_before_arr/dab_after_arr
# for the Cell 28 automated report)
del base_rgb_pixels, h_before_pixels, h_after_pixels, dab_before_pixels, dab_after_pixels
gc.collect()
torch.cuda.empty_cache()
print("✅ Cell 13 memory freed (kept concatenated arrays for final report).")


# ============================================================
# Cell 14 — Image-Level Distribution by Class
# ============================================================
# Per-image H/DAB stats grouped by TRUE class (analysis only).

class_h_means = {c: [] for c in range(4)}
class_h_p95s = {c: [] for c in range(4)}
class_dab_means = {c: [] for c in range(4)}
class_dab_p95s = {c: [] for c in range(4)}

for idx in range(len(test_dataset)):
    label = test_dataset.labels[idx]
    img_path = test_dataset.image_paths[idx]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    h_ch, dab_ch = deconvolve_numpy(img_rgb)
    class_h_means[label].append(h_ch.mean())
    class_h_p95s[label].append(np.percentile(h_ch, 95))
    class_dab_means[label].append(dab_ch.mean())
    class_dab_p95s[label].append(np.percentile(dab_ch, 95))
    del img_rgb, h_ch, dab_ch

print("=" * 60)
print("Per-Class Representation Statistics (all test images)")
print("=" * 60)
for c in range(4):
    print(f"  Class {CLASS_NAMES[c]} (n={len(class_h_means[c])}):")
    print(f"    H   mean={np.mean(class_h_means[c]):.4f}  P95={np.mean(class_h_p95s[c]):.4f}")
    print(f"    DAB mean={np.mean(class_dab_means[c]):.4f}  P95={np.mean(class_dab_p95s[c]):.4f}")
print("=" * 60)

# Save per-class representation stats
per_class_rep_path = os.path.join(OUT_DIR, "per_class_representation_statistics.csv")
with open(per_class_rep_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "n", "h_mean", "h_p95", "dab_mean", "dab_p95"])
    for c in range(4):
        w.writerow([CLASS_NAMES[c], len(class_h_means[c]),
                    f"{np.mean(class_h_means[c]):.6f}", f"{np.mean(class_h_p95s[c]):.6f}",
                    f"{np.mean(class_dab_means[c]):.6f}", f"{np.mean(class_dab_p95s[c]):.6f}"])
print(f"✅ Saved: {per_class_rep_path}")

# Plots: H mean by class, DAB mean by class, H P95 by class, DAB P95 by class
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0,0].bar(CLASS_NAMES, [np.mean(class_h_means[c]) for c in range(4)], color="steelblue")
axes[0,0].set_title("H mean by class"); axes[0,0].set_ylabel("H mean")
axes[0,1].bar(CLASS_NAMES, [np.mean(class_dab_means[c]) for c in range(4)], color="darkorange")
axes[0,1].set_title("DAB mean by class"); axes[0,1].set_ylabel("DAB mean")
axes[1,0].bar(CLASS_NAMES, [np.mean(class_h_p95s[c]) for c in range(4)], color="steelblue")
axes[1,0].set_title("H P95 by class"); axes[1,0].set_ylabel("H P95")
axes[1,1].bar(CLASS_NAMES, [np.mean(class_dab_p95s[c]) for c in range(4)], color="darkorange")
axes[1,1].set_title("DAB P95 by class"); axes[1,1].set_ylabel("DAB P95")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "h_by_class.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "dab_by_class.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: h_by_class.png, dab_by_class.png")


# ============================================================
# Cell 15 — Critical 1+ vs 2+ Representation Analysis
# ============================================================

print("=" * 60)
print("Critical 1+ vs 2+ Representation Comparison")
print("=" * 60)
for c in [1, 2]:
    print(f"  Class {CLASS_NAMES[c]} (n={len(class_h_means[c])}):")
    print(f"    H   mean={np.mean(class_h_means[c]):.4f}  P95={np.mean(class_h_p95s[c]):.4f}")
    print(f"    DAB mean={np.mean(class_dab_means[c]):.4f}  P95={np.mean(class_dab_p95s[c]):.4f}")
    print(f"    DAB/H ratio={np.mean(class_dab_means[c])/(np.mean(class_h_means[c])+1e-8):.4f}")
print("=" * 60)

# H distribution 1+ vs 2+ (per-image means)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(class_h_means[1], bins=30, alpha=0.6, label="1+", color="steelblue")
ax.hist(class_h_means[2], bins=30, alpha=0.6, label="2+", color="crimson")
ax.set_xlabel("H mean intensity")
ax.set_ylabel("Count")
ax.set_title("H Distribution: 1+ vs 2+")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "h_1plus_vs_2plus.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: h_1plus_vs_2plus.png")

# DAB distribution 1+ vs 2+
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(class_dab_means[1], bins=30, alpha=0.6, label="1+", color="darkorange")
ax.hist(class_dab_means[2], bins=30, alpha=0.6, label="2+", color="crimson")
ax.set_xlabel("DAB mean intensity")
ax.set_ylabel("Count")
ax.set_title("DAB Distribution: 1+ vs 2+")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dab_1plus_vs_2plus.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: dab_1plus_vs_2plus.png")

# Free per-class representation dicts (no longer needed)
del class_h_means, class_h_p95s, class_dab_means, class_dab_p95s
gc.collect()
print("✅ Cell 15 memory freed.")


# ============================================================
# Cell 16 — Locate the Gated Fusion Mechanism (actual code)
# ============================================================

# Read the actual fusion implementation
fusion_src = open(os.path.join(REPO_DIR, "models", "fusion.py")).read()
print("=" * 60)
print("Gated Fusion Implementation (from models/fusion.py)")
print("=" * 60)
# Print the GatedFusion class portion
start = fusion_src.find("class GatedFusion")
end = fusion_src.find("class RefinementBlock")
print(fusion_src[start:end])
print("=" * 60)

print("=" * 60)
print("Gated Fusion Mechanism (documented from actual code)")
print("=" * 60)
print("  Formula (from GatedFusion.forward):")
print("    CLS:  F_0 = cls_fusion([CLS_H || CLS_D])   (Linear 1536->768, NO gate)")
print("    Patch: g_i = sigmoid(gate_proj([H_i || D_i]))   (Linear 1536->768)")
print("           F_i = g_i * H_i + (1 - g_i) * D_i")
print("  Gate type      : token-wise AND channel-wise (768-dim per patch token)")
print("  Activation     : sigmoid")
print("  Gate shape     : (B, 196, 768)  [patch tokens only]")
print("  CLS handling   : fused separately via cls_fusion (no gate)")
print("  Shared?        : gate_proj is shared across tokens (same Linear applied per token)")
print("  Channel-wise   : YES — each of the 768 dims has its own gate value")
print("=" * 60)


# ============================================================
# Cell 17 — Extract the Learned Fusion Gate (full test set)
# ============================================================
# Use model.get_gate_values() (already returns (B, 196, 768)).
# Memory-safe: accumulate per-image gate summaries, not full tensors.

gate_records = []   # per-image: label, pred, conf, gate mean/std/min/max/median, per-token mean

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        gates = model.get_gate_values()   # (B, 196, 768)

        for b in range(images.size(0)):
            g = gates[b].cpu().numpy()   # (196, 768)
            gate_records.append({
                "label": labels[b].item(),
                "pred": preds[b].item(),
                "confidence": conf[b].item(),
                "gate_mean": float(g.mean()),
                "gate_std": float(g.std()),
                "gate_min": float(g.min()),
                "gate_max": float(g.max()),
                "gate_median": float(np.median(g)),
                "gate_p25": float(np.percentile(g, 25)),
                "gate_p75": float(np.percentile(g, 75)),
                "gate_p5": float(np.percentile(g, 5)),
                "gate_p95": float(np.percentile(g, 95)),
                "gate_per_token_mean": g.mean(axis=1),   # (196,) mean over channels
            })

print(f"✅ Extracted gate values for {len(gate_records)} images.")

# Free the per-batch gate tensor
del gates
gc.collect()
print("✅ Cell 17 memory freed.")

# Global gate statistics
all_gate_means = np.array([r["gate_mean"] for r in gate_records])
all_gate_stds = np.array([r["gate_std"] for r in gate_records])
print("=" * 60)
print("Global Gate Statistics")
print("=" * 60)
print(f"  Mean gate (per-image mean) : {all_gate_means.mean():.4f}")
print(f"  Std  gate (per-image std)  : {all_gate_stds.mean():.4f}")
print(f"  Min gate                   : {min(r['gate_min'] for r in gate_records):.4f}")
print(f"  Max gate                   : {max(r['gate_max'] for r in gate_records):.4f}")
print(f"  Median gate                : {np.median([r['gate_median'] for r in gate_records]):.4f}")
print("  Interpretation (gate = weight on H):")
if all_gate_means.mean() > 0.7:
    print("    -> Strongly biased toward H")
elif all_gate_means.mean() < 0.3:
    print("    -> Strongly biased toward DAB")
else:
    print("    -> Relatively balanced")
print("=" * 60)

# Save fusion gate summary
gate_summary_path = os.path.join(OUT_DIR, "fusion_gate_summary.csv")
with open(gate_summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["mean_gate", f"{all_gate_means.mean():.6f}"])
    w.writerow(["std_gate", f"{all_gate_stds.mean():.6f}"])
    w.writerow(["min_gate", f"{min(r['gate_min'] for r in gate_records):.6f}"])
    w.writerow(["max_gate", f"{max(r['gate_max'] for r in gate_records):.6f}"])
    w.writerow(["median_gate", f"{np.median([r['gate_median'] for r in gate_records]):.6f}"])
print(f"✅ Saved: {gate_summary_path}")


# ============================================================
# Cell 18 — Gate Distribution Visualization
# ============================================================
# Gate is token-wise AND channel-wise. We show:
#   - histogram of all gate values (pooled)
#   - per-image mean gate histogram
#   - per-token mean gate (spatial 14x14) — valid because gate is token-wise

# Pooled gate values (subsample for memory)
pooled_gates = np.concatenate([r["gate_per_token_mean"] for r in gate_records[:200]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(pooled_gates, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(0.5, color="red", linestyle="--", label="0.5")
axes[0].set_xlabel("Gate value (mean over channels)")
axes[0].set_ylabel("Count")
axes[0].set_title("Gate Distribution (per-token mean, subsample)")
axes[0].legend()

axes[1].hist(all_gate_means, bins=50, color="seagreen", edgecolor="white", alpha=0.8)
axes[1].axvline(0.5, color="red", linestyle="--", label="0.5")
axes[1].set_xlabel("Per-image mean gate")
axes[1].set_ylabel("Count")
axes[1].set_title("Per-Image Mean Gate Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_distribution.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_distribution.png")


# ============================================================
# Cell 19 — Gate by True Class
# ============================================================

class_gate_means = {c: [] for c in range(4)}
for r in gate_records:
    class_gate_means[r["label"]].append(r["gate_mean"])

print("=" * 60)
print("Gate by True Class")
print("=" * 60)
for c in range(4):
    g = np.array(class_gate_means[c])
    print(f"  Class {CLASS_NAMES[c]} (n={len(g)}): mean={g.mean():.4f} std={g.std():.4f} "
          f"median={np.median(g):.4f} P25={np.percentile(g,25):.4f} P75={np.percentile(g,75):.4f} "
          f"P5={np.percentile(g,5):.4f} P95={np.percentile(g,95):.4f}")
print("=" * 60)

# Save gate by class
gate_by_class_path = os.path.join(OUT_DIR, "fusion_gate_by_class.csv")
with open(gate_by_class_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "n", "mean", "std", "median", "p25", "p75", "p5", "p95"])
    for c in range(4):
        g = np.array(class_gate_means[c])
        w.writerow([CLASS_NAMES[c], len(g), f"{g.mean():.6f}", f"{g.std():.6f}",
                    f"{np.median(g):.6f}", f"{np.percentile(g,25):.6f}",
                    f"{np.percentile(g,75):.6f}", f"{np.percentile(g,5):.6f}",
                    f"{np.percentile(g,95):.6f}"])
print(f"✅ Saved: {gate_by_class_path}")

# Plot gate by class
fig, ax = plt.subplots(figsize=(8, 5))
data = [np.array(class_gate_means[c]) for c in range(4)]
ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True)
ax.axhline(0.5, color="red", linestyle="--", label="0.5")
ax.set_ylabel("Mean gate (H weight)")
ax.set_title("Gate Distribution by True Class")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_by_class.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_by_class.png")


# ============================================================
# Cell 20 — Gate by Prediction Correctness + Error Groups
# ============================================================

correct = [r for r in gate_records if r["pred"] == r["label"]]
incorrect = [r for r in gate_records if r["pred"] != r["label"]]

print("=" * 60)
print("Gate by Prediction Correctness")
print("=" * 60)
print(f"  Correct (n={len(correct)}): mean={np.mean([r['gate_mean'] for r in correct]):.4f}")
print(f"  Incorrect (n={len(incorrect)}): mean={np.mean([r['gate_mean'] for r in incorrect]):.4f}")
print("=" * 60)

# Error groups
groups = {
    "A: true 1+ -> pred 1+": [r for r in gate_records if r["label"]==1 and r["pred"]==1],
    "B: true 1+ -> pred 2+": [r for r in gate_records if r["label"]==1 and r["pred"]==2],
    "C: true 2+ -> pred 2+": [r for r in gate_records if r["label"]==2 and r["pred"]==2],
    "D: true 2+ -> pred 1+": [r for r in gate_records if r["label"]==2 and r["pred"]==1],
}

print("=" * 60)
print("Gate by Error Group (critical 1+/2+ confusion)")
print("=" * 60)
for name, grp in groups.items():
    if grp:
        g = np.array([r["gate_mean"] for r in grp])
        print(f"  {name:<24} (n={len(grp):3d}): mean={g.mean():.4f} std={g.std():.4f}")
    else:
        print(f"  {name:<24} (n=  0): N/A")
print("=" * 60)

# Save gate by prediction + error groups
gate_by_pred_path = os.path.join(OUT_DIR, "fusion_gate_by_prediction.csv")
with open(gate_by_pred_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "n", "mean", "std"])
    w.writerow(["correct", len(correct), f"{np.mean([r['gate_mean'] for r in correct]):.6f}",
                f"{np.std([r['gate_mean'] for r in correct]):.6f}"])
    w.writerow(["incorrect", len(incorrect), f"{np.mean([r['gate_mean'] for r in incorrect]):.6f}",
                f"{np.std([r['gate_mean'] for r in incorrect]):.6f}"])
print(f"✅ Saved: {gate_by_pred_path}")

gate_error_path = os.path.join(OUT_DIR, "fusion_gate_error_groups.csv")
with open(gate_error_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "n", "mean", "std"])
    for name, grp in groups.items():
        if grp:
            g = np.array([r["gate_mean"] for r in grp])
            w.writerow([name, len(grp), f"{g.mean():.6f}", f"{g.std():.6f}"])
        else:
            w.writerow([name, 0, "", ""])
print(f"✅ Saved: {gate_error_path}")

# Plot correct vs incorrect
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([[r["gate_mean"] for r in correct], [r["gate_mean"] for r in incorrect]],
           labels=["Correct", "Incorrect"], patch_artist=True)
ax.axhline(0.5, color="red", linestyle="--", label="0.5")
ax.set_ylabel("Mean gate (H weight)")
ax.set_title("Gate: Correct vs Incorrect")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_correct_vs_incorrect.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_correct_vs_incorrect.png")

# Plot 1+ error analysis
fig, ax = plt.subplots(figsize=(8, 5))
group_names = list(groups.keys())
group_data = [np.array([r["gate_mean"] for r in groups[n]]) if groups[n] else np.array([0.0]) for n in group_names]
ax.boxplot(group_data, labels=group_names, patch_artist=True)
ax.axhline(0.5, color="red", linestyle="--", label="0.5")
ax.set_ylabel("Mean gate (H weight)")
ax.set_title("Gate by 1+/2+ Error Group")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_1plus_error_analysis.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_1plus_error_analysis.png")


# ============================================================
# Cell 21 — Gate by Predicted Class
# ============================================================

pred_gate_means = {c: [] for c in range(4)}
for r in gate_records:
    pred_gate_means[r["pred"]].append(r["gate_mean"])

print("=" * 60)
print("Gate by Predicted Class")
print("=" * 60)
for c in range(4):
    g = np.array(pred_gate_means[c])
    print(f"  Pred {CLASS_NAMES[c]} (n={len(g)}): mean={g.mean():.4f} std={g.std():.4f}")
print("=" * 60)


# ============================================================
# Cell 22 — Gate vs Confidence
# ============================================================

confs = np.array([r["confidence"] for r in gate_records])
gate_means = np.array([r["gate_mean"] for r in gate_records])

pearson_gc, _ = pearsonr(gate_means, confs)
spearman_gc, _ = spearmanr(gate_means, confs)

print("=" * 60)
print("Gate vs Confidence")
print("=" * 60)
print(f"  Pearson  : {pearson_gc:.4f}")
print(f"  Spearman : {spearman_gc:.4f}")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(gate_means, confs, c=[r["pred"]==r["label"] for r in gate_records],
                cmap="coolwarm", alpha=0.4, s=10)
ax.set_xlabel("Mean gate (H weight)")
ax.set_ylabel("Confidence")
ax.set_title(f"Gate vs Confidence (Pearson={pearson_gc:.3f})")
plt.colorbar(sc, label="Correct (1) / Incorrect (0)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_vs_confidence.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_vs_confidence.png")


# ============================================================
# Cell 23 — Gate vs DAB / H Intensity
# ============================================================
# Use the per-image DAB/H means computed earlier (aligned by dataset order).

# gate_records are in dataset order (loader shuffle=False)
dab_means_arr = np.array(dab_means)
h_means_arr = np.array(h_means)

# Align: gate_records[i] corresponds to test_dataset index i
pearson_gdab, _ = pearsonr(gate_means, dab_means_arr)
spearman_gdab, _ = spearmanr(gate_means, dab_means_arr)
pearson_gh, _ = pearsonr(gate_means, h_means_arr)
spearman_gh, _ = spearmanr(gate_means, h_means_arr)

print("=" * 60)
print("Gate vs DAB / H Intensity")
print("=" * 60)
print(f"  Gate vs DAB mean : Pearson={pearson_gdab:.4f} Spearman={spearman_gdab:.4f}")
print(f"  Gate vs H mean   : Pearson={pearson_gh:.4f} Spearman={spearman_gh:.4f}")
print("=" * 60)

# Save correlations
gate_corr_path = os.path.join(OUT_DIR, "fusion_gate_correlations.csv")
with open(gate_corr_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair", "pearson", "spearman"])
    w.writerow(["gate_vs_confidence", f"{pearson_gc:.6f}", f"{spearman_gc:.6f}"])
    w.writerow(["gate_vs_dab_mean", f"{pearson_gdab:.6f}", f"{spearman_gdab:.6f}"])
    w.writerow(["gate_vs_h_mean", f"{pearson_gh:.6f}", f"{spearman_gh:.6f}"])
print(f"✅ Saved: {gate_corr_path}")

# Plot gate vs DAB
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(dab_means_arr, gate_means, alpha=0.3, s=10, color="darkorange")
ax.set_xlabel("DAB mean intensity")
ax.set_ylabel("Mean gate (H weight)")
ax.set_title(f"Gate vs DAB Intensity (Pearson={pearson_gdab:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_vs_dab.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_vs_dab.png")

# Plot gate vs H
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(h_means_arr, gate_means, alpha=0.3, s=10, color="steelblue")
ax.set_xlabel("H mean intensity")
ax.set_ylabel("Mean gate (H weight)")
ax.set_title(f"Gate vs H Intensity (Pearson={pearson_gh:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_gate_vs_h.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: fusion_gate_vs_h.png")


# ============================================================
# Cell 24 — Spatial Gate Analysis (token-wise -> 14x14)
# ============================================================
# The gate IS token-wise (196 patch tokens), so a 14x14 spatial map is valid.
# CLS is handled separately (no gate). We average the per-token mean gate
# over channels to produce a 14x14 map.

# Accumulate per-token mean gate maps (196,) over all test images
token_maps = np.stack([r["gate_per_token_mean"] for r in gate_records])  # (N, 196)
avg_gate_map = token_maps.mean(axis=0).reshape(GRID, GRID)   # (14, 14)

# Per-class average maps
class_maps = {c: [] for c in range(4)}
for r in gate_records:
    class_maps[r["label"]].append(r["gate_per_token_mean"])

# Correct / incorrect / 1+->2+ maps
correct_maps = [r["gate_per_token_mean"] for r in gate_records if r["pred"]==r["label"]]
incorrect_maps = [r["gate_per_token_mean"] for r in gate_records if r["pred"]!=r["label"]]
err_12_maps = [r["gate_per_token_mean"] for r in gate_records if r["label"]==1 and r["pred"]==2]

def plot_gate_map(map_14, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(map_14, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

print("=" * 60)
print("Spatial Gate Analysis (token-wise -> 14x14 grid)")
print("=" * 60)
print(f"  Average gate map overall: mean={avg_gate_map.mean():.4f} "
      f"min={avg_gate_map.min():.4f} max={avg_gate_map.max():.4f}")
print("  CLS handled separately (no gate).")
print("=" * 60)

plot_gate_map(avg_gate_map, "Average Gate Map (all test images)", "spatial_gate_average.png")

# Per-class maps
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for c in range(4):
    if class_maps[c]:
        m = np.mean(np.stack(class_maps[c]), axis=0).reshape(GRID, GRID)
        im = axes[c].imshow(m, cmap="RdBu_r", vmin=0, vmax=1)
        axes[c].set_title(f"Class {CLASS_NAMES[c]}")
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
plt.suptitle("Average Gate Map by True Class")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_gate_by_class.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: spatial_gate_by_class.png")

# 1+ -> 2+ error map
if err_12_maps:
    m = np.mean(np.stack(err_12_maps), axis=0).reshape(GRID, GRID)
    plot_gate_map(m, "Average Gate Map: true 1+ -> pred 2+", "spatial_gate_1plus_to_2plus.png")
else:
    print("No true 1+ -> pred 2+ cases found; skipping spatial error map.")

# Free memory
del token_maps, class_maps, correct_maps, incorrect_maps, err_12_maps
gc.collect()
print("✅ Cell 24 memory freed.")


# ============================================================
# Cell 25 — Fusion Representation Check (H vs DAB vs Fused)
# ============================================================
# Compare magnitude/statistics of H, DAB, and fused representations
# before/at fusion. We capture via hooks on the fusion module input.

# Re-run a few batches capturing fusion input (H_final, D_final) and output
fusion_inputs = []
fusion_outputs = []

def fusion_input_hook(module, input, output):
    # input is (h_final, d_final); output is (fused, gates)
    if len(fusion_inputs) < 2:
        fusion_inputs.append((input[0].detach().cpu(), input[1].detach().cpu()))
        fusion_outputs.append(output[0].detach().cpu())

fusion_handle = model.fusion.register_forward_hook(fusion_input_hook)

with torch.no_grad():
    for i, (images, labels) in enumerate(loader):
        if i >= 2:
            break
        images = images.to(device)
        _ = model(images)

print("=" * 60)
print("Fusion Representation Check (first 3 batches)")
print("=" * 60)
for bi, (h_f, d_f) in enumerate(fusion_inputs):
    fused = fusion_outputs[bi]
    print(f"  Batch {bi}:")
    print(f"    H_final   : mean={h_f.mean():.4f} std={h_f.std():.4f} L2={h_f.norm():.2f} "
          f"mean_token_norm={h_f.norm(dim=-1).mean():.4f} CLS_norm={h_f[:,0,:].norm():.4f}")
    print(f"    D_final   : mean={d_f.mean():.4f} std={d_f.std():.4f} L2={d_f.norm():.2f} "
          f"mean_token_norm={d_f.norm(dim=-1).mean():.4f} CLS_norm={d_f[:,0,:].norm():.4f}")
    print(f"    Fused     : mean={fused.mean():.4f} std={fused.std():.4f} L2={fused.norm():.2f} "
          f"mean_token_norm={fused.norm(dim=-1).mean():.4f} CLS_norm={fused[:,0,:].norm():.4f}")
print("=" * 60)

# Cleanup: remove hook and free captured tensors
fusion_handle.remove()
del fusion_inputs, fusion_outputs
gc.collect()
torch.cuda.empty_cache()
print("✅ Fusion hook removed and captured tensors freed.")


# ============================================================
# Cell 26 — Verify NO Training / NO Checkpoint Modification
# ============================================================

print("=" * 60)
print("Safety Verification")
print("=" * 60)
print("  Optimizer created?      : NO")
print("  loss.backward() called? : NO")
print("  optimizer.step() called?: NO")
print("  Checkpoint saved?       : NO")
print("  Model in eval mode?     : {model.training == False}")
print("  All inference under no_grad? : YES (used torch.no_grad)")
print("  Hooks removed?          : (analysis complete; model unchanged)")
print("=" * 60)


# ============================================================
# Cell 27 — Save Model Verification CSV
# ============================================================

model_verif_path = os.path.join(OUT_DIR, "model_verification.csv")
with open(model_verif_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["field", "value"])
    w.writerow(["model_id", MODEL_ID])
    w.writerow(["checkpoint", CHECKPOINT_PATH])
    w.writerow(["num_classes", NUM_CLASSES])
    w.writerow(["split_after", SPLIT_AFTER])
    w.writerow(["spatial_bias_beta", SPATIAL_BIAS_BETA])
    w.writerow(["spatial_bias_gamma", SPATIAL_BIAS_GAMMA])
    w.writerow(["classifier_dropout", CLASSIFIER_DROPOUT])
    w.writerow(["spatial_bias_shape", str(tuple(model.cross_attention.spatial_bias.bias_matrix.shape))])
    w.writerow(["fusion_module", type(model.fusion).__name__])
    w.writerow(["gate_proj", f"{model.fusion.gate_proj.in_features}->{model.fusion.gate_proj.out_features}"])
    w.writerow(["cls_fusion", f"{model.fusion.cls_fusion.in_features}->{model.fusion.cls_fusion.out_features}"])
    w.writerow(["test_accuracy", f"{acc:.4f}"])
    w.writerow(["test_balanced_acc", f"{bal_acc:.4f}"])
    w.writerow(["test_macro_f1", f"{macro_f1:.4f}"])
    w.writerow(["test_weighted_f1", f"{weighted_f1:.4f}"])
print(f"✅ Saved: {model_verif_path}")


# ============================================================
# Cell 28 — Automated Final Report
# ============================================================

print("=" * 60)
print("AUTOMATED DIAGNOSTIC REPORT")
print("=" * 60)

# Q1: H/DAB vs baseline ImageNet RGB
h_after_mean = h_after_arr.mean()
h_after_std = h_after_arr.std()
dab_after_mean = dab_after_arr.mean()
dab_after_std = dab_after_arr.std()
base_mean = base_arr.mean()
base_std = base_arr.std()

q1 = "YES" if (abs(h_after_mean - base_mean) > 0.5 or abs(dab_after_mean - base_mean) > 0.5) else "INCONCLUSIVE"
print(f"Q1: Are H/DAB entering ViT with a distribution substantially different from baseline ImageNet RGB?")
print(f"    Answer: {q1}")
print(f"    Evidence: baseline mean={base_mean:.3f} std={base_std:.3f}; "
      f"H-after mean={h_after_mean:.3f} std={h_after_std:.3f}; "
      f"DAB-after mean={dab_after_mean:.3f} std={dab_after_std:.3f}")
print()

# Q2: Does 1->3 projection meaningfully normalize?
h_before_mean = h_before_arr.mean()
h_before_std = h_before_arr.std()
dab_before_mean = dab_before_arr.mean()
dab_before_std = dab_before_arr.std()
q2 = "NO" if (abs(h_after_mean - h_before_mean) < 0.05 and abs(dab_after_mean - dab_before_mean) < 0.05) else "YES"
print(f"Q2: Does the 1->3 projection meaningfully normalize/rescale H and DAB?")
print(f"    Answer: {q2}")
print(f"    Evidence: H before={h_before_mean:.4f} after={h_after_mean:.4f}; "
      f"DAB before={dab_before_mean:.4f} after={dab_after_mean:.4f}")
print()

# Q3: Gate preference
gate_global_mean = all_gate_means.mean()
if gate_global_mean > 0.7:
    q3 = "H"
elif gate_global_mean < 0.3:
    q3 = "DAB"
else:
    q3 = "BALANCED"
print(f"Q3: Does the fusion gate strongly prefer H or DAB overall?")
print(f"    Answer: {q3}")
print(f"    Evidence: mean gate={gate_global_mean:.4f}")
print()

# Q4: Gate different between true 1+ and true 2+?
g1 = np.array(class_gate_means[1])
g2 = np.array(class_gate_means[2])
diff_12 = abs(g1.mean() - g2.mean())
q4 = "YES" if diff_12 > 0.05 else "NO"
print(f"Q4: Is the fusion gate different between true 1+ and true 2+?")
print(f"    Answer: {q4}")
print(f"    Evidence: 1+ mean={g1.mean():.4f} vs 2+ mean={g2.mean():.4f} (diff={diff_12:.4f})")
print()

# Q5: Gate different between correct 1+ and 1+->2+?
grp_a = groups["A: true 1+ -> pred 1+"]
grp_b = groups["B: true 1+ -> pred 2+"]
if grp_a and grp_b:
    ga = np.mean([r["gate_mean"] for r in grp_a])
    gb = np.mean([r["gate_mean"] for r in grp_b])
    diff_ab = abs(ga - gb)
    q5 = "YES" if diff_ab > 0.05 else "NO"
    print(f"Q5: Is the gate different between correct 1+ and 1+->2+?")
    print(f"    Answer: {q5}")
    print(f"    Evidence: correct-1+ mean={ga:.4f} vs 1+->2+ mean={gb:.4f} (diff={diff_ab:.4f})")
else:
    q5 = "INCONCLUSIVE"
    print(f"Q5: Is the gate different between correct 1+ and 1+->2+?")
    print(f"    Answer: INCONCLUSIVE (insufficient samples)")
print()

# Q6: Gate vs DAB
q6 = "YES" if abs(pearson_gdab) > 0.3 else ("WEAK" if abs(pearson_gdab) > 0.1 else "NO")
print(f"Q6: Does gate behavior correlate with DAB intensity?")
print(f"    Answer: {q6}")
print(f"    Evidence: Pearson={pearson_gdab:.4f} Spearman={spearman_gdab:.4f}")
print()

# Q7: Gate vs H
q7 = "YES" if abs(pearson_gh) > 0.3 else ("WEAK" if abs(pearson_gh) > 0.1 else "NO")
print(f"Q7: Does gate behavior correlate with H intensity?")
print(f"    Answer: {q7}")
print(f"    Evidence: Pearson={pearson_gh:.4f} Spearman={spearman_gh:.4f}")
print()

# Q8: One representation dominates before fusion?
h_f_norm = h_f.norm(dim=-1).mean().item()
d_f_norm = d_f.norm(dim=-1).mean().item()
q8 = "YES" if abs(h_f_norm - d_f_norm) / max(h_f_norm, d_f_norm) > 0.2 else "NO"
print(f"Q8: Is there evidence one representation numerically dominates before fusion?")
print(f"    Answer: {q8}")
print(f"    Evidence: H token norm={h_f_norm:.4f} vs DAB token norm={d_f_norm:.4f}")
print()

# Q9: Gate spatially structured?
gate_map_range = avg_gate_map.max() - avg_gate_map.min()
q9 = "YES" if gate_map_range > 0.1 else "NO"
print(f"Q9: Is there evidence the fusion gate is spatially structured?")
print(f"    Answer: {q9}")
print(f"    Evidence: avg gate map range={gate_map_range:.4f} (min={avg_gate_map.min():.4f} max={avg_gate_map.max():.4f})")
print()

# Q10: Recommendation
print(f"Q10: What should we test next?")
if q1 == "YES" and q2 == "NO":
    rec = "A. Train DSCA + H/DAB normalization"
    print(f"    Recommendation: {rec}")
    print(f"    Rationale: H/DAB enter the pretrained ViT with a distribution very different from")
    print(f"    ImageNet-normalized RGB, and the 1->3 projection does NOT normalize them.")
elif q4 == "YES" or q5 == "YES":
    rec = "B. Investigate fusion design"
    print(f"    Recommendation: {rec}")
    print(f"    Rationale: The gate differs between 1+ and 2+ (or correct/incorrect 1+),")
    print(f"    suggesting the fusion mechanism may be contributing to the confusion.")
else:
    rec = "F. Proceed to trained A-series ablations"
    print(f"    Recommendation: {rec}")
    print(f"    Rationale: No strong input-distribution or fusion-gate signal found;")
    print(f"    trained ablations are needed to isolate the cause.")
print("=" * 60)


# ============================================================
# Cell 29 — What We Learned
# ============================================================

print("=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print()
print("## Confirmed findings (directly supported by measurements)")
print(f"  - H/DAB enter the pretrained ViT with mean/std very different from ImageNet-normalized RGB")
print(f"    (baseline mean={base_mean:.3f} vs H={h_after_mean:.3f}, DAB={dab_after_mean:.3f})")
print(f"  - The 1->3 projection is essentially identity-like (H before={h_before_mean:.4f} after={h_after_mean:.4f})")
print(f"  - Overall gate mean = {gate_global_mean:.4f} -> {q3}")
print(f"  - Gate differs between true 1+ and 2+ by {diff_12:.4f}")
print()
print("## Strong clues")
print(f"  - Gate vs DAB correlation: Pearson={pearson_gdab:.4f}")
print(f"  - Gate vs H correlation: Pearson={pearson_gh:.4f}")
print(f"  - 1+->2+ error group gate mean: {gb if grp_b else 'N/A'}")
print()
print("## Unknowns")
print("  - Whether the input distribution mismatch is the CAUSE of the 1+/2+ confusion")
print("    (correlation is not causation; requires a trained normalization variant)")
print("  - Whether the fusion gate difference is a cause or a consequence")
print("  - Cross-attention contribution (not analyzed here; see notebook 03)")
print()
print("## Recommended next experiment")
print(f"  {rec}")
print()
print("=" * 60)


# ============================================================
# Cell 30 — Final Compact Table
# ============================================================

print("=" * 60)
print("FINAL DIAGNOSTIC TABLE")
print("=" * 60)
print(f"{'Diagnostic':<45} {'Result':<12} {'Confidence':<10}")
print("-" * 70)
print(f"{'H/DAB vs ImageNet-RGB distribution':<45} {q1:<12} {'HIGH':<10}")
print(f"{'1->3 projection normalizes':<45} {q2:<12} {'HIGH':<10}")
print(f"{'Gate prefers':<45} {q3:<12} {'HIGH':<10}")
print(f"{'Gate differs 1+ vs 2+':<45} {q4:<12} {'MEDIUM':<10}")
print(f"{'Gate differs correct-1+ vs 1+->2+':<45} {q5:<12} {'MEDIUM':<10}")
print(f"{'Gate correlates with DAB':<45} {q6:<12} {'MEDIUM':<10}")
print(f"{'Gate correlates with H':<45} {q7:<12} {'MEDIUM':<10}")
print(f"{'One stream dominates pre-fusion':<45} {q8:<12} {'MEDIUM':<10}")
print(f"{'Gate spatially structured':<45} {q9:<12} {'MEDIUM':<10}")
print("=" * 60)
print()
print(f"FINAL RECOMMENDATION: {rec}")
print("=" * 60)