# DSCA-ViT v3 Training Notebook
# ============================================================
# This file is written as a Python script with cell markers.
# Paste each section into a Google Colab cell.
#
# Implements the locked DSCA-ViT v3 specification:
#   - models_v3/ independent implementation
#   - multi-scale (fine + low-frequency coarse) with shared modules
#   - stain-domain augmentation (training-only, in the dataset pipeline)
#   - 3-stage training (single persistent AdamW, per-stage cosine)
#   - deterministic stratified validation holdout
#   - one final test evaluation
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

subprocess.run(
    ["pip", "install", "timm", "pyyaml", "scipy", "seaborn", "--quiet"],
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
# str(param.device) for GPU tensors in the device check of Cell 7.
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
# Same proven download+extract logic as the v2 training notebook.

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
# Cell 5 — Stratified Validation Split + Stain Augmentation
# ============================================================
# Deterministic stratified 10% holdout from the official train split.
# Save exact indices for reproducibility.
#
# The training-only StainAugmentation is added to the TRAINING
# transform pipeline (NOT inside the model). Validation and test
# use the non-augmented test transform, so train/eval separation
# is explicit and auditable.

from sklearn.model_selection import train_test_split
from torchvision import transforms
from datasets import HER2Dataset, get_train_transform, get_test_transform
from torch.utils.data import DataLoader, Subset

from models_v3.stain_augmentation import StainAugmentation

# Load config
with open(os.path.join(REPO_DIR, "configs", "dsca_v3_config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

IMAGE_SIZE = CONFIG["dataset"]["image_size"]
VAL_FRACTION = CONFIG["dataset"]["val_fraction"]
VAL_SEED = CONFIG["dataset"]["val_seed"]
BATCH_SIZE = CONFIG["training"]["batch_size"]

# Base transforms (same as v2)
base_train_transform = get_train_transform(image_size=IMAGE_SIZE)
test_transform = get_test_transform(image_size=IMAGE_SIZE)

# Append the training-only stain-domain augmentation AFTER ToTensor
# (StainAugmentation operates on a [3,H,W] tensor in [0,1]).
if CONFIG["stain_augmentation"]["enabled"]:
    sa = CONFIG["stain_augmentation"]
    train_transform = transforms.Compose(
        list(base_train_transform.transforms)
        + [
            StainAugmentation(
                probability=sa["probability"],
                h_concentration_range=tuple(sa["concentration_range"]),
                dab_concentration_range=tuple(sa["concentration_range"]),
                brightness_range=tuple(sa["brightness_range"]),
                contrast_range=tuple(sa["contrast_range"]),
            )
        ]
    )
    print("✅ StainAugmentation enabled in the TRAINING transform only.")
else:
    train_transform = base_train_transform
    print("⚠️  StainAugmentation disabled (config).")

# IMPORTANT: the validation subset must use the NON-augmented test
# transform. Using the train transform (with augmentation) for
# validation would corrupt the validation accuracy.
full_train_dataset = HER2Dataset(root_dir=TRAIN_DIR, transform=train_transform)
full_train_dataset_val = HER2Dataset(root_dir=TRAIN_DIR, transform=test_transform)
test_dataset = HER2Dataset(root_dir=TEST_DIR, transform=test_transform)

# Stratified split
all_labels = full_train_dataset.labels
train_idx, val_idx = train_test_split(
    np.arange(len(full_train_dataset)),
    test_size=VAL_FRACTION,
    random_state=VAL_SEED,
    stratify=all_labels,
)

train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset_val, val_idx)

# Save split indices
CHECKPOINT_ROOT = CONFIG["paths"]["checkpoint_root"]
EXPERIMENT_DIR = os.path.join(CHECKPOINT_ROOT, CONFIG["paths"]["experiment_name"])
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

SPLIT_INDICES_PATH = os.path.join(EXPERIMENT_DIR, "split_indices.npz")
np.savez(
    SPLIT_INDICES_PATH,
    train_idx=train_idx,
    val_idx=val_idx,
    seed=VAL_SEED,
    val_fraction=VAL_FRACTION,
)
print(f"✅ Split indices saved: {SPLIT_INDICES_PATH}")

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
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print("=" * 60)
print("Dataset Split")
print("=" * 60)
print(f"  Full train      : {len(full_train_dataset)}")
print(f"  Train (90%)     : {len(train_dataset)}")
print(f"  Validation (10%): {len(val_dataset)}")
print(f"  Test (official) : {len(test_dataset)}")
print("=" * 60)


# ============================================================
# Cell 6 — Build Model + Parameter Counts
# ============================================================

from models_v3 import DSCAViTv3

model = DSCAViTv3(
    num_classes=CONFIG["model"]["num_classes"],
    pretrained=CONFIG["model"]["pretrained"],
    split_after=CONFIG["model"]["split_after"],
    hidden_channels=CONFIG["model"]["hidden_channels"],
    interaction_hidden_dim=CONFIG["model"]["interaction_hidden_dim"],
    adapter_final_scale=CONFIG["model"]["adapter_final_scale"],
    coarse_size=CONFIG["model"]["coarse_size"],
    spatial_bias_beta=CONFIG["model"]["spatial_bias_beta"],
    spatial_bias_gamma=CONFIG["model"]["spatial_bias_gamma"],
    classifier_dropout=CONFIG["model"]["classifier_dropout"],
)
model = model.to(device)

counts = model.count_parameters()
print("=" * 60)
print("DSCA-ViT v3 Parameter Summary")
print("=" * 60)
for name, count in counts.items():
    print(f"  {name:<22} : {count:>12,}")
print("=" * 60)

# Verify parameter groups (validated against ALL parameters)
groups = model.get_parameter_groups()
print("Parameter groups (5):")
for name, params in groups.items():
    n_params = sum(p.numel() for p in params)
    print(f"  {name:<16} : {len(params):>5} tensors, {n_params:>12,} params")
print("✅ Parameter-group validation passed (no duplicates, all assigned).")

# Verify exactly one instance of each shared module
model.assert_single_shared_instances()
print("✅ Single-instance verification passed (one ViT, one cross-attn, "
      "one interaction, one stain gate, one scale gate).")


# ============================================================
# Cell 7 — Forward / Shape / Backward Sanity Check
# ============================================================

model.eval()

# Small batch for shape check.
# NOTE: use torch.rand (uniform [0,1]) NOT torch.randn — the color
# deconvolution computes -log10(x + eps), which produces NaN for
# negative inputs (randn can be negative).
x_check = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

with torch.no_grad():
    h_ch, d_ch = model.color_deconv(x_check)
    print(f"RGB                : {tuple(x_check.shape)}")
    print(f"H                  : {tuple(h_ch.shape)}")
    print(f"DAB                : {tuple(d_ch.shape)}")

    h = model.norm_h(h_ch)
    d = model.norm_dab(d_ch)
    h = model.adapter_h(h)
    d = model.adapter_dab(d)
    h = model.channel_affine_h(h)
    d = model.channel_affine_dab(d)
    print(f"H after adapter    : {tuple(h.shape)}")
    print(f"DAB after adapter  : {tuple(d.shape)}")

    h_coarse = model.coarse(h)
    d_coarse = model.coarse(d)
    print(f"H coarse view      : {tuple(h_coarse.shape)}")
    print(f"DAB coarse view    : {tuple(d_coarse.shape)}")

    h_fine_t = model.encoder.embed(h)
    d_fine_t = model.encoder.embed(d)
    h_coarse_t = model.encoder.embed(h_coarse)
    d_coarse_t = model.encoder.embed(d_coarse)
    print(f"Fine H tokens      : {tuple(h_fine_t.shape)}")
    print(f"Fine DAB tokens    : {tuple(d_fine_t.shape)}")
    print(f"Coarse H tokens    : {tuple(h_coarse_t.shape)}")
    print(f"Coarse DAB tokens  : {tuple(d_coarse_t.shape)}")

    f_fine = model._process_scale(h_fine_t, d_fine_t)
    f_coarse = model._process_scale(h_coarse_t, d_coarse_t)
    print(f"F_fine (stain gate): {tuple(f_fine.shape)}")
    print(f"F_coarse (stain g.) : {tuple(f_coarse.shape)}")

    fused, scale_gate = model.scale_gate(f_fine, f_coarse)
    print(f"Scale gate         : {tuple(scale_gate.shape)}")
    print(f"Fused (scale gate) : {tuple(fused.shape)}")

    refined = model.refinement(fused)
    logits = model.classifier(refined)
    print(f"Logits             : {tuple(logits.shape)}")

# Backward check
model.train()
x_check.requires_grad_(True)
logits = model(x_check)
loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 4, (2,)).to(device))
loss.backward()
print(f"loss.backward()     : OK (loss={loss.item():.4f})")

# NaN / Inf / dtype / device checks
for name, p in model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        raise RuntimeError(f"NaN/Inf in parameter: {name}")
    if p.dtype != torch.float32:
        raise RuntimeError(f"Unexpected dtype for {name}: {p.dtype}")
    # Compare as strings: torch.device('cuda') != torch.device('cuda:0')
    # in some PyTorch versions, so use str().
    if str(p.device) != str(device):
        raise RuntimeError(f"Unexpected device for {name}: {p.device}")

print("✅ Forward/backward sanity check passed (shapes, NaN/Inf, dtype, device).")


# ============================================================
# Cell 8 — V2 Compatibility Loading
# ============================================================
# Loads preserved weights (encoder, cross_attention, refinement,
# classifier) from the v2 best_stage3 checkpoint. New v3 modules
# (norm/adapter/affine/coarse/interaction/stain_gate/scale_gate)
# start FRESH.

V2_CKPT = CONFIG["paths"]["v2_checkpoint"]
assert os.path.exists(V2_CKPT), f"V2 checkpoint not found: {V2_CKPT}"

load_report = model.load_v2_weights(V2_CKPT, device)
print("✅ V2 preserved weights loaded into v3 (missing preserved = NONE).")


# ============================================================
# Cell 9 — Initialization Verification
# ============================================================
# Verify the locked initialization guarantees BEFORE training:
#   - interaction residuals delta_H = 0, delta_D = 0
#   - stain gate ~= 0.5
#   - scale gate ~= 0.5
#   - fine and coarse representations have the same shape and are
#     NOT exactly identical (no minimum-difference threshold imposed)

model.eval()
with torch.no_grad():
    x_init = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    h_ch, d_ch = model.color_deconv(x_init)
    h = model.channel_affine_h(model.adapter_h(model.norm_h(h_ch)))
    d = model.channel_affine_dab(model.adapter_dab(model.norm_dab(d_ch)))

    h_coarse = model.coarse(h)
    d_coarse = model.coarse(d)

    h_fine_t = model.encoder.embed(h)
    d_fine_t = model.encoder.embed(d)
    h_coarse_t = model.encoder.embed(h_coarse)
    d_coarse_t = model.encoder.embed(d_coarse)

    # Interaction residuals (zero-init final linear -> delta = 0)
    bs = h_fine_t.shape[0]
    stacked = torch.cat([h_fine_t, d_fine_t], dim=0)
    stacked = model.encoder.forward_before(stacked)
    h_t, d_t = stacked.split(bs, dim=0)
    h_t, d_t = model.cross_attention(h_t, d_t)
    stacked = torch.cat([h_t, d_t], dim=0)
    stacked = model.encoder.forward_after(stacked)
    h_f, d_f = stacked.split(bs, dim=0)
    delta_h = model.interaction.interaction_d_to_h(torch.cat([h_f, d_f], dim=-1))
    delta_d = model.interaction.interaction_h_to_d(torch.cat([d_f, h_f], dim=-1))

    print("=" * 60)
    print("INITIALIZATION VERIFICATION")
    print("=" * 60)
    print(f"  mean(|delta_H|) : {delta_h.abs().mean().item():.6f}  (expect ~0)")
    print(f"  mean(|delta_D|) : {delta_d.abs().mean().item():.6f}  (expect ~0)")

    # Stain gate ~= 0.5 (from the last _process_scale call)
    stain_gate = model.get_stain_gate_values()
    if stain_gate is not None:
        sg = stain_gate.detach().cpu().numpy()
        print(f"  stain gate mean : {sg.mean():.4f}  (expect ~0.5)")

    # Scale gate ~= 0.5
    scale_gate = model.get_scale_gate_values()
    if scale_gate is not None:
        scg = scale_gate.detach().cpu().numpy()
        print(f"  scale gate mean : {scg.mean():.4f}  (expect ~0.5)")

    # Fine vs coarse: same shape, not exactly identical
    f_fine = model._process_scale(h_fine_t, d_fine_t)
    f_coarse = model._process_scale(h_coarse_t, d_coarse_t)
    print(f"  F_fine shape    : {tuple(f_fine.shape)}")
    print(f"  F_coarse shape  : {tuple(f_coarse.shape)}")
    diff = (f_fine - f_coarse).abs().max().item()
    print(f"  max|F_fine-F_coarse| : {diff:.6f}  (expect > 0, no min threshold)")
    assert f_fine.shape == f_coarse.shape == (2, 197, 768)
    assert diff > 0.0, "Fine and coarse representations must not be exactly identical."
    print("=" * 60)

print("✅ Initialization verification passed.")


# ============================================================
# Cell 10 — Smoke Test
# ============================================================
# Short training smoke test (1-2 batches) before the full run.

from utils.train_v3 import (
    train_one_epoch_v3,
    validate_one_epoch_v3,
    set_stage_requires_grad,
    set_stage_lrs,
    snapshot_initial_params_v3,
    collect_telemetry_v3,
    save_stage_checkpoint,
)

criterion = nn.CrossEntropyLoss()

# Snapshot initial params of the new modules BEFORE any training,
# so telemetry can report meaningful parameter deltas at the end.
initial_state = snapshot_initial_params_v3(model)

# Build the single persistent optimizer with 5 groups
groups = model.get_parameter_groups()

def make_param_groups(groups, weight_decay=0.05):
    """Weight decay on weights only; bias/norm params -> weight_decay=0.

    Each optimizer param group carries a "name" key identifying its
    architecture group (vit / existing_dsca / input_modules /
    fusion_modules / classifier). set_stage_lrs matches by name, so
    the 10 decay/no-decay sub-groups are handled correctly.
    """
    param_groups = []
    for name, params in groups.items():
        decay_params = []
        no_decay_params = []
        for p in params:
            if p.ndim <= 1:  # bias / norm / scale / affine params
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups.append({
            "name": name,
            "params": decay_params,
            "weight_decay": weight_decay,
        })
        param_groups.append({
            "name": name,
            "params": no_decay_params,
            "weight_decay": 0.0,
        })
    return param_groups

optimizer = optim.AdamW(
    make_param_groups(groups, CONFIG["training"]["weight_decay"]),
    lr=1e-4,
)

# Smoke test: Stage 1 freeze config, 1 batch
set_stage_requires_grad(model, stage=1)
set_stage_lrs(optimizer, stage=1, stage_config=CONFIG["stage1"])

smoke_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

for i, (images, labels) in enumerate(smoke_loader):
    if i >= 1:
        break
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    params_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=CONFIG["training"]["gradient_clip"])
    optimizer.step()
    print(f"Smoke test batch: loss={loss.item():.4f}")

print("✅ Smoke test passed.")


# ============================================================
# Cell 11 — Stage 1 (Input + Fusion Adaptation)
# ============================================================

STAGE1_EPOCHS = CONFIG["stage1"]["epochs"]
STAGE1_CKPT = os.path.join(EXPERIMENT_DIR, "stage1_end.pt")

set_stage_requires_grad(model, stage=1)
set_stage_lrs(optimizer, stage=1, stage_config=CONFIG["stage1"])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

best_acc = 0.0
best_epoch = 0

print("=" * 60)
print("Stage 1 — Input + Fusion Adaptation")
print(f"  input_modules : {CONFIG['stage1']['input_lr']}")
print(f"  fusion_modules: {CONFIG['stage1']['fusion_lr']}")
print(f"  classifier    : {CONFIG['stage1']['classifier_lr']}")
print(f"  epochs        : {STAGE1_EPOCHS}")
print("=" * 60)

for epoch in range(STAGE1_EPOCHS):
    train_loss, train_acc = train_one_epoch_v3(
        model, train_loader, criterion, optimizer, device,
        gradient_clip=CONFIG["training"]["gradient_clip"],
    )
    val_loss, val_acc, preds, labels = validate_one_epoch_v3(
        model, val_loader, criterion, device,
    )
    scheduler.step()

    # Generalization monitoring: per-class recall at every validation epoch
    from utils.metrics_v3 import compute_metrics_v3
    val_metrics = compute_metrics_v3(np.array(labels), np.array(preds), full_train_dataset.get_class_names())
    recall_str = " ".join(
        f"{cls}={val_metrics['per_class_recall'][i]:.3f}"
        for i, cls in enumerate(val_metrics["class_names"])
    )

    print(
        f"Epoch [{epoch+1:02d}/{STAGE1_EPOCHS}] | "
        f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
        f"BalAcc {val_metrics['balanced_accuracy']:.2f}% | "
        f"MacroF1 {val_metrics['macro_f1']:.4f} | "
        f"Recall [{recall_str}]"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1

save_stage_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=best_epoch,
    stage=1,
    val_acc=best_acc,
    save_path=STAGE1_CKPT,
    config=CONFIG,
    seed=SEED,
    split_indices_path=SPLIT_INDICES_PATH,
)
print(f"✅ Stage 1 checkpoint saved: {STAGE1_CKPT} (best val acc: {best_acc:.2f}%)")


# ============================================================
# Cell 12 — Stage 2 (Existing DSCA Adaptation)
# ============================================================

STAGE2_EPOCHS = CONFIG["stage2"]["epochs"]
STAGE2_CKPT = os.path.join(EXPERIMENT_DIR, "stage2_end.pt")

# Transition: set requires_grad, then LRs, then new cosine scheduler
set_stage_requires_grad(model, stage=2)
set_stage_lrs(optimizer, stage=2, stage_config=CONFIG["stage2"])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

best_acc = 0.0
best_epoch = 0

print("=" * 60)
print("Stage 2 — Existing DSCA Adaptation")
print(f"  input_modules : {CONFIG['stage2']['input_lr']}")
print(f"  existing_dsca : {CONFIG['stage2']['existing_lr']}")
print(f"  fusion_modules: {CONFIG['stage2']['fusion_lr']}")
print(f"  classifier    : {CONFIG['stage2']['classifier_lr']}")
print(f"  epochs        : {STAGE2_EPOCHS}")
print("=" * 60)

for epoch in range(STAGE2_EPOCHS):
    train_loss, train_acc = train_one_epoch_v3(
        model, train_loader, criterion, optimizer, device,
        gradient_clip=CONFIG["training"]["gradient_clip"],
    )
    val_loss, val_acc, preds, labels = validate_one_epoch_v3(
        model, val_loader, criterion, device,
    )
    scheduler.step()

    from utils.metrics_v3 import compute_metrics_v3
    val_metrics = compute_metrics_v3(np.array(labels), np.array(preds), full_train_dataset.get_class_names())
    recall_str = " ".join(
        f"{cls}={val_metrics['per_class_recall'][i]:.3f}"
        for i, cls in enumerate(val_metrics["class_names"])
    )

    print(
        f"Epoch [{epoch+1:02d}/{STAGE2_EPOCHS}] | "
        f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
        f"BalAcc {val_metrics['balanced_accuracy']:.2f}% | "
        f"MacroF1 {val_metrics['macro_f1']:.4f} | "
        f"Recall [{recall_str}]"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1

save_stage_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=best_epoch,
    stage=2,
    val_acc=best_acc,
    save_path=STAGE2_CKPT,
    config=CONFIG,
    seed=SEED,
    split_indices_path=SPLIT_INDICES_PATH,
)
print(f"✅ Stage 2 checkpoint saved: {STAGE2_CKPT} (best val acc: {best_acc:.2f}%)")


# ============================================================
# Cell 13 — Stage 3 (Joint Optimization)
# ============================================================

STAGE3_EPOCHS = CONFIG["stage3"]["epochs"]
BEST_S3_CKPT = os.path.join(EXPERIMENT_DIR, "best_stage3.pt")
LAST_CKPT = os.path.join(EXPERIMENT_DIR, "last.pt")

# Transition: set requires_grad, then LRs, then new cosine scheduler
set_stage_requires_grad(model, stage=3)
set_stage_lrs(optimizer, stage=3, stage_config=CONFIG["stage3"])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE3_EPOCHS)

best_acc = 0.0
best_epoch = 0

print("=" * 60)
print("Stage 3 — Joint Optimization")
print(f"  vit           : {CONFIG['stage3']['vit_lr']}")
print(f"  existing_dsca : {CONFIG['stage3']['existing_lr']}")
print(f"  input_modules : {CONFIG['stage3']['input_lr']}")
print(f"  fusion_modules: {CONFIG['stage3']['fusion_lr']}")
print(f"  classifier    : {CONFIG['stage3']['classifier_lr']}")
print(f"  epochs        : {STAGE3_EPOCHS}")
print("=" * 60)

for epoch in range(STAGE3_EPOCHS):
    train_loss, train_acc = train_one_epoch_v3(
        model, train_loader, criterion, optimizer, device,
        gradient_clip=CONFIG["training"]["gradient_clip"],
    )
    val_loss, val_acc, preds, labels = validate_one_epoch_v3(
        model, val_loader, criterion, device,
    )
    scheduler.step()

    from utils.metrics_v3 import compute_metrics_v3
    val_metrics = compute_metrics_v3(np.array(labels), np.array(preds), full_train_dataset.get_class_names())
    recall_str = " ".join(
        f"{cls}={val_metrics['per_class_recall'][i]:.3f}"
        for i, cls in enumerate(val_metrics["class_names"])
    )

    print(
        f"Epoch [{epoch+1:02d}/{STAGE3_EPOCHS}] | "
        f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
        f"BalAcc {val_metrics['balanced_accuracy']:.2f}% | "
        f"MacroF1 {val_metrics['macro_f1']:.4f} | "
        f"Recall [{recall_str}]"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        save_stage_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=best_epoch,
            stage=3,
            val_acc=best_acc,
            save_path=BEST_S3_CKPT,
            config=CONFIG,
            seed=SEED,
            split_indices_path=SPLIT_INDICES_PATH,
        )
        print(f"  ✅ New best model saved (Epoch {best_epoch} | Val Acc: {best_acc:.2f}%)")

save_stage_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=STAGE3_EPOCHS,
    stage=3,
    val_acc=best_acc,
    save_path=LAST_CKPT,
    config=CONFIG,
    seed=SEED,
    split_indices_path=SPLIT_INDICES_PATH,
)
print(f"✅ Stage 3 finished. Best val acc: {best_acc:.2f}% @ epoch {best_epoch}")
print(f"   Best checkpoint : {BEST_S3_CKPT}")
print(f"   Last checkpoint : {LAST_CKPT}")


# ============================================================
# Cell 13b — Load Best Stage 3 Checkpoint
# ============================================================
# Loads the best Stage 3 checkpoint (selected on validation) into
# the model for evaluation. Self-contained: works right after
# Cell 13, or in a fresh session (rebuilds model from CONFIG).

# Self-contained imports (in case this cell is run in a fresh session)
import os
import torch
import yaml

from models_v3 import DSCAViTv3
from utils.checkpoint import load_checkpoint

if "REPO_DIR" not in globals():
    REPO_DIR = "/content/DSCA-ViT"

if "device" not in globals():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Rebuild CONFIG / paths if not already defined in this session
if "CONFIG" not in globals():
    with open(os.path.join(REPO_DIR, "configs", "dsca_v3_config.yaml")) as f:
        CONFIG = yaml.safe_load(f)

if "EXPERIMENT_DIR" not in globals():
    CHECKPOINT_ROOT = CONFIG["paths"]["checkpoint_root"]
    EXPERIMENT_DIR = os.path.join(CHECKPOINT_ROOT, CONFIG["paths"]["experiment_name"])

BEST_S3_CKPT = os.path.join(EXPERIMENT_DIR, "best_stage3.pt")

# Rebuild model if not already defined in this session
if "model" not in globals():
    model = DSCAViTv3(
        num_classes=CONFIG["model"]["num_classes"],
        pretrained=CONFIG["model"]["pretrained"],
        split_after=CONFIG["model"]["split_after"],
        hidden_channels=CONFIG["model"]["hidden_channels"],
        interaction_hidden_dim=CONFIG["model"]["interaction_hidden_dim"],
        adapter_final_scale=CONFIG["model"]["adapter_final_scale"],
        coarse_size=CONFIG["model"]["coarse_size"],
        spatial_bias_beta=CONFIG["model"]["spatial_bias_beta"],
        spatial_bias_gamma=CONFIG["model"]["spatial_bias_gamma"],
        classifier_dropout=CONFIG["model"]["classifier_dropout"],
    )
    model = model.to(device)

assert os.path.exists(BEST_S3_CKPT), f"Best Stage 3 checkpoint not found: {BEST_S3_CKPT}"

ckpt = load_checkpoint(path=BEST_S3_CKPT, model=model, device=device)
model.eval()

best_val = ckpt.get("best_val_accuracy", "N/A")
best_val_str = f"{best_val:.2f}%" if isinstance(best_val, (int, float)) else str(best_val)

print("=" * 60)
print("Best Stage 3 Checkpoint Loaded")
print("=" * 60)
print(f"  Checkpoint    : {BEST_S3_CKPT}")
print(f"  Stage         : {ckpt.get('stage', 'N/A')}")
print(f"  Epoch         : {ckpt.get('epoch', 'N/A')}")
print(f"  Best val acc  : {best_val_str}")
print("=" * 60)


# ============================================================
# Cell 14 — Validation Metrics
# ============================================================

from utils.metrics_v3 import compute_metrics_v3, print_metrics_v3

# Load best Stage 3 checkpoint
from utils.checkpoint import load_checkpoint
load_checkpoint(path=BEST_S3_CKPT, model=model, device=device)
model.eval()

val_loss, val_acc, val_preds, val_labels = validate_one_epoch_v3(
    model, val_loader, criterion, device,
)

class_names = full_train_dataset.get_class_names()
val_metrics = compute_metrics_v3(np.array(val_labels), np.array(val_preds), class_names)
print_metrics_v3(val_metrics, title="VALIDATION METRICS (v3)")


# ============================================================
# Cell 15 — One-Time Test Evaluation
# ============================================================

test_loss, test_acc, test_preds, test_labels = validate_one_epoch_v3(
    model, test_loader, criterion, device,
)

test_metrics = compute_metrics_v3(np.array(test_labels), np.array(test_preds), class_names)
print("=" * 60)
print("FINAL TEST EVALUATION (official test split, evaluated once)")
print("=" * 60)
print_metrics_v3(test_metrics, title="TEST METRICS (v3)")


# ============================================================
# Cell 16 — Baseline Comparison
# ============================================================

print("=" * 60)
print("BASELINE COMPARISON (official test split)")
print("=" * 60)
print(f"{'Model':<20} {'Accuracy':>10} {'Balanced Acc':>14} {'Macro-F1':>10}")
print("-" * 60)
print(f"{'ViT baseline':<20} {'95.02%':>10} {'—':>14} {'—':>10}")
print(f"{'Original DSCA':<20} {'~92.26%':>10} {'—':>14} {'—':>10}")
print(f"{'DSCA-ViT v2':<20} {'87.22%':>10} {'—':>14} {'—':>10}")
print(
    f"{'DSCA-ViT v3':<20} "
    f"{test_metrics['accuracy']:>9.2f}% "
    f"{test_metrics['balanced_accuracy']:>13.2f}% "
    f"{test_metrics['macro_f1']:>10.4f}"
)
print("-" * 60)
print("\nPer-class recall (v3):")
for i, cls in enumerate(class_names):
    print(f"  {cls}: recall={test_metrics['per_class_recall'][i]:.4f}")
print("=" * 60)

# Validation -> test gap (the key generalization metric)
gap = val_metrics["accuracy"] - test_metrics["accuracy"]
print("=" * 60)
print("GENERALIZATION GAP (validation - test)")
print("=" * 60)
print(f"  Validation acc : {val_metrics['accuracy']:.2f}%")
print(f"  Test acc       : {test_metrics['accuracy']:.2f}%")
print(f"  Gap            : {gap:+.2f} pp  (v2 was -12.04 pp)")
print("=" * 60)


# ============================================================
# Cell 17 — Telemetry
# ============================================================
# Lightweight telemetry: confirm the new modules actually learned.
# Uses the `initial_state` snapshot captured in Cell 10 (before training).

# Run one validation batch to collect interaction output norms + gate stats
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        h_ch, d_ch = model.color_deconv(images)
        h = model.channel_affine_h(model.adapter_h(model.norm_h(h_ch)))
        d = model.channel_affine_dab(model.adapter_dab(model.norm_dab(d_ch)))
        h_coarse = model.coarse(h)
        d_coarse = model.coarse(d)
        h_fine_t = model.encoder.embed(h)
        d_fine_t = model.encoder.embed(d)
        h_coarse_t = model.encoder.embed(h_coarse)
        d_coarse_t = model.encoder.embed(d_coarse)

        # Interaction residuals (fine branch)
        bs = h_fine_t.shape[0]
        stacked = torch.cat([h_fine_t, d_fine_t], dim=0)
        stacked = model.encoder.forward_before(stacked)
        h_t, d_t = stacked.split(bs, dim=0)
        h_t, d_t = model.cross_attention(h_t, d_t)
        stacked = torch.cat([h_t, d_t], dim=0)
        stacked = model.encoder.forward_after(stacked)
        h_f, d_f = stacked.split(bs, dim=0)
        delta_h = model.interaction.interaction_d_to_h(torch.cat([h_f, d_f], dim=-1))
        delta_d = model.interaction.interaction_h_to_d(torch.cat([d_f, h_f], dim=-1))

        # Full forward to capture stain + scale gates
        logits = model(images)
        stain_gate = model.get_stain_gate_values()
        scale_gate = model.get_scale_gate_values()
        break

telemetry = collect_telemetry_v3(
    model,
    initial_state=initial_state,
    delta_h=delta_h,
    delta_d=delta_d,
    stain_gate=stain_gate,
    scale_gate=scale_gate,
)

print("=" * 60)
print("TELEMETRY (new v3 modules)")
print("=" * 60)
for name, stats in telemetry.items():
    if isinstance(stats, dict):
        print(f"  {name:<28} grad_norm={stats['grad_norm']:.4f} "
              f"param_delta={stats['parameter_delta']:.4f} "
              f"rel_delta={stats['relative_delta']:.4f}")
    else:
        print(f"  {name:<28} {stats:.6f}")
print("=" * 60)

# Gate statistics
if stain_gate is not None:
    sg = stain_gate.cpu().numpy()
    print("=" * 60)
    print("STAIN GATE STATISTICS (validation batch)")
    print("=" * 60)
    print(f"  mean   : {sg.mean():.4f}")
    print(f"  std    : {sg.std():.4f}")
    print(f"  min    : {sg.min():.4f}")
    print(f"  max    : {sg.max():.4f}")
    print(f"  median : {np.median(sg):.4f}")
    print("=" * 60)

if scale_gate is not None:
    scg = scale_gate.cpu().numpy()
    print("=" * 60)
    print("SCALE GATE STATISTICS (validation batch)")
    print("=" * 60)
    print(f"  mean   : {scg.mean():.4f}")
    print(f"  std    : {scg.std():.4f}")
    print(f"  min    : {scg.min():.4f}")
    print(f"  max    : {scg.max():.4f}")
    print(f"  median : {np.median(scg):.4f}")
    print("=" * 60)

# Gate/confidence correlations
probs = torch.softmax(logits, dim=1)
confidence = probs.max(dim=1).values.cpu().numpy()

if stain_gate is not None:
    sample_stain_gate = stain_gate.mean(axis=(1, 2)).cpu().numpy()  # [B]
    pearson_sg, _ = pearsonr(sample_stain_gate, confidence)
    spearman_sg, _ = spearmanr(sample_stain_gate, confidence)
    print("=" * 60)
    print("STAIN GATE / CONFIDENCE CORRELATION")
    print("=" * 60)
    print(f"  Pearson  : {pearson_sg:.4f}")
    print(f"  Spearman : {spearman_sg:.4f}")
    print("=" * 60)

if scale_gate is not None:
    sample_scale_gate = scale_gate.mean(axis=(1, 2)).cpu().numpy()  # [B]
    pearson_scg, _ = pearsonr(sample_scale_gate, confidence)
    spearman_scg, _ = spearmanr(sample_scale_gate, confidence)
    print("=" * 60)
    print("SCALE GATE / CONFIDENCE CORRELATION")
    print("=" * 60)
    print(f"  Pearson  : {pearson_scg:.4f}")
    print(f"  Spearman : {spearman_scg:.4f}")
    print("=" * 60)

print("\n✅ DSCA-ViT v3 training complete.")


# ============================================================
# Cell 18 — Final Report
# ============================================================

print("=" * 60)
print("DSCA-ViT v3 FINAL REPORT")
print("=" * 60)
print(f"  Validation accuracy : {val_metrics['accuracy']:.2f}%")
print(f"  Test accuracy       : {test_metrics['accuracy']:.2f}%")
print(f"  Generalization gap  : {gap:+.2f} pp")
print()
print("  Baseline comparison (official test):")
print(f"    ViT baseline       : 95.02%")
print(f"    Original DSCA      : ~92.26%")
print(f"    DSCA-ViT v2        : 87.22%")
print(f"    DSCA-ViT v3        : {test_metrics['accuracy']:.2f}%")
print()
print("  Success criteria:")
print("    - Increase official test accuracy vs v2 (87.22%)")
print("    - Reduce the validation -> test gap vs v2 (-12.04 pp)")
print("=" * 60)