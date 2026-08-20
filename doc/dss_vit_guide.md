# DSS-ViT — Complete Implementation & Usage Guide

> **Dual-Stream Stain Vision Transformer for HER2 IHC Scoring**
> RGB main stream (ViT-B16) + H/DAB stain auxiliary branch + cross-attention fusion + ordinal head.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Context & Motivation](#2-project-context--motivation)
3. [Architecture](#3-architecture)
4. [File Structure](#4-file-structure)
5. [Configuration Reference](#5-configuration-reference)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Process](#7-training-process)
8. [HPC Usage (Toubkal / UM6P)](#8-hpc-usage-toubkal--um6p)
9. [Evaluation](#9-evaluation)
10. [Checkpointing & Resume](#10-checkpointing--resume)
11. [DDP (Multi-GPU) Details](#11-ddp-multi-gpu-details)
12. [AMP (Mixed Precision)](#12-amp-mixed-precision)
13. [Sanity Checks & Debugging](#13-sanity-checks--debugging)
14. [Troubleshooting](#14-troubleshooting)
15. [Results Reporting](#15-results-reporting)

---

## 1. Overview

**DSS-ViT** is a deep learning model for HER2 Immunohistochemistry (IHC) patch classification. It classifies tissue patches into 4 ordinal classes: `{0, 1+, 2+, 3+}`.

The model combines:

| Stream | Description |
|---|---|
| **RGB main stream** | A pretrained ViT-B16 (ImageNet) processes the raw RGB image (normalized internally with ImageNet mean/std). |
| **Stain auxiliary stream** | Fixed Ruifrok color deconvolution separates H (Hematoxylin) and DAB channels; these are normalized with global statistics and encoded by a lightweight CNN (`StainEncoder`) into 16 tokens. |
| **Fusion** | Cross-attention (CLS query × stain tokens) + gated residual combines the two streams. |
| **Head** | An ordinal head predicts 3 cutpoints → class probabilities. |

**Goal**: beat the plain ViT-B16 baseline of **95.02% test accuracy** on the same official test split.

**Key design decisions**:
- Raw RGB `[0,1]` input; all normalization (ImageNet + stain) happens **inside the model**.
- Ordinal loss (BCE on cumulative cutpoints) combined with cross-entropy.
- 3-stage training (55 epochs total) with staged fine-tuning.
- HPC-ready: SLURM scripts, DDP support, AMP, resumable checkpoints.

---

## 2. Project Context & Motivation

### 2.1 The clinical problem

HER2 (Human Epidermal growth factor Receptor 2) status determines whether breast cancer patients are eligible for targeted therapy (e.g., trastuzumab). Pathologists score HER2 IHC slides as:

| Score | Meaning |
|---|---|
| 0 | No staining or incomplete membrane staining in <10% of cells |
| 1+ | Weak, incomplete membrane staining |
| 2+ | Equivocal — needs FISH test |
| 3+ | Strong, complete membrane staining |

### 2.2 Why a dual-stream design?

- **Hematoxylin (H)** stains nuclei → reveals **morphology**.
- **DAB** stains HER2 protein → reveals **membrane HER2 expression**.
- Both stains are pixel-perfectly registered (same tissue section, same scan).
- Color deconvolution (Ruifrok & Johnston) separates them mathematically.

### 2.3 Why DSS-ViT (vs previous versions)?

| Model | Test accuracy | Notes |
|---|---|---|
| Plain ViT-B16 baseline | **95.02%** | The number to beat |
| DSCA-ViT v1 (`models/`) | ~92.26% | Original dual-stain design |
| DSCA-ViT v2 (`models_v2/`) | ~87.22% | Worse than baseline |
| **DSS-ViT (`models_v2_1/`)** | **To be measured** | RGB main + stain auxiliary + ordinal head + longer training |

DSS-ViT keeps RGB as the **main** input (unlike v1/v2 which used stain-only streams) and adds the stain information as an **auxiliary** branch. This is the key difference: the pretrained ViT sees the natural RGB image (with ImageNet normalization) while the stain branch provides complementary biological signal.

---

## 3. Architecture

### 3.1 Overall flow

```
Raw RGB [B,3,H,W] in [0,1]
      │
      ├─────────────────────────────┐
      │                             │
      ▼                             ▼
ColorDeconvolution            Normalize RGB
      │                      (ImageNet mean/std)
      ├──── H [B,1,H,W]           │
      ├──── DAB [B,1,H,W]         ▼
      │                      ViT-B16 (timm)
      ▼                             │
  StainEncoder                    features [B,197,768]
      │                             │
      ▼                             ├── x_cls [B,768]
  StainTokens [B,16,768]             └── x_patch [B,196,768]
      │
      └───────────────┬─────────────┘
                      ▼
              Cross-Attention Fusion
                      │
                      ▼
                 fused_cls [B,768]
                      │
                      ▼
                 Ordinal Head
                      │
                      ▼
                cutpoints [B,3]  →  probs [B,4]
```

### 3.2 Component details

#### 3.2.1 ColorDeconvolution (`models_v2_1/color_deconv.py`)

- Fixed Ruifrok & Johnston H-DAB stain matrix (non-learnable, registered as a buffer).
- Input: RGB `[0,1]`; computes `OD = -log10(x + ε)`; multiplies by inverse stain matrix.
- Returns `(h, dab)` each `[B,1,H,W]`, clamped to ≥ 0.
- Wrapped in `torch.no_grad()` in the model forward — no gradients flow through it.
- Self-contained copy (identical to `models/color_deconv.py` and `models_v2/color_deconv.py`).

#### 3.2.2 RGB Backbone (`models_v2_1/dss_vit.py`)

```python
self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
```

- ImageNet normalization applied **inside** forward:
  ```python
  rgb_norm = (x_rgb - imagenet_mean) / imagenet_std
  features = self.vit.forward_features(rgb_norm)  # [B, 197, 768]
  ```
- `x_cls = features[:, 0]` → `[B, 768]`
- `x_patch = features[:, 1:]` → `[B, 196, 768]` (kept for future use)

#### 3.2.3 Stain normalization

Global H/DAB mean & std (computed once on the 90% training split) are registered as buffers:

```python
h_norm = (h_channel - h_mean) / h_std
dab_norm = (dab_channel - dab_mean) / dab_std
stain_input = torch.cat([h_norm, dab_norm], dim=1)  # [B, 2, H, W]
```

#### 3.2.4 StainEncoder (`models_v2_1/stain_encoder.py`)

A lightweight CNN (bottleneck, ~3–4M params) converting `[B,2,224,224]` → `[B,16,768]`:

```
Conv2d(2, 32, 3, stride=2) → BN → GELU   # [B,32,112,112]
Conv2d(32, 64, 3, stride=2) → BN → GELU  # [B,64,56,56]
Conv2d(64, 128, 3, stride=2) → BN → GELU # [B,128,28,28]
Conv2d(128, 256, 3, stride=2) → BN → GELU # [B,256,14,14]
AdaptiveAvgPool2d((4,4))                  # [B,256,4,4]
Flatten                                   # [B,4096]
Linear(4096, 512) → GELU                  # bottleneck
Linear(512, 16*768)                       # [B, 12288]
Reshape to [B, 16, 768]
```

- `num_tokens=16`, `out_dim=768`, `bottleneck_dim=512`.
- Xavier init for linear layers; BN init to 1/0.

#### 3.2.5 Cross-Attention Fusion (`models_v2_1/dss_vit.py`)

```python
self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)

# Query: x_cls [B,1,768]; Key/Value: stain_tokens [B,16,768]
attn_out, _ = self.cross_attn(query=x_cls.unsqueeze(1), key=stain_tokens, value=stain_tokens)
attn_out = attn_out.squeeze(1)  # [B,768]

# Gate
concat = torch.cat([x_cls, attn_out], dim=-1)  # [B,1536]
gate = torch.sigmoid(self.gate_mlp(concat))     # [B,768]
fused_cls = x_cls + gate * attn_out            # [B,768]
```

- `gate_mlp`: `Linear(1536→768) → GELU → Linear(768→768)`.
- The last linear bias is initialized to 0 → `sigmoid(0) = 0.5` → gate ≈ 0.5 at init.

#### 3.2.6 OrdinalHead (`models_v2_1/ordinal_head.py`)

```python
class OrdinalHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=4):
        self.fc = nn.Linear(in_dim, num_classes - 1)  # 3 cutpoints
```

Cutpoint logits → class probabilities:

```python
cutpoints = torch.sigmoid(logits)                # [B,3]
p0 = 1 - cutpoints[:, 0]
p1 = cutpoints[:, 0] - cutpoints[:, 1]
p2 = cutpoints[:, 1] - cutpoints[:, 2]
p3 = cutpoints[:, 2]
probs = torch.stack([p0, p1, p2, p3], dim=1)     # [B,4]
probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
```

#### 3.2.7 Loss functions (`models_v2_1/ordinal_head.py`)

```python
def ordinal_loss(logits, targets, num_classes=4):
    ord_targets = (targets.unsqueeze(1) > torch.arange(num_classes-1)).float()
    return F.binary_cross_entropy_with_logits(logits, ord_targets, reduction="mean")

def total_loss(logits, targets, probs, alpha=0.1, label_smoothing=0.1):
    ce = F.cross_entropy(probs.log(), targets, label_smoothing=label_smoothing)
    ord = ordinal_loss(logits, targets)
    return ce + alpha * ord, ce, ord
```

- `alpha=0.1` (configurable via `training.ordinal_alpha`).
- `label_smoothing=0.1` (configurable via `training.label_smoothing`).

#### 3.2.8 Model forward return

```python
{
    "logits": logits,        # [B,3] cutpoint logits
    "probs": probs,          # [B,4] class probabilities
    "x_cls": x_cls,          # [B,768] ViT CLS features
    "fused_cls": fused_cls,  # [B,768] fused features
    "mean_gate": gate.mean() # scalar
}
```

#### 3.2.9 Parameter groups

```python
{
    "vit": list(self.vit.parameters()),                    # ~86M
    "stain_encoder": list(self.stain_encoder.parameters()), # ~3-4M
    "cross_fusion_gate": cross_attn + gate_mlp,             # ~2.4M
    "ordinal_head": list(self.ordinal_head.parameters()),   # ~2.3K
}
```

---

## 4. File Structure

```
models_v2_1/                    # Model package (self-contained)
├── __init__.py                 # Exports DSSViT, ColorDeconvolution, StainEncoder, OrdinalHead, losses, stain_stats
├── color_deconv.py             # Fixed Ruifrok H/DAB deconvolution (self-contained copy)
├── dss_vit.py                  # DSSViT main assembly
├── stain_encoder.py            # CNN → [B,16,768] tokens
├── ordinal_head.py             # OrdinalHead + cutpoints_to_probs + ordinal_loss + total_loss
└── stain_stats.py              # load_stain_stats / save_stain_stats (JSON)

utils/
├── train_dss_vit.py            # CLI training (3 stages, DDP, AMP, resumable, --debug)
├── evaluate_dss_vit.py         # CLI official-test evaluation
├── metrics_dss_vit.py          # accuracy, balanced-acc, macro-F1, per-class, QWK, CM
└── split_utils.py              # Shared stratified 10% holdout (seed 42)

configs/
└── dss_vit_config.yaml         # All paths + hyperparameters

scripts/
└── precompute_stain_stats.py   # Global H/DAB mean/std (90% train split only)

slurm/
├── train_dss_vit.slurm         # SLURM training job
├── evaluate_dss_vit.slurm      # SLURM evaluation job
└── precompute_stain_stats.slurm # SLURM stain-stats job

notebooks/
└── 07_DSS_ViT_Training.py      # Colab end-to-end alternative

doc/
├── dss_vit_architecture.md     # Architecture summary
└── dss_vit_guide.md            # THIS DOCUMENT

README_HPC.md                   # HPC quick-start
```

---

## 5. Configuration Reference

`configs/dss_vit_config.yaml` — every field explained:

```yaml
# ------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------
model:
  backbone: "vit_base_patch16_224"   # timm backbone name
  pretrained: true                   # Load ImageNet weights
  num_classes: 4                     # HER2 classes {0, 1+, 2+, 3+}
  num_stain_tokens: 16               # StainEncoder output tokens
  stain_bottleneck_dim: 512          # StainEncoder bottleneck width
  image_size: 224                    # Input spatial size
  classifier_dropout: 0.1            # (reserved; not used by OrdinalHead)

# ------------------------------------------------------------
# Stain Statistics (precomputed on the 90% training split)
# ------------------------------------------------------------
stain_stats:
  path: "configs/stain_stats.json"   # Output of precompute_stain_stats.py

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
dataset:
  name: "HER2-IHC-40x"
  classes: ["class_0", "class_1+", "class_2+", "class_3+"]
  image_size: 224
  val_fraction: 0.1                  # 10% validation holdout
  val_seed: 42                       # Deterministic split seed
  split_indices_path: "split_indices_dss_vit.npz"  # Shared split file

# ------------------------------------------------------------
# Training — Stages
# ------------------------------------------------------------
stage1:
  epochs: 5                          # Stage 1 epochs
  lr: 2.0e-4                         # New-module LR (vit frozen)
  freeze: ["vit"]                    # Frozen groups

stage2:
  epochs: 10                         # Stage 2 epochs
  lr: 1.0e-4                         # New-module LR (vit frozen)
  freeze: ["vit"]

stage3:
  epochs: 40                         # Stage 3 epochs
  vit_lr: 1.0e-5                     # ViT backbone LR
  new_lr: 1.0e-4                     # New-module LR
  freeze: []                         # Everything trainable

# ------------------------------------------------------------
# Training — General
# ------------------------------------------------------------
training:
  batch_size: 64                     # PER GPU (A100 default)
  num_workers: 8                     # DataLoader workers
  optimizer: "AdamW"
  weight_decay: 0.05                 # Weights only (bias/LN/BN → 0)
  gradient_clip: 1.0                 # Max grad norm
  amp: true                          # Mixed precision (auto-enables on GPU)
  label_smoothing: 0.1               # CE label smoothing
  ordinal_alpha: 0.1                 # Ordinal loss weight
  mixup_alpha: 0.0                   # 0.0 = disabled; 0.2 = enabled
  main_metric: "accuracy"            # "accuracy" or "qwk" (selection metric)
  seed: 42                           # Reproducibility
  debug: false                       # Run on a few batches

# ------------------------------------------------------------
# Paths (HPC / absolute paths — change as needed)
# ------------------------------------------------------------
paths:
  data_root: "/path/to/HER2_Dataset"
  train_dir: "/path/to/HER2_Dataset/WSI-based-dataset/train"
  test_dir:  "/path/to/HER2_Dataset/WSI-based-dataset/test"
  checkpoint_dir: "/path/to/checkpoints/DSS_ViT"
  log_dir: "/path/to/logs/DSS_ViT"
  experiment_name: "DSS_ViT"
```

---

## 6. Data Pipeline

### 6.1 Dataset

- **Source**: Zenodo — `https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1`
- **Structure** (after extraction):
  ```
  HER2_Dataset/WSI-based-dataset/
  ├── train/  (class_0, class_1+, class_2+, class_3+)
  └── test/   (class_0, class_1+, class_2+, class_3+)
  ```
- **Class distribution** (real):
  | Split | class_0 | class_1+ | class_2+ | class_3+ | Total |
  |---|---|---|---|---|---|
  | train | 3,131 | 1,837 | 523 | 2,602 | 8,093 |
  | test | 658 | 316 | 111 | 762 | 1,847 |

### 6.2 Transforms (unchanged from baseline)

```python
# Train: Resize(224) → HFlip → VFlip → Rotate(10°) → ToTensor()
# Val/Test: Resize(224) → ToTensor()
```

- **No ImageNet Normalize** in the dataloader — raw RGB `[0,1]` is passed to the model, which normalizes internally.

### 6.3 Split (shared utility `utils/split_utils.py`)

- Deterministic **stratified 10% validation holdout** from `train/` (seed 42, `stratify=labels`).
- Saved to `split_indices_dss_vit.npz` (created once, loaded if exists).
- **Shared** by `scripts/precompute_stain_stats.py` and `utils/train_dss_vit.py` — both use the SAME split.

### 6.4 Stain statistics (`scripts/precompute_stain_stats.py`)

- Computes global H/DAB mean & std over the **90% training split only**.
- Runs on **single GPU or CPU** (no DDP).
- Saves to `configs/stain_stats.json`:
  ```json
  {"h_mean": ..., "h_std": ..., "dab_mean": ..., "dab_std": ...}
  ```
- Std clamped to ≥ 1e-6.

---

## 7. Training Process

### 7.1 Stage protocol (55 epochs total)

| Stage | Freeze | Trainable | LR | Epochs |
|---|---|---|---|---|
| 1 | `vit` | StainEncoder, CrossFusion, OrdinalHead | 2e-4 | 5 |
| 2 | `vit` | All new modules | 1e-4 | 10 |
| 3 | — | Entire model | ViT 1e-5, new 1e-4 | 40 |

### 7.2 Optimizer & scheduler

- **AdamW**, weight decay 0.05 on weights only (bias/LayerNorm/BatchNorm → 0).
- Parameter groups **rebuilt at each stage transition** (because `requires_grad` changes).
- Per-stage `CosineAnnealingLR(T_max=stage_epochs)` — recreated per stage.
- Gradient clipping: 1.0 (only params with gradients).

### 7.3 AMP (mixed precision)

- `training.amp: true` by default.
- Auto-enables when a GPU is available:
  ```python
  amp_config = config["training"].get("amp", True)
  use_amp = amp_config and torch.cuda.is_available()
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
  ```
- Training forward under `torch.amp.autocast(device_type="cuda", enabled=...)`.
- `GradScaler`: `scale(loss).backward()` → `unscale_(optimizer)` → clip → `step(optimizer)` → `update()`.
- Validation runs in full precision (`enabled=False`).
- AMP status logged at startup.

### 7.4 MixUp / CutMix

- `training.mixup_alpha: 0.0` (disabled by default).
- Set to `0.2` to enable MixUp (CE + ordinal computed on both mixed targets).

### 7.5 Validation & model selection

- Validation on the 10% holdout **every epoch** (rank 0 only in DDP).
- Metrics tracked: accuracy, balanced accuracy, macro-F1, per-class P/R/F1, **QWK**, confusion matrix, CE loss, ordinal loss.
- **Selection metric: validation accuracy** (`main_metric: accuracy`). QWK is reported but not used for selection.
- Best Stage-3 checkpoint → `best_stage3.pt`.

### 7.6 Logging

- Console + `train.log` in `log_dir`.
- Per-epoch metrics appended to `metrics.jsonl` (JSON lines).

---

## 8. HPC Usage (Toubkal / UM6P)

### 8.1 Environment setup

```bash
# Load modules (adjust to your cluster)
module load CUDA/11.8
module load Python/3.10

# Create conda environment
conda create -n dss_vit_env python=3.10 -y
conda activate dss_vit_env

# Install PyTorch (match CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other deps
pip install timm numpy scipy scikit-learn pyyaml pillow matplotlib
```

> Adjust the PyTorch index URL to match the cluster's CUDA version (cu118, cu121, etc.).

### 8.2 Edit the config

Set all absolute paths in `configs/dss_vit_config.yaml`:

```yaml
paths:
  data_root: "/path/to/HER2_Dataset"
  train_dir: "/path/to/HER2_Dataset/WSI-based-dataset/train"
  test_dir:  "/path/to/HER2_Dataset/WSI-based-dataset/test"
  checkpoint_dir: "/path/to/checkpoints/DSS_ViT"
  log_dir: "/path/to/logs/DSS_ViT"
```

### 8.3 Step 1 — Precompute stain statistics

```bash
sbatch slurm/precompute_stain_stats.slurm
```

Or interactively:

```bash
python scripts/precompute_stain_stats.py --config configs/dss_vit_config.yaml
```

Creates:
- `split_indices_dss_vit.npz` (deterministic split)
- `configs/stain_stats.json` (global H/DAB stats)

### 8.4 Step 2 — Train

**Single GPU** (default):

```bash
sbatch slurm/train_dss_vit.slurm
```

**Multi-GPU (DDP)**: edit the SLURM script:

```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
...
srun python utils/train_dss_vit.py --config "${CONFIG}"
```

**Interactive debug** (few batches):

```bash
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug
```

**Resume** from latest checkpoint:

```bash
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --resume
```

### 8.5 Step 3 — Evaluate on official test

```bash
sbatch slurm/evaluate_dss_vit.slurm
```

Or interactively:

```bash
python utils/evaluate_dss_vit.py --config configs/dss_vit_config.yaml
```

### 8.6 SLURM script templates

#### `slurm/train_dss_vit.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dss_vit_train
#SBATCH --partition=<PARTITION>          # e.g. gpu-a100
#SBATCH --gres=gpu:1                     # Number of GPUs (1, 2, 4)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8-16 recommended
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# module load CUDA/11.8
# module load Python/3.10
# conda activate dss_vit_env

REPO_DIR="/path/to/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_config.yaml"

cd "${REPO_DIR}"
python utils/train_dss_vit.py --config "${CONFIG}"
```

#### `slurm/evaluate_dss_vit.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dss_vit_eval
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

REPO_DIR="/path/to/DSS-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_config.yaml"

cd "${REPO_DIR}"
python utils/evaluate_dss_vit.py --config "${CONFIG}"
```

#### `slurm/precompute_stain_stats.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dss_vit_precompute
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:0                # CPU / single GPU, no DDP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err

REPO_DIR="/path/to/DSS-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_config.yaml"

cd "${REPO_DIR}"
python scripts/precompute_stain_stats.py --config "${CONFIG}"
```

---

## 9. Evaluation

### 9.1 What is evaluated

- **Official test set** (`paths.test_dir`), evaluated **exactly once** at the end.
- Loads `best_stage3.pt` (best validation accuracy in Stage 3).

### 9.2 Metrics reported

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- **QWK** (quadratic weighted kappa)
- Per-class precision / recall / F1 / support
- Confusion matrix

### 9.3 Outputs

- Console output
- `eval.log` in `log_dir`
- `test_results.json` (structured)
- `test_report.txt` (human-readable)

---

## 10. Checkpointing & Resume

### 10.1 Checkpoint files (in `checkpoint_dir`)

| File | Purpose |
|---|---|
| `last.pt` | Latest state (used by `--resume`) |
| `best_stage3.pt` | Best validation accuracy in Stage 3 (used for test eval) |
| `stage1_end.pt` | End of Stage 1 |
| `stage2_end.pt` | End of Stage 2 |

### 10.2 Checkpoint contents

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": ...,
    "stage": ...,
    "metrics": {...},
    "config": {...},
    "split_indices_path": "...",
}
```

### 10.3 Resume behavior

- `--resume` loads `last.pt`.
- Restores model weights, optimizer state, scheduler state, stage, epoch.
- Continues from the exact point of interruption.
- In DDP, the checkpoint is loaded on all ranks (rank 0 reads from disk).

---

## 11. DDP (Multi-GPU) Details

### 11.1 How DDP is triggered

- Auto-detected via `LOCAL_RANK` / `WORLD_SIZE` / `RANK` env vars (set by SLURM `srun`).
- `--distributed` flag forces DDP (single-node multi-GPU fallback).

### 11.2 Key implementation points

| Aspect | Implementation |
|---|---|
| Training sampler | `DistributedSampler` (shuffle=True, `set_epoch(epoch)` each epoch) |
| Validation | **Rank 0 only**, no `DistributedSampler` |
| Model access | `model.module` on rank 0 for metrics/logging/checkpointing |
| Stage freezing | `find_unused_parameters=True` tolerates frozen params (grad=None) |
| Optimizer | **Rebuilt after each `requires_grad` change** (param groups reference current tensors) |
| Checkpointing | **Only rank 0** saves checkpoints/logs |
| AMP + DDP | `GradScaler` created after DDP wrap; `scale/unscale/step/update` used correctly |
| Batch size | Config `batch_size` is **per GPU** (effective = batch_size × world_size) |

### 11.3 SLURM multi-GPU example

```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
...
srun python utils/train_dss_vit.py --config "${CONFIG}"
```

---

## 12. AMP (Mixed Precision)

### 12.1 Default behavior

- `training.amp: true` in config (default).
- Auto-enables when a GPU is available.
- Set `amp: false` to disable.

### 12.2 Implementation

```python
# Training
with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
    pred = model(images)
    loss = total_loss(...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)
scaler.step(optimizer)
scaler.update()

# Validation (full precision)
with torch.amp.autocast(device_type="cuda", enabled=False):
    pred = model(images)
```

### 12.3 Why AMP on A100

- A100 has dedicated Tensor Core acceleration for FP16.
- ~2× training speedup vs FP32.
- Memory savings allow larger batch sizes.

---

## 13. Sanity Checks & Debugging

### 13.1 `--debug` flag

```bash
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug
```

- Runs training on a few batches (2–3) per stage.
- Runs validation on a few batches.
- Quick smoke test before the full run.

### 13.2 What to verify before a full run

1. **Stain stats exist**: `configs/stain_stats.json` (run precompute first).
2. **Split exists**: `split_indices_dss_vit.npz` (created by precompute or training).
3. **Config paths correct**: train/test/checkpoint/log dirs.
4. **AMP status**: log line `AMP (mixed precision): enabled`.
5. **Parameter counts**: log shows per-group counts (StainEncoder ~3–4M, ViT ~86M).

### 13.3 Expected parameter summary (from training log)

```
DSS-ViT Parameter Summary
  color_deconv         :            0
  vit                  :   85,798,656
  stain_encoder        :    3,xxx,xxx   (≤8-9M)
  cross_attn           :    2,360,832
  gate_mlp             :    1,180,416
  ordinal_head         :        2,307
  total                :   9x,xxx,xxx
```

---

## 14. Troubleshooting

### 14.1 `FileNotFoundError: Stain statistics file not found`

Run the precompute script first:

```bash
python scripts/precompute_stain_stats.py --config configs/dss_vit_config.yaml
```

### 14.2 OOM (out of memory)

- Reduce `training.batch_size` (per GPU).
- Reduce `training.num_workers`.
- Check AMP is enabled (`amp: true`).

### 14.3 DDP "unused parameter" errors

- `find_unused_parameters=True` is already set in the DDP wrapper.
- Optimizer is rebuilt after each stage's `requires_grad` change.

### 14.4 NaN/Inf in loss

- Check stain stats are reasonable (not 0 std).
- Check learning rates (Stage 1/2: 2e-4/1e-4; Stage 3: ViT 1e-5).
- Check gradient clipping is active (1.0).

### 14.5 Resume not working

- Ensure `--resume` is passed.
- Ensure `last.pt` exists in `checkpoint_dir`.
- Check the config paths match the training run.

### 14.6 SLURM job fails immediately

- Check `#SBATCH --partition` matches an available partition.
- Check `#SBATCH --gres=gpu:N` matches available GPUs.
- Check module/conda environment is correctly loaded.
- Check `REPO_DIR` and `CONFIG` paths in the SLURM script.

---

## 15. Results Reporting

### 15.1 Training metrics (`metrics.jsonl`)

Each line is a JSON object with:

```json
{
  "accuracy": 0.95,
  "balanced_accuracy": 0.94,
  "macro_f1": 0.93,
  "weighted_f1": 0.95,
  "qwk": 0.97,
  "per_class": {...},
  "confusion_matrix": [[...]],
  "val_total_loss": 0.5,
  "val_ce_loss": 0.4,
  "val_ord_loss": 0.1,
  "train_total_loss": 0.3,
  "train_ce_loss": 0.25,
  "train_ord_loss": 0.05,
  "train_acc": 98.5,
  "epoch": 42,
  "stage": 3
}
```

### 15.2 Test results (`test_results.json`)

```json
{
  "checkpoint": ".../best_stage3.pt",
  "metrics": {
    "accuracy": 0.95,
    "balanced_accuracy": 0.94,
    "macro_f1": 0.93,
    "weighted_f1": 0.95,
    "qwk": 0.97,
    "per_class": {...}
  },
  "confusion_matrix": [[...]],
  "class_names": ["class_0", "class_1+", "class_2+", "class_3+"]
}
```

### 15.3 Comparison vs baseline

| Model | Test accuracy |
|---|---|
| Plain ViT-B16 baseline | **95.02%** |
| **DSS-ViT** | **To be measured** |

The goal is to beat 95.02% on the same official test split.

---

*Document generated from the actual codebase (`models_v2_1/`, `utils/`, `configs/`, `scripts/`, `slurm/`, `notebooks/`).*