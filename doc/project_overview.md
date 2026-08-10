# DSCA-ViT — Complete Project Documentation

> **Dual-Stain Cross-Attention Vision Transformer for HER2 IHC Scoring**
> A biologically-informed deep learning architecture that separates Hematoxylin and DAB stains and fuses them via spatially-biased cross-attention.

---

## Table of Contents

1. [Motivation & Problem](#1-motivation--problem)
2. [Biological Background](#2-biological-background)
3. [Design Philosophy](#3-design-philosophy)
4. [Architecture Overview](#4-architecture-overview)
5. [Component-by-Component Explanation](#5-component-by-component-explanation)
6. [The Core Novelty: Spatially-Biased Cross-Attention](#6-the-core-novelty-spatially-biased-cross-attention)
7. [Data Pipeline](#7-data-pipeline)
8. [Training Strategy](#8-training-strategy)
9. [The Baseline: Plain ViT-B/16](#9-the-baseline-plain-vit-b16)
10. [Project Structure](#10-project-structure)
11. [Motivation for Every Choice](#11-motivation-for-every-choice)
12. [Verification & Testing](#12-verification--testing)
13. [Known Risks & Open Questions](#13-known-risks--open-questions)
14. [Ablation Plan](#14-ablation-plan)
15. [Reproducing the Project](#15-reproducing-the-project)
16. [Future Work & Extensions](#16-future-work--extensions)

---

## 1. Motivation & Problem

**Clinical problem**: HER2 (Human Epidermal growth factor Receptor 2) is a protein that promotes aggressive breast cancer growth. Determining the HER2 status of a tumor is a critical decision point — it determines whether the patient is eligible for targeted therapy (e.g., trastuzumab/Herceptin).

**The IHC assay**: Immunohistochemistry (IHC) uses two stains on a single tissue section:
- **Hematoxylin (H)** — stains cell nuclei blue/purple → reveals **morphology**
- **DAB** (3,3'-Diaminobenzidine) — stains HER2 protein brown → reveals **membrane HER2 expression**

**The scoring task**: Pathologists assign a score from {0, 1+, 2+, 3+} based on the completeness and intensity of brown membrane staining:
| Score | Meaning |
|:---|:---|
| 0 | No staining or incomplete membrane staining in <10% of cells |
| 1+ | Weak, incomplete membrane staining |
| 2+ | Weak-to-moderate complete membrane staining (equivocal — needs FISH test) |
| 3+ | Strong, complete membrane staining |

**Why automate it**:
- Manual scoring is subjective and has high inter-observer variability
- It is time-consuming (pathologist hours per case)
- It is a high-volume test (routinely ordered for all breast cancer patients)
- An objective, reproducible ML model could standardize scoring

**Why HER2 IHC images are NOT natural images**:
- They contain **two overlapping stains** with different biological meanings
- The stains are mixed in RGB space — the raw image must be unmixed
- The stains are **pixel-perfectly registered** (same tissue section, same scan)
- These properties make generic image classifiers suboptimal

---

## 2. Biological Background

### The Beer-Lambert Law

Color deconvolution is based on the physics of light absorption:

$$OD = -\log_{10}\left(\frac{I}{I_0}\right)$$

where $I_0$ is the incident light intensity and $I$ is the transmitted intensity. Each stain has a characteristic **absorption spectrum** (stain vector in RGB OD space).

### Ruifrok & Johnston Stain Matrix

The fixed stain matrix used in this project (calibrated by Ruifrok & Johnston, 2001):

```
H:    [0.650, 0.704, 0.286]
DAB:  [0.269, 0.570, 0.777]
Res:  [0.711, 0.423, 0.562]
```

The inverse of this matrix unmixes an RGB image into H, DAB, and Residual channels:

```
OD_RGB (B,3,H,W)  →  OD @ M_inv  →  [H, DAB, Residual] (B,3,H,W)
```

### Why pixel-perfect registration matters

This is the **unique scientific property** that no generic multimodal architecture can exploit:

> Pixel $(x,y)$ in the H channel corresponds **exactly** to pixel $(x,y)$ in the DAB channel.

In other modalities (RGB-Depth, CT-PET, RGB-Thermal), spatial correspondence is approximate at best — different sensors, different optics, parallax offsets. But in color deconvolution, the decomposition is **mathematically exact by construction**. This is the core scientific insight of the project.

---

## 3. Design Philosophy

> **HER2 IHC images are not natural images. The architecture must encode what pathologists already know.**

Three biological facts drive the design:

| Biological Fact | Architectural Consequence |
|:---|:---|
| H and DAB encode *different* biological signals | → **Dual-stream processing** |
| Both stains originate from the *same* tissue section | → **Shared encoder** (same spatial features) |
| Pixel $(x,y)$ in H corresponds *exactly* to pixel $(x,y)$ in DAB | → **Spatially-biased cross-attention** |

The third fact — perfect spatial registration — is the core novelty that most multimodal architectures cannot exploit because their modalities are not pixel-aligned.

---

## 4. Architecture Overview

```
                        ┌─────────────────────┐
                        │   RGB Patch (224²×3) │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Color Deconvolution │
                        │  (Ruifrok, fixed)    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │   H Channel       │         │   DAB Channel      │
          │   (224²×1)        │         │   (224²×1)         │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │  Proj_H: 1→3 ch   │         │  Proj_D: 1→3 ch   │
          │  Conv2d(1,3,1×1)  │         │  Conv2d(1,3,1×1)  │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
                    │   ┌─────────────────────┐   │
                    ├──►│  Shared ViT-B/16    │◄──┤
                    │   │  Blocks 1–9         │   │
                    │   │  (100% shared)      │   │
                    │   └──────────┬──────────┘   │
                    │              │               │
          ┌─────────▼────┐                 ┌──────▼────────┐
          │  H₉ Tokens   │                 │  D₉ Tokens    │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
          ┌──────▼────────────────────────────────▼────────┐
          │    SPATIALLY-BIASED BIDIRECTIONAL               │
          │    CROSS-ATTENTION MODULE                       │
          │    (attention biased toward corresponding       │
          │     spatial positions)                          │
          └──────┬────────────────────────────────┬────────┘
                 │                                │
          ┌──────▼───────┐                 ┌──────▼────────┐
          │  Ĥ Tokens    │                 │  D̂ Tokens     │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
                 │   ┌─────────────────────┐      │
                 ├──►│  Shared ViT-B/16    │◄─────┤
                 │   │  Blocks 10–12       │      │
                 │   │  (100% shared)      │      │
                 │   └──────────┬──────────┘      │
                 │              │                  │
          ┌──────▼───────┐                 ┌──────▼────────┐
          │  H_final     │                 │  D_final      │
          │  (197×768)   │                 │  (197×768)    │
          └──────┬───────┘                 └──────┬────────┘
                 │                                │
          ┌──────▼────────────────────────────────▼────────┐
          │           TOKEN FUSION                         │
          │           (Gated per-token)                    │
          └────────────────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Fused Tokens       │
                    │  (197×768)          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  REFINEMENT BLOCK   │
                    │  (1 Transformer     │
                    │   Block, new init)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  CLASSIFICATION     │
                    │  HEAD               │
                    └──────────┬──────────┘
                               │
                          HER2 Score
                        {0, 1+, 2+, 3+}
```

### What was removed from earlier designs (and why)

| Component | Status | Reason |
|:---|:---|:---|
| RGB residual stream | Removed | Muddies the contribution — reviewer could attribute gains to RGB |
| Stain-specific adapters | Removed | Introduces another variable; would be a separate paper |
| Multi-depth cross-attention | Removed | Single insertion keeps design clean; multi-depth is an ablation |

### What was added (and why)

| Component | Status | Reason |
|:---|:---|:---|
| Spatial correspondence bias | **Added** | Core novelty — exploits perfect registration |
| Refinement block | **Added** | Allows reasoning over fused representation |

---

## 5. Component-by-Component Explanation

### 5.1 Color Deconvolution (`models/color_deconv.py`)

**What it does**: Separates RGB into H and DAB channels using the fixed Ruifrok stain matrix.

**Implementation details**:
- Stain matrix registered as a **buffer** → automatically moves to GPU, never updated by backprop
- Wrapped in `torch.no_grad()` in `dsca_vit.py` → no gradient flows through deconv (it's physics, not learnable)
- Output clamped to `>= 0` (negative optical density is non-physical)
- `deconvolve_numpy()` variant for visualization without PyTorch

**Motivation**: Fixed, interpretable, zero data-dependency. The biology is known ahead of time, so we encode it rather than learn it.

### 5.2 Stain Channel Projection (`models/shared_vit.py`)

**What it does**: `Conv2d(1, 3, 1×1)` that converts a single-channel stain into a 3-channel pseudo-RGB image so the pretrained ViT patch embedding (which expects 3 channels) can be used.

**Initialization**: "repeat" mode initializes weights to `ones` and bias to `zeros`, so each output channel ≈ input (identity-like). This preserves the pretrained ViT behavior early in training.

**Motivation**: The pretrained ViT expects 3-channel input. Rather than modifying the patch embedding (which would break weight loading), we add a minimal learnable projection.

### 5.3 Shared ViT Encoder (`models/shared_vit.py`)

**What it does**: Wraps `timm.create_model("vit_base_patch16_224")` and:
- Removes the classification head (`num_classes=0`)
- Splits the 12 blocks into `blocks_before` (0..8) and `blocks_after` (9..11)
- Exposes `embed()`, `forward_before()`, `forward_after()`

**Critical**: Both H and DAB streams use the **exact same parameter tensors** — weight sharing is real, not simulated.

**Motivation**: The two stains come from the same tissue. The morphological features (edges, textures, nuclei shapes) are largely shared. A shared encoder prevents 2× parameter blowup and acts as a strong regularizer.

### 5.4 Bidirectional Cross-Attention (`models/cross_attention.py`)

**Three classes**:

1. **`SpatialBiasMatrix`** — Learnable `(197, 197)` parameter initialized with:
   - `B[i,i] = +beta` (self-correspondence bonus, default 1.0)
   - `B[i,j] = -gamma * distance(i,j)` (distance penalty, default gamma=0.1)
   - `B[0,:] = B[:,0] = 0` (CLS has no spatial bias)
   - Prints initialization values at construction:
     ```
     beta         : 1.0
     gamma        : 0.1
     max distance : 18.38
     max penalty  : -1.84
     ```

2. **`CrossAttentionLayer`** — Single-direction cross-attention with pre-norm:
   ```
   Q = W_q(LN(source))
   K = W_k(LN(context))
   V = V_q(LN(context))
   logits = QKᵀ/√d_k + B
   attn = softmax(logits)
   output = source + W_o(attn @ V)
   ```
   Stores `attn_weights` for visualization.

3. **`BidirectionalCrossAttention`** — Parallel bidirectional computation from the **original** tokens (no ordering bias), each direction with its own FFN.

### 5.5 Gated Token Fusion (`models/fusion.py`)

**What it does**:
```python
# CLS token (index 0):
F_0 = Linear(1536, 768)([CLS_H || CLS_D])

# Patch tokens (indices 1..196):
g_i = sigmoid(Linear(1536, 768)([H_i || D_i]))
F_i = g_i ⊙ H_i + (1 - g_i) ⊙ D_i
```

**Why gated fusion is right here**: The gate `g_i` is **biologically interpretable**:
| Gate value | Meaning |
|:---|:---|
| `g_i ≈ 1` | Patch relies on **morphology** (Hematoxylin) |
| `g_i ≈ 0` | Patch relies on **HER2 signal** (DAB) |
| `g_i ≈ 0.5` | Both stains contribute equally |

**This can be visualized as heatmaps** — a strong interpretability figure showing the model learned biologically meaningful fusion patterns.

### 5.6 Refinement Block (`models/fusion.py`)

**What it is**: A single standard transformer block (self-attention + FFN), Xavier initialized (trained from scratch, not pretrained).

**Why it's needed**: Cross-attention *exchanges* information between stains. But the network still needs to *reason over* the fused representation — e.g., aggregate evidence across the whole patch (is membrane staining complete? what fraction of cells are HER2+?).

**Initialization note**: This block learns to reason over *fused stain representations*, which have no ImageNet counterpart — so it's intentionally trained from scratch.

### 5.7 Classification Head (`models/fusion.py`)

```python
cls = tokens[:, 0, :]           # CLS token (B, 768)
gap = mean(tokens[:, 1:, :])    # Global average pooling (B, 768)
z   = [cls || gap]              # (B, 1536)
out = LN → Linear(768) → GELU → Dropout(0.1) → Linear(4)
```

**Why CLS + GAP**: CLS captures the global summary; GAP captures average spatial evidence. This dual pooling is more robust than either alone and is standard in recent ViT papers.

---

## 6. The Core Novelty: Spatially-Biased Cross-Attention

### 6.1 The Problem with Standard Cross-Attention

Standard multi-head cross-attention computes:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

This produces a $197 \times 197$ attention matrix where **every H token can attend equally to every DAB token**. The network must *learn* that patch 37 in H should focus on patch 37 in DAB — wasteful, because we already *know* this from the physics.

### 6.2 The Insight

> **Patch $i$ in H is derived from exactly the same $16 \times 16$ pixel region as patch $i$ in DAB.**

- Patch 37 in H should **strongly** attend to patch 37 in DAB (same tissue location)
- Patch 37 may **weakly** attend to patches 36, 38, 23, 51 (spatial neighbors)
- Patch 37 should **rarely** attend to patch 182 (distant, unrelated tissue)

### 6.3 Strategy S1 — Learnable Additive Bias (implemented)

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right) V$$

where $B$ is initialized:
$$B_{ij} = \begin{cases} +\beta & \text{if } i = j \\ -\gamma \cdot d(i,j) & \text{if } i \neq j \end{cases}$$

**Properties**:
- At init, attention is strongly biased toward the corresponding patch
- During training, the network can override the bias if non-local attention is useful
- CLS token has no spatial bias (row/col = 0)

### 6.4 Why gamma = 0.1 (not 0.5)

On a 14×14 grid, the maximum distance is `sqrt(13² + 13²) ≈ 18.38`:
- `gamma=0.5` → max penalty `-9.2` logits → effectively a **hard mask** (attention logits are usually only `[-3, 3]`)
- `gamma=0.1` → max penalty `-1.84` → a **soft prior** that gently encourages locality while still allowing long-range learning

> **Design decision**: `gamma=0.1` provides a soft inductive bias, not a hard constraint.

### 6.5 Why This Is Novel

No existing cross-attention mechanism in the literature exploits **guaranteed pixel-level spatial registration** between streams because:
1. RGB-Depth fusion → depth maps may be misaligned (sensor offset)
2. CT-PET fusion → different scanners, different resolutions
3. RGB-Thermal → different optics, parallax
4. Text-image fusion → no spatial correspondence at all

**Color deconvolution is unique**: the decomposition is pixel-perfect by construction. The spatial bias is not an approximation — it encodes a **mathematical certainty**.

---

## 7. Data Pipeline

### 7.1 Dataset — HER2-IHC-40x

**Source**: Zenodo — `https://zenodo.org/records/15179608/files/her2-ihc-40x-wsi.zip?download=1`

**Size**: ~346 MB compressed (downloads in ~2 min on Colab)

**Structure** (after download + extraction):
```
HER2_Dataset/
└── WSI-based-dataset/
    ├── train/
    │   ├── class_0/     (HER2 negative)
    │   ├── class_1+/    (weak positive)
    │   ├── class_2+/    (equivocal)
    │   └── class_3+/    (strong positive)
    └── test/
        ├── class_0/
        ├── class_1+/
        ├── class_2+/
        └── class_3+/
```

**Image format**: PNG, RGB, 40x magnification (WSI-based split)

**Classes** (ordinal diagnostic categories):
| Class | Clinical Meaning | Expected DAB signal |
|:---|:---|:---|
| `class_0` | HER2 negative | Very low (little/no brown) |
| `class_1+` | Weak positive | Slightly higher, faint incomplete membrane |
| `class_2+` | Equivocal (needs FISH) | Moderate, weak-to-moderate complete membrane |
| `class_3+` | Strong positive | High (intense brown membrane) |

**Download & extraction** (proven logic from `model_HER2_ViT.ipynb` Cell 2, used in both `train.py` Cell 5 and `visualize.py`):
1. `wget` the main `her2-ihc-40x-wsi.zip` (only if not present)
2. Extract main archive → `WSI-based-dataset/` containing `train_data_wsi.zip` + `test_data_wsi.zip`
3. Extract nested `train_data_wsi.zip` → `train/`, `test_data_wsi.zip` → `test/`
4. Delete all `.zip` files to save disk space
5. Assert `train/` and `test/` exist
6. Print per-class image counts

**Real class distribution (observed from actual download)**:

```
Dataset location:
/content/HER2_Dataset/WSI-based-dataset/train
/content/HER2_Dataset/WSI-based-dataset/test

Dataset successfully prepared!

Folder structure:

train/
   class_0     3131 images
   class_1+    1837 images
   class_2+     523 images
   class_3+    2602 images

test/
   class_0      658 images
   class_1+     316 images
   class_2+     111 images
   class_3+     762 images
```

| Split | class_0 | class_1+ | class_2+ | class_3+ | Total |
|:---|:---|:---|:---|:---|:---|
| **train** | 3,131 | 1,837 | **523** | 2,602 | 8,093 |
| **test** | 658 | 316 | **111** | 762 | 1,847 |
| **Total** | 3,789 | 2,153 | **634** | 3,364 | 9,940 |

**Class imbalance note**: `class_2+` (equivocal, the hardest clinically) is severely underrepresented — only **523 train / 111 test** images vs. 3,131/658 for `class_0`. This imbalance may require **class-weighted loss or oversampling** for reliable 2+ classification.

### 7.2 Transforms (`datasets/transforms.py`)

```python
# CRITICAL: NO ImageNet Normalize() here
get_train_transform():  Resize(224) → HFlip → VFlip → Rotate(10°) → ToTensor()
get_test_transform():   Resize(224) → ToTensor()
```

**Why NO ImageNet normalization**:
- Color deconvolution requires raw RGB in `[0, 1]` range
- ImageNet normalization would corrupt the optical density computation (Beer-Lambert law)
- The pretrained ViT still works due to transfer learning robustness, but needs adaption epochs

**Why augmentation is on RGB before deconvolution**:
- If H and DAB are augmented independently, the spatial correspondence is broken
- Both channels must see the **same geometric transform**

### 7.3 Dataset Class (`datasets/dataset.py`)

- `HER2Dataset` — expects `class_0/`, `class_1+/`, `class_2+/`, `class_3+/` subdirectories
- Opens with `PIL.Image.open().convert("RGB")`, `ToTensor()` scales to `[0, 1]`
- Provides `get_class_distribution()` for class-balance analysis

### 7.4 DataLoader configuration

| Parameter | Value | Reason |
|:---|:---|:---|
| `batch_size` | 32 (drop to 16 if OOM) | Effective 64 through shared encoder |
| `shuffle` | True (train) / False (val) | Standard |
| `num_workers` | 2 | Colab CPU budget |
| `pin_memory` | True | Faster GPU transfer |

---

## 8. Training Strategy

### 8.1 Complete Hyperparameter Table

| Hyperparameter | Stage 1 | Stage 2 |
|:---|:---|:---|
| **Model** | DSCA-ViT (ViT-B/16 backbone) | Same |
| **Backbone** | `vit_base_patch16_224` (ImageNet pretrained) | Same |
| **Split point** | `split_after=9` (cross-attn after block 9) | Same |
| **Spatial bias beta** | 1.0 | Same |
| **Spatial bias gamma** | 0.1 | Same |
| **Classifier dropout** | 0.1 | Same |
| **Input size** | 224×224 | Same |
| **Num classes** | 4 | Same |
| **Batch size** | 32 | 32 |
| **Epochs** | 30 | 30 |
| **Optimizer** | Adam | Adam |
| **Encoder LR** | 0 (frozen) | 1e-5 |
| **New components LR** | 1e-4 | 1e-4 |
| **Scheduler** | CosineAnnealingLR (T_max=30) | CosineAnnealingLR (T_max=30) |
| **Loss** | CrossEntropyLoss | CrossEntropyLoss |
| **Seed** | 42 | 42 |
| **Augmentation** | Resize, HFlip, VFlip, Rotate(10°) | (val: Resize only) |

### 8.2 Two-Stage Training — How It Goes

#### Stage 1 — Train New Components (Encoder Frozen)

**Goal**: Let the new components (1→3 projections, cross-attention, fusion, refinement, head) learn to produce useful features *given* the frozen pretrained backbone.

**Flow**:
1. Freeze all encoder params: `for param in model.encoder.parameters(): param.requires_grad = False`
2. Build optimizer from `param_groups["new"]` only (projections, cross-attn, fusion, refinement, classifier) at LR 1e-4
3. Train 30 epochs with CosineAnnealingLR
4. Each epoch: `train_one_epoch` (calls `model.train()`) → `validate_one_epoch` (calls `model.eval()`)
5. Save best checkpoint (by val acc) to `/content/best_stage1_DSCA_ViT.pth`
6. Copy to Google Drive: `MyDrive/HER2_Checkpoints/DSCA_ViT/Stage1/`

**Why**: The new components start from random init. Training them on a frozen backbone is stable and fast — they learn to interpret the pretrained features without disturbing them.

#### Stage 2 — Full Fine-tuning

**Goal**: Adapt the entire model (including the 86M shared ViT weights) to the IHC input distribution.

**Flow**:
1. Load best Stage 1 checkpoint: `load_checkpoint(path=DEST_S1, model=model, device=device)`
2. Unfreeze everything: `for param in model.parameters(): param.requires_grad = True`
3. Build **discriminative** optimizer:
   ```python
   optimizer = optim.Adam([
       {"params": param_groups["encoder"], "lr": 1e-5},   # slow backbone adaptation
       {"params": param_groups["new"],     "lr": 1e-4},   # faster new components
   ])
   ```
4. Train 30 epochs with CosineAnnealingLR
5. Save best checkpoint to `/content/best_stage2_DSCA_ViT.pth`
6. Copy checkpoint + weights-only file to Drive: `MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/`

**Why**: The backbone was pretrained on ImageNet (natural images). The IHC input distribution (stain OD values, no normalization) is different. Stage 2 lets the backbone slowly adapt at a low LR (1e-5) while new components continue at 1e-4.

### 8.3 Training-critical verification

Runtime-tested that in the real notebook pipeline:
- ✅ Stage 1: encoder truly frozen (`requires_grad=False` on all 150 encoder params)
- ✅ Stage 2: encoder truly trainable (`requires_grad=True` on all 150)
- ✅ `model.train()` is called in `train_one_epoch` → encoder modules are in `.training == True`
- ✅ Optimizer includes encoder params at 1e-5 in Stage 2
- ✅ Gradients actually flow to encoder after `loss.backward()`

### 8.4 Loss

- **Cross-entropy** (start here)
- Ordinal regression as an ablation (HER2 scores are ordinal: 0 < 1+ < 2+ < 3+)

### 8.5 Memory note

- Batch size 32 → effective batch 64 through the shared encoder (batched cat trick)
- ViT-B/16 at batch 64 needs ~13-14 GB → **tight on Colab T4 (15 GB)**
- If OOM: drop to batch 16, then try 24

### 8.6 Observed runtime output (Colab, T4 GPU)

When the model is built, you should see (verified output):

```
model.safetensors: reconstructing file: 100%
 346MB / 346MB, 30.2MB/s
model.safetensors: downloading bytes:
 330MB, 26.8MB/s

Spatial Bias Initialization
---------------------------
beta         : 1.0
gamma        : 0.1
max distance : 18.38
max penalty  : -1.84

DSCA-ViT Parameter Summary
  color_deconv         :            0
  proj_h               :            6
  proj_d               :            6
  encoder              :   85,798,656
  cross_attention      :   14,217,625
  fusion               :    2,360,832
  refinement           :    7,087,872
  classifier           :    1,186,564
  total                :  110,651,561
  trainable            :  110,651,561
```

- The `model.safetensors` lines are the ImageNet ViT-B/16 weights downloading (~346 MB)
- The `Spatial Bias Initialization` block confirms gamma=0.1, max penalty -1.84
- The parameter summary confirms the real total: **110,651,561 (~110.6M)** with the encoder being 85.8M (78%)

---

## 9. The Baseline: Plain ViT-B/16

> The baseline is a **standard ImageNet-pretrained ViT-B/16**, trained directly on the raw RGB IHC images in the standard two-stage transfer-learning paradigm. It is the model that DSCA-ViT is designed to outperform.

### 9.1 What the baseline is

The baseline is the exact model from `model_HER2_ViT.ipynb`:

- **Backbone**: `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=4)`
- **Input**: Raw RGB IHC image (224×224), **with ImageNet normalization** applied before the ViT
- **No color deconvolution** — it sees the mixed RGB stains
- **No dual-stream, no cross-attention, no spatial bias** — it's a plain ViT classifier
- Uses the same dataset, the same 4-class output, and the same 2-stage training paradigm as DSCA-ViT

**Why it's the baseline**: It is the natural "what if we just use a pretrained ViT?" comparison. Any advantage DSCA-ViT shows over this model demonstrates the value of the domain-specific design (stain separation, dual streams, spatially-biased cross-attention).

### 9.2 How the baseline is trained

The baseline uses the **same 2-stage paradigm** as DSCA-ViT (from `model_HER2_ViT.ipynb`):

#### Stage 1 — Frozen backbone, train head only

| Setting | Value |
|:---|:---|
| Backbone | ViT-B/16 `vit_base_patch16_224` (ImageNet pretrained) |
| Trainable | Only the classification head (`model.head`) |
| Frozen | All transformer blocks + patch embedding + positional embeddings |
| Optimizer | Adam, LR 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 30 |
| Augmentation | Resize(224), HFlip, VFlip, Rotate(10°), **Normalize(ImageNet)** |

#### Stage 2 — Full fine-tuning

| Setting | Value |
|:---|:---|
| Load | Best Stage 1 checkpoint |
| Trainable | Entire model (all params unfrozen) |
| Discriminative LR | Backbone 1e-5, Head 1e-4 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 30 |

### 9.3 Baseline results (real, from the actual run)

**Stage 1** (frozen backbone):

- Best Validation Accuracy: **89.17%** @ Epoch 21
- Trained: 8,093 train images / 1,847 validation images

**Stage 2** (full fine-tune):

- Best Validation Accuracy: **95.02%** @ Epoch 3

**Final per-class performance (Stage 2, 95.02% accuracy)**:

| Class | Precision | Recall | F1-score | Support |
|:---|:---|:---|:---|:---|
| `class_0` | 0.9521 | 0.9362 | 0.9441 | 658 |
| `class_1+` | 0.8571 | 0.8734 | 0.8652 | 316 |
| `class_2+` | 0.8879 | 0.9279 | 0.9075 | 111 |
| `class_3+` | 0.9974 | 0.9974 | 0.9974 | 762 |

| Metric | Value |
|:---|:---|
| Accuracy | 95.02% |
| Macro F1 | 0.9285 |
| Weighted F1 | 0.9504 |
| Weighted Precision | 0.9507 |
| Weighted Recall | 0.9502 |

**Interesting observations about the baseline**:
1. `class_3+` is nearly perfect (F1 0.9974) — strong, complete membrane staining is the easiest to detect
2. `class_1+` is the weakest (F1 0.8652) — weak/faint staining is genuinely hard
3. `class_2+` (equivocal) does surprisingly well (F1 0.9075) despite only **111 test samples** — but the small support makes this unreliable
4. **Stage 2 best epoch was only 3** — the model quickly overfits after the backbone unfreezes (train acc → 100%, val plateaus). This is a known risk flagged for DSCA-ViT too.

### 9.4 Baseline vs. DSCA-ViT — the comparison to make

| Aspect | Baseline (ViT-B/16) | DSCA-ViT |
|:---|:---|:---|
| Input | Raw RGB, ImageNet-normalized | Raw RGB → color deconvolution → H + DAB |
| Architecture | Single-stream ViT | Dual-stream shared ViT + cross-attention |
| Spatial bias | None | Spatially-biased cross-attention (gamma=0.1) |
| Fusion | None (plain classifier) | Gated token fusion |
| Parameters | ~86M | ~110.6M (+22%) |
| Baseline accuracy | **95.02%** | To be measured |

**What the A-series ablations will prove**:
- `A1` (baseline) → the number to beat
- `A6` (DSCA-ViT) → the proposed architecture
- If `A6 > A1`, the domain-specific design wins

---

## 10. Project Structure

```
DSCA-ViT/
├── models/                    # Neural network modules
│   ├── __init__.py            # Package exports
│   ├── dsca_vit.py            # Top-level DSCAViT assembly (9-step forward pass)
│   ├── shared_vit.py          # SharedViTEncoder + StainChannelProjection
│   ├── color_deconv.py        # Fixed Ruifrok H-DAB color deconvolution
│   ├── cross_attention.py     # Spatially-biased bidirectional cross-attention
│   └── fusion.py              # GatedFusion, RefinementBlock, ClassificationHead
│
├── datasets/
│   ├── __init__.py
│   ├── dataset.py             # HER2Dataset (unchanged from baseline)
│   └── transforms.py          # Transforms WITHOUT ImageNet normalization
│
├── utils/
│   ├── __init__.py
│   ├── train.py               # train_one_epoch (calls model.train())
│   ├── evaluate.py            # validate_one_epoch (calls model.eval())
│   ├── metrics.py             # compute_metrics, print_metrics (sklearn)
│   └── checkpoint.py          # save_checkpoint, load_checkpoint
│
├── configs/
│   └── dsca_vit_b16.yaml      # Hyperparameters (reference — not wired in yet)
│
├── notebooks/
│   ├── train.py               # Colab training script (13 cells, 2-stage)
│   ├── sanity_check.py        # 8 module unit tests (run before training)
│   ├── visualize.py           # Deconv sanity check + gate/attn visualizations
│   └── deconv_sanity_check.py # Dedicated 20-patch deconv verification
│
├── doc/
│   ├── implementation_plan.md # Architecture blueprint (v2)
│   ├── walkthrough.md         # Implementation summary & risk analysis
│   └── project_overview.md    # THIS DOCUMENT
│
└── convert_to_ipynb.py        # .py → .ipynb converter (cell markers)
```

---

## 11. Motivation for Every Choice

| Decision | Motivation |
|:---|:---|
| **Fixed color deconvolution** (not learned) | Physics-based, interpretable, no data dependency. The stain vectors are known constants from the assay. |
| **Ruifrok matrix** (not Macenko per-slide) | Standard, reproducible, exactly what the clinical literature uses. Macenko is an ablation/future work. |
| **1→3 channel projection** (per stain) | Preserves the pretrained ViT patch embedding without modification. "Repeat" init = identity-like, keeps pretrained behavior. |
| **Shared encoder (100%)** | Both stains from same tissue; prevents 2× parameters; strong regularization. Parameter cost: +22% not +100%. |
| **Split at block 9** (not 6 or 12) | Late-mid fusion — allows deep shared features but leaves 3 blocks for post-fusion reasoning. Placement is ablation C-series. |
| **Bidirectional cross-attention** (not unidirectional) | Both H→DAB and DAB→H carry clinical meaning: morphology informs signal interpretation, and signal informs morphology context. |
| **Spatial bias (gamma=0.1)** | Encodes the mathematical certainty of pixel registration as a soft prior, not a hard mask. |
| **Gated fusion** (not concat/average) | Per-patch adaptive weighting with biological interpretability (gate = which stain drives classification). |
| **Refinement block** | Gives the network capacity to reason over the fused representation (aggregating evidence across the slide patch). |
| **CLS + GAP classification head** | More robust than either alone; standard in modern ViT papers. |
| **No ImageNet normalization** | Would corrupt Beer-Lambert optical density computation. |
| **Augment on RGB before deconv** | Ensures both stains see the identical transform, preserving spatial correspondence. |
| **Two-stage training** | Stage 1 trains new components safely on frozen backbone; Stage 2 slowly adapts backbone at low LR. |
| **Batched dual-stream forward** | cat([H, D], dim=0) → single GPU call → ~40-60% wall-clock overhead instead of 2× |
| **Cosine annealing** | Standard robust LR schedule for fine-tuning. |
| **gamma=0.1 default in config/yaml/notebook** | Single source of truth for the soft-prior spatial bias. |

### Parameter budget (~110.6M total)

| Component | Params | Share | Notes |
|:---|:---|:---|:---|
| Shared ViT-B/16 encoder | 85.8M | 78% | Pretrained ImageNet, the heavy lifting |
| Cross-attention (QKV + projections + FFN ×2) | 14.2M | 13% | The novel contribution |
| Refinement block | 7.1M | 6% | Post-fusion reasoning, from scratch |
| Gated fusion | 2.4M | 2% | Lightweight, interpretable |
| Classification head | 1.2M | 1% | CLS + GAP dual pooling |
| 1→3 projections ×2 | 12 | ~0% | Negligible |
| Color deconvolution | 0 | 0% | Fixed buffer, no learnable params |

**Only ~22% of parameters are new** — the rest are pretrained. This is a strong argument for reviewers worried about data efficiency.

---

## 12. Verification & Testing

### Module-level tests (`notebooks/sanity_check.py`) — ALL PASSED

| Test | Result |
|:---|:---|
| ColorDeconvolution | `(B,1,224,224)`, non-negative, 0 learnable params |
| StainChannelProjection | `(B,3,224,224)` |
| SharedViTEncoder | embed/before/after all `(B,197,768)` |
| BidirectionalCrossAttention | H/D out `(B,197,768)`, attn_weights stored |
| GatedFusion | fused `(B,197,768)`, gates `(B,196,768)` in [0,1] |
| RefinementBlock | `(B,197,768)` |
| ClassificationHead | logits `(B,4)` |
| Full DSCAViT end-to-end | logits `(B,4)`, gates `(B,196,768)` |

### Spatial bias runtime verification

```
Spatial Bias Initialization
---------------------------
beta         : 1.0
gamma        : 0.1
max distance : 18.38
max penalty  : -1.84
```
- ✅ Diagonal = +1.0 (self-correspondence bonus)
- ✅ CLS row/col = 0 (no spatial bias)
- ✅ Max penalty = -1.84 (soft prior, not hard mask)

### Gradient flow verification

- ✅ Gradients flow through 1→3 projections (deconv is `no_grad`, projections get grads)
- ✅ 150/150 encoder params receive gradients in Stage 2

### Stage 2 trainability verification

- ✅ All encoder params `requires_grad=True` after unfreezing
- ✅ `model.train()` puts encoder modules in training mode
- ✅ Encoder in optimizer at 1e-5
- ✅ Gradients flow to encoder

---

## 13. Known Risks & Open Questions

### 🔴 Critical

| Risk | Detection | Fix |
|:---|:---|:---|
| Stain vectors miscalibrated for this dataset | `visualize_deconvolution_batch()` on 20 random patches — if DAB looks gray/noisy or H looks brown, vectors are wrong | Macenko per-slide estimation; per-image stain vectors |
| OOM at batch_size=32 on T4 | First training step | Reduce to 16 |

### 🟠 High

| Risk | Detection | Fix |
|:---|:---|:---|
| Input distribution mismatch (non-ImageNet inputs to pretrained ViT) | Stage 1 val acc plateau; DAB stream underperforms H | Post-projection LayerNorm |
| Overfitting on small dataset | Train acc → 100%, val degrades | Early stopping (patience=7) |
| DAB stream systematically underperforms H | Per-stream analysis | Stain-specific normalization |

### 🟡 Medium

| Risk | Detection | Fix |
|:---|:---|:---|
| YAML config not wired into notebook | Code review | Add `yaml.safe_load` |
| Checkpoint key naming mismatch | `sanity_check.py` + round-trip test | Use `best_val_accuracy` consistently |
| GPU memory fragmentation | Mid-epoch OOM | `torch.cuda.empty_cache()` |

### Open Questions

1. **Q1 — Stain vectors**: Standard Ruifrok fixed matrix (current) vs. Macenko per-slide estimation? Standard is simpler and reproducible; Macenko handles cross-lab variation.
2. **Q2 — Dataset balance**: Class distribution unknown yet → may need class-weighted loss or oversampling.
3. **Q3 — Venue**: MICCAI/MedIA/TMI (medical) vs. CVPR workshop (vision)? Affects framing.
4. **Q4 — Spatial bias strategy**: S1 learnable additive (implemented) vs. S2 fixed Gaussian vs. S3 hard mask? S1 recommended; S2/S3 are ablations.

---

## 14. Ablation Plan

### Core architecture ablation (A-series)

| ID | Experiment | Proves |
|:---|:---|:---|
| A1 | Baseline ViT-B/16 (RGB) | Baseline to beat |
| A2 | H-only → ViT → classify | Morphology alone is insufficient |
| A3 | DAB-only → ViT → classify | HER2 signal alone lacks context |
| A4 | [H, DAB, 0] as 3-channel input, single ViT | Stain separation helps even without dual-stream |
| A5 | Dual-stream, no cross-attention, CLS concat | Dual-stream alone (no interaction) |
| A6 | **Full DSCA-ViT** | Complete architecture |

### Cross-attention ablation (B-series)

| ID | Experiment | Proves |
|:---|:---|:---|
| B1 | Unidirectional H←DAB only | Whether bidirectional is needed |
| B2 | Unidirectional DAB←H only | Which direction is more important |
| B3 | Bidirectional, CLS-only | Whether token-level attention matters |
| B4 | Bidirectional, no spatial bias (S0) | Whether spatial bias helps |
| B5 | Bidirectional, Gaussian prior (S2) | Fixed vs. learnable bias |
| B6 | Bidirectional, hard mask (S3, window=3) | Whether strict locality helps |

### Placement ablation (C-series)

| ID | After Layer | Proves |
|:---|:---|:---|
| C1 | 4 | Early fusion |
| C2 | 6 | Mid fusion |
| C3 | 9 (default) | Late-mid fusion |
| C4 | 11 | Very late fusion |

### Refinement ablation (D-series)

| ID | Experiment | Proves |
|:---|:---|:---|
| D1 | No refinement block | Whether post-fusion reasoning helps |
| D2 | 1 refinement block (default) | Proposed design |

**Priority order**: A1–A6 (architecture) → B1–B6 (attention) → C1–C4 (placement) → D1–D2 (fusion).

---

## 15. Reproducing the Project

### Environment

- Python 3.11
- PyTorch 2.x
- timm (`uv pip install timm`)
- Google Colab (GPU, T4)

### Run in Colab

1. **Sanity check** (no dataset needed, `pretrained=False`):
   ```
   notebooks/sanity_check.py
   ```

2. **Deconvolution sanity check** (downloads dataset, 20-patch visualization):
   ```
   notebooks/deconv_sanity_check.py   # auto-discovers dataset, per-class stats
   notebooks/visualize.py             # auto-downloads dataset, 20-patch grid
   ```
   *Verify: H shows nuclei (blue), DAB shows brown membrane signal. If not → stain vectors miscalibrated, do NOT train.*

3. **Training**:
   ```
   notebooks/train.py   # Cell 1-13: mount drive → clone repo → download data → Stage 1 → Stage 2 → evaluate
   ```

### Notebook conversion

The `.py` files use cell markers (`# ====...====` / `# Cell N — Title`) and can be converted to `.ipynb`:
```bash
python convert_to_ipynb.py
```

---

## 16. Future Work & Extensions

The current implementation is the **baseline version** of DSCA-ViT — architecture, training, and verification are working. These are the natural next steps, each motivated by a specific limitation or open question.

### 16.1 Stain Normalization — Macenko Method

**Problem**: The fixed Ruifrok stain matrix is calibrated for a specific lab's staining protocol. Real-world slides vary across antibody lots, scanners, and labs, which can corrupt the H/DAB separation.

**Options**:
| Approach | Pros | Cons |
|:---|:---|:---|
| Keep fixed Ruifrok (current) | Simple, reproducible | Fails if stain vectors drift |
| Macenko per-slide estimation | Adapts to each slide's actual stains | Adds preprocessing cost; per-slide variance |
| Learnable stain matrix | Can adapt end-to-end | Breaks the "fixed physics" story |

**Priority**: Moderate — only needed if the deconvolution sanity check reveals poor separation on real slides.

### 16.2 Ordinal Regression Loss

**Problem**: HER2 scores are **ordinal** (0 < 1+ < 2+ < 3+), but cross-entropy treats them as independent categories.

**Options**:
- Rank-consistent ordinal regression (CORN)
- Ordinal encoding + binary classifiers per threshold
- Custom penalty for distant misclassifications (e.g., predicting 0 when truth is 3+ should cost more than 0 vs 1+)

**Priority**: High — this directly encodes clinical severity ordering.

### 16.3 Multi-Magnification Input

**Problem**: The current model sees only 224×224 patches at a single magnification. Pathologists use whole-slide context (tumor architecture, heterogeneity).

**Options**:
- Multi-scale patch sampling (e.g., 224 at 20x + 224 at 40x)
- ViT with multi-resolution tokens
- Additional global context token from a low-magnification view

**Priority**: Low — requires dataset restructuring.

### 16.4 Stain-Specific Adapters (LoRA)

**Problem**: The shared encoder must serve both H and DAB streams, which have different input distributions. A single weight set might not be optimal for both.

**Options**:
- Per-stream LoRA adapters on the shared encoder
- Per-stream normalization layers after the 1→3 projections
- Separate small adapter heads per stain before fusion

**Priority**: Low — this was explicitly removed from the initial design to keep the contribution clean; it's a natural follow-up paper.

### 16.5 Attention Rollout Analysis

**Problem**: The cross-attention maps and gate values are stored, but not yet aggregated into a global interpretability picture.

**Options**:
- Attention rollout: propagate attention weights through layers to see which input patches drive the CLS decision
- Gate-value heatmap aggregation: average gates across heads to identify H-dominant vs DAB-dominant regions per class
- Per-class attention prototypes: average attention maps per HER2 score

**Priority**: High for the paper — this is the "built-in interpretability" selling point (gate values = which stain drives classification).

### 16.6 Multi-Depth Cross-Attention

**Problem**: The cross-attention is inserted at a single depth (after block 9). Information exchange might benefit from happening at multiple levels.

**Options**:
- Cross-attention after blocks 4, 9, and 12
- Progressive fusion with growing interaction

**Priority**: Low — this is ablation C-series (placement); multi-depth is a natural extension.

### 16.7 Full Ablation Suite

**Problem**: The paper needs the ablation experiments to justify each design choice.

**Priority**: Very high — A-series first (architecture), then B-series (attention), then C/D (placement/fusion). See Section 14 for the full table.

### 16.8 Summary of Priorities

| Priority | Extension | Why |
|:---|:---|:---|
| 🔴 Very High | Full ablation suite (A→B→C→D) | Required for publication |
| 🔴 Very High | Attention rollout + gate visualizations | The interpretability story |
| 🟠 High | Ordinal regression loss | Encodes clinical severity |
| 🟡 Moderate | Macenko stain normalization | Robustness across labs |
| 🟢 Low | Multi-magnification, adapters, multi-depth | Follow-up work |

---

## Appendix: Complete Tensor Flow

```
Step 0: Input
        x_rgb ∈ ℝ^{B × 3 × 224 × 224}

Step 1: Color Deconvolution (fixed, no gradient)
        OD = -log₁₀(x_rgb / 255 + ε)         ∈ ℝ^{B × 3 × 224 × 224}
        [H, DAB, Res] = M_stain⁻¹ · OD
        x_H ∈ ℝ^{B × 1 × 224 × 224}, x_D ∈ ℝ^{B × 1 × 224 × 224}

Step 2: Learnable Channel Projection (separate weights per stain)
        x_H' = Proj_H(x_H) ∈ ℝ^{B × 3 × 224 × 224}   Conv2d(1, 3, 1×1)
        x_D' = Proj_D(x_D) ∈ ℝ^{B × 3 × 224 × 224}   Conv2d(1, 3, 1×1)

Step 3: Shared Patch Embedding + Positional Encoding
        t_H = PatchEmbed(x_H') + pos_embed  ∈ ℝ^{B × 197 × 768}
        t_D = PatchEmbed(x_D') + pos_embed  ∈ ℝ^{B × 197 × 768}

Step 4: Shared Encoder Blocks 1–9 (batched)
        stacked = cat([t_H, t_D], dim=0)     ∈ ℝ^{2B × 197 × 768}
        H₉, D₉ = stacked.split(B)

Step 5: Spatially-Biased Bidirectional Cross-Attention (parallel)
        Ĥ = H₉ + MHA(Q=LN(H₉), K=LN(D₉), V=LN(D₉), bias=B)
        D̂ = D₉ + MHA(Q=LN(D₉), K=LN(H₉), V=LN(H₉), bias=B)
        Ĥ, D̂ each ∈ ℝ^{B × 197 × 768}

Step 6: Shared Encoder Blocks 10–12 (batched)
        stacked = cat([Ĥ, D̂], dim=0)  →  split → H_f, D_f

Step 7: Gated Token Fusion
        F_0 = Linear([CLS_H ‖ CLS_D])            (B, 768)
        g_i = σ(Linear([H_i ‖ D_i]))             (B, 196, 768)
        F_i = g_i ⊙ H_i + (1-g_i) ⊙ D_i
        F   = stack([F_0, F_1, ..., F_196])      (B, 197, 768)

Step 8: Refinement Block (1 transformer block, from scratch)
        F_ref = RefinementBlock(F)               (B, 197, 768)

Step 9: Classification
        cls = F_ref[:, 0, :]; gap = mean(F_ref[:, 1:, :])
        z   = [cls ‖ gap]                        (B, 1536)
        logits = ClassHead(z)                    (B, 4)
```

---

*Document generated from the actual codebase (`models/`, `datasets/`, `utils/`, `notebooks/`, `configs/`) and the design documents (`doc/implementation_plan.md`, `doc/walkthrough.md`).*