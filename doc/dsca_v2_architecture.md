# DSCA-ViT v2 — Architecture & Implementation

## Overview

DSCA-ViT v2 is an experimental successor to the original DSCA-ViT, designed to improve the weak interfaces around the existing architecture:

```text
stain representation
        ↓
ViT-compatible representation
        ↓
H/DAB interaction
        ↓
fusion
```

The **original `models/` implementation is untouched** and remains runnable. v2 lives in an independent package `models_v2/`.

**Objective:** beat the plain ViT baseline (**95.02%**) and the original DSCA-ViT (**~92.26%**) on the same official test split.

---

## High-level architecture

```
                         RGB IMAGE
                            │
                            ▼
                  Fixed Color Deconvolution
                       [no gradients]
                            │
                     ┌──────┴──────┐
                     │             │
                     ▼             ▼
                    H            DAB
                 [B,1,H,W]    [B,1,H,W]
                     │             │
                     ▼             ▼
              Learnable GN_H   Learnable GN_DAB
              (GroupNorm(1,1)) (GroupNorm(1,1))
                     │             │
                     ▼             ▼
              Adapter_H       Adapter_DAB
                1→32→3          1→32→3
                     │             │
                     ▼             ▼
        LearnableChannelAffine_H  LearnableChannelAffine_DAB
              (scale/bias)        (scale/bias)
                     │             │
                     ▼             ▼
              Shared ViT backbone
                  H branch / DAB branch
                     │             │
                     ▼             ▼
                 H tokens       DAB tokens
               [B,197,768]    [B,197,768]
                     │             │
                     └──────┬──────┘
                     ┌──────┴──────┐
                     ▼             ▼
                    H̃             D̃
                     │             │
              ┌──────┘             └──────┐
              ▼                            ▼
        D→H interaction              H→D interaction
              │                            │
              ▼                            ▼
           ΔH (zero-init)             ΔD (zero-init)
              │                            │
              ▼                            ▼
          H_e = H̃ + ΔH             D_e = D̃ + ΔD
              │                            │
              └────────────┬───────────────┘
                           ▼
                    Adaptive Gate
                    G = σ(gate_mlp([H_e,D_e]))
                    [B,197,768]
                           │
                           ▼
              F = G⊙H_e + (1−G)⊙D_e   [B,197,768]
                           │
                           ▼
                  Existing Refinement
                           │
                           ▼
                    Existing Classifier
                           │
                           ▼
                        Logits [B,4]
```

---

## Components

### Preserved unchanged (copied into `models_v2/`)

| Component | Source | Notes |
|---|---|---|
| `ColorDeconvolution` | `models/color_deconv.py` | Fixed Ruifrok, returns `(h,d)` `[B,1,H,W]` |
| `SharedViTEncoder` | `models/shared_vit.py` | `embed_dim=768`, `num_tokens=197`, `embed()/forward_before()/forward_after()` |
| `BidirectionalCrossAttention` + `SpatialBiasMatrix` | `models/cross_attention.py` | Returns `(h_out,d_out)` `[B,197,768]`; bias added pre-softmax |
| `RefinementBlock` | `models/fusion.py` | Copied into `fusion_v2.py` |
| `ClassificationHead` | `models/fusion.py` | Copied into `fusion_v2.py` |

### New components (`models_v2/input_adapters.py`, `models_v2/fusion_v2.py`)

**`StainNorm1ch`** — per-stain learnable spatial normalization:
```python
GroupNorm(num_groups=1, num_channels=1, affine=True)
```
- Trainable scale + bias, batch-independent, separate instance per stream (H and DAB), no sharing.
- Per-image spatial normalization (**not** ImageNet normalization).

**`StainAdapter`** — nonlinear 1→32→3 stain adapter:
```python
Conv2d(1, 32, kernel_size=3, padding=1) → GELU → Conv2d(32, 3, kernel_size=3, padding=1)
```
- Init: `conv1` kaiming_normal (bias 0); `conv2` kaiming_normal then `weight *= adapter_final_scale` (default 0.1), bias 0.
- The adapter output is a **meaningful 1→3 projection** — NOT near-zero (ViT receives real input at init).
- Separate instance per stream, no sharing.

**`LearnableChannelAffine`** — per-channel learnable scale/bias on `[B,3,H,W]`:
```python
y = x * scale + bias
scale = ones(1,3,1,1); bias = zeros(1,3,1,1)   # identity at init
```
- Replaces the originally considered `GroupNorm(3,3)`.
- **Why:** GroupNorm(3,3) normalizes each channel over its full spatial field, removing **absolute intensity information** (e.g., weak DAB ≈ 0.15 vs strong DAB ≈ 0.80 become similar). HER2/DAB intensity carries biological signal.
- Learnable channel affine preserves spatial/intensity structure while giving the ViT input interface genuine learnable adaptation.
- Separate instance per stream, no sharing.

**`BidirectionalInteraction`** — explicit cross-stream enrichment:
```python
delta_h = interaction_d_to_h(torch.cat([h̃, d̃], dim=-1))   # Linear(1536→192)→GELU→Linear(192→768)
delta_d = interaction_h_to_d(torch.cat([d̃, h̃], dim=-1))   # Linear(1536→192)→GELU→Linear(192→768)
h_e = h̃ + delta_h
d_e = d̃ + delta_d
```
- **Both final Linear layers zero-initialized** → at init `ΔH=0`, `ΔD=0`, so `H_e=H̃`, `D_e=D̃`.
- Two independent MLPs, no sharing. `interaction_hidden_dim = embed_dim // 4 = 192`.

**`AdaptiveGate`** — token- and channel-wise gate:
```python
gate_input = torch.cat([h_e, d_e], dim=-1)   # [B,197,1536]
g = torch.sigmoid(gate_mlp(gate_input))      # Linear(1536→192)→GELU→Linear(192→768)
fused = g * h_e + (1 - g) * d_e              # [B,197,768]
```
- **`g.shape == [B,197,768]`** — per-image, per-token, per-channel (NOT `[B,768]` or `[B,197,1]`).
- Final gate Linear bias ≈ 0 → `sigmoid(0) = 0.5` → gate ≈ 0.5 at init.

---

## Parameter groups (exactly 5)

| Group | Params |
|---|---|
| `vit` | `encoder.parameters()` (150 tensors, 85,798,656 params) |
| `existing_dsca` | `cross_attention.parameters()` + `refinement.parameters()` (49 tensors, 21,305,497 params) |
| `input_modules` | `norm_h, norm_d, adapter_h, adapter_d, channel_affine_h, channel_affine_d` (16 tensors, 2,390 params) |
| `fusion_modules` | `interaction.parameters()` + `gate.parameters()` (12 tensors, 1,329,984 params) |
| `classifier` | `classifier.parameters()` (6 tensors, 1,186,564 params) |

- Validation: no parameter appears twice; every trainable parameter belongs to exactly one group. Raises an error otherwise.

---

## Compatibility loading

`load_original_weights(checkpoint_path)` loads the original DSCA checkpoint into v2:

- **Loaded old:** `encoder.*`, `cross_attention.*`, `refinement.*`, `classifier.*`
- **Expected fresh (new):** `norm_h.*`, `norm_d.*`, `adapter_h.*`, `adapter_d.*`, `channel_affine_h.*`, `channel_affine_d.*`, `interaction.*`, `gate.*`
- **Legacy ignored:** `proj_h.*`, `proj_d.*`, `fusion.*`
- **Assertion:** any missing parameter in a preserved module (`encoder`, `cross_attention`, `refinement`, `classifier`) raises an exception.

Checkpoint path: `/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT/Stage2/weights_DSCA_ViT.pth`

---

## Training protocol (3 stages)

- **One persistent AdamW** with 5 parameter groups, created once.
- Freeze/unfreeze via `requires_grad`; group LRs changed at stage transitions.
- **Per-stage cosine scheduler** — LR set first, then new `CosineAnnealingLR(T_max=stage_epochs)` per stage.
- Weight decay 0.05 on weights only; bias/norm → 0.
- Gradient clipping 1.0 (only params with gradients).
- No AMP. No ordinal loss / class weighting / focal loss.

| Stage | Freeze | Train (LR) | Epochs |
|---|---|---|---|
| 1 | vit, existing_dsca, fusion | input_modules (2e-4), classifier (1e-5) | 4 |
| 2 | vit | input_modules (1e-4), existing_dsca (5e-5), fusion_modules (2e-4), classifier (1e-4) | 6 |
| 3 | — | vit (1e-5), existing_dsca (5e-5), input_modules (1e-4), fusion_modules (1e-4), classifier (1e-4) | 20 |

---

## Data policy

- Train on official `train/`.
- **Deterministic stratified 10% validation holdout** from `train/` (seed 42, `stratify=labels`, indices saved to `split_indices.npz`).
- Model selection + best checkpoint on **validation**.
- Official `test/` evaluated **exactly once** at the end.
- Preserve the exact same preprocessing, label mapping, and augmentation as the baseline.

---

## Metrics & telemetry

**Metrics** (`utils/metrics_v2.py`): accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.

**Telemetry** (lightweight, inline in notebook):
- Per new module: `grad_norm`, `parameter_delta` (‖p−p0‖), `relative_delta`.
- Interaction output norms: `mean(|ΔH|)`, `mean(|ΔD|)`.
- Gate stats: mean/std/min/max/median.
- Gate/confidence correlation: `sample_gate = gate.mean(dim=(1,2))` vs predicted confidence (Pearson + Spearman).

---

## Checkpoints → `.../HER2_Checkpoints/DSCA_ViT_v2/`

`stage1_end.pt`, `stage2_end.pt`, `best_stage3.pt` (best on validation), `last.pt` — each with model/optimizer/scheduler state, epoch, stage, best val acc, resolved config, seed, split indices path.

---

## Files

```
models_v2/
├── __init__.py
├── dsca_vit_v2.py
├── input_adapters.py
├── fusion_v2.py
├── color_deconv.py
├── shared_vit.py
└── cross_attention.py

utils/
├── train_v2.py
└── metrics_v2.py

configs/
└── dsca_v2_config.yaml

notebooks/
└── 05_DSCA_ViT_v2_Training.py  (+ .ipynb)
```

---

## Local sanity checks (passed)

- Parameter-group validation (5 groups, no duplicates, all assigned).
- Forward shapes: RGB→H/DAB→adapter→ViT-in→tokens [B,197,768]→cross-attn→enriched→gate [B,197,768]→fused→refined→logits [B,4].
- `loss.backward()` OK; no NaN/Inf; all params float32.
- Interaction residuals zero at init (mean|ΔH|=0, mean|ΔD|=0).
- Gate ≈ 0.5 at init.
- Old `models/` DSCAViT still works.