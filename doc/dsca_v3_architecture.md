# DSCA-ViT v3 — Architecture & Implementation

## Overview

DSCA-ViT v3 is an experimental successor to DSCA-ViT v2, focused on the **generalization problem** observed in v2:

```text
DSCA-ViT v2:
    Validation accuracy : ~99.26%
    Official test acc   :  87.22%
    Validation→test gap : -12.04 percentage points
```

The primary objective of v3 is therefore **not** merely high validation accuracy, but:

1. **Increase** official test accuracy vs v2 (87.22%)
2. **Reduce** the validation → test gap vs v2 (−12.04 pp)

The **original `models/` package and `models_v2/` are completely untouched** and remain runnable. v3 lives in an independent package `models_v3/`.

**Baseline comparison table (official test split):**

```text
ViT baseline       = 95.02%
Original DSCA-ViT  = ~92.26%
DSCA-ViT v2        = 87.22%
DSCA-ViT v3        = new result  ← target: ≥ v2 and reduced gap
```

---

## High-level architecture

```
                       RGB IMAGE  [B,3,224,224]
                             │
        (training-only StainAugmentation applied in the
         DATASET transform pipeline — NOT inside the model)
                             │
                             ▼
                   Fixed Color Deconvolution
                        [no gradients]
                             │
                      ┌──────┴──────┐
                      │             │
                      ▼             ▼
                     H            DAB
                  [B,1,224,224]  [B,1,224,224]
                      │             │
                      ▼             ▼
               StainNorm1ch_H   StainNorm1ch_DAB
               (GroupNorm(1,1)) (GroupNorm(1,1))
                      │             │
                      ▼             ▼
                StainAdapter_H  StainAdapter_DAB
                   1→32→3          1→32→3
                      │             │
                      ▼             ▼
         LearnableChannelAffine_H  LearnableChannelAffine_DAB
               (scale/bias)        (scale/bias)
                      │             │
                ┌─────┴─────────────┴─────┐
                │                         │
        FINE VIEW (224)           COARSE VIEW (224→112→224)
        [B,3,224,224]            [B,3,224,224]  (zero params)
                │                         │
        ┌───────┴────────┐        ┌───────┴────────┐
        │ H_fine, D_fine │        │ H_coarse,D_coarse │
        └───────┬────────┘        └───────┬────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │   Shared ViT   │        │   Shared ViT   │   ← SAME instance
        │  embed+before  │        │  embed+before  │
        └───────┬────────┘        └───────┬────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │   CrossAttn    │        │   CrossAttn    │   ← SAME instance
        └───────┬────────┘        └───────┬────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │  Shared ViT   |        │  Shared ViT   |   ← SAME instance
        │   forward_after       │   forward_after
        └───────┬────────┘        └───────┬────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │ Interaction    │        │ Interaction    │   ← SAME instance
        │ (ΔH=ΔD=0 init) │        │ (ΔH=ΔD=0 init) │
        └───────┬────────┘        └───────┬────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │   StainGate    │        │   StainGate    │   ← SAME instance
        │  (≈0.5 init)   │        │  (≈0.5 init)   │
        └───────┬────────┘        └───────┬────────┘
                │                         │
             F_fine                    F_coarse
          [B,197,768]                [B,197,768]
                └───────────┬────────────┘
                            ▼
                      ScaleGate
                     (≈0.5 init)
                            │
                            ▼
            F = G_scale⊙F_fine + (1−G_scale)⊙F_coarse
                        [B,197,768]
                            │
                            ▼
                 Existing RefinementBlock
                            │
                            ▼
                  Existing ClassificationHead
                            │
                            ▼
                       Logits [B,4]
```

### Shared-module rule (locked)

The model contains **exactly one instance** of each shared module:

```python
self.encoder           # SharedViTEncoder
self.cross_attention   # BidirectionalCrossAttention
self.interaction       # BidirectionalInteraction
self.stain_gate        # StainGate
self.scale_gate        # ScaleGate
```

Each scale is processed **independently** through the same shared objects via `_process_scale()` (weights shared, computation separate):

```python
f_fine   = self._process_scale(h_fine_tokens,   d_fine_tokens)
f_coarse = self._process_scale(h_coarse_tokens, d_coarse_tokens)
fused    = self.scale_gate(f_fine, f_coarse)
```

`model.assert_single_shared_instances()` verifies exactly one instance of each type and that the same Python objects are wired into the forward path.

---

## Components

### Preserved unchanged (copied into `models_v3/`)

| Component | Source | Notes |
|---|---|---|
| `ColorDeconvolution` | `models_v2/color_deconv.py` | Fixed Ruifrok, returns `(h,d)` `[B,1,H,W]` |
| `SharedViTEncoder` | `models_v2/shared_vit.py` | `embed_dim=768`, `num_tokens=197`, `embed()/forward_before()/forward_after()` |
| `BidirectionalCrossAttention` + `SpatialBiasMatrix` | `models_v2/cross_attention.py` | Returns `(h_out,d_out)` `[B,197,768]`; bias added pre-softmax |
| `RefinementBlock` | `models_v2/fusion_v2.py` | Copied into `fusion_v3.py` |
| `ClassificationHead` | `models_v2/fusion_v2.py` | Copied into `fusion_v3.py` |
| `StainNorm1ch` | `models_v2/input_adapters.py` | Copied into `input_adapters_v3.py` |
| `StainAdapter` | `models_v2/input_adapters.py` | Copied into `input_adapters_v3.py` |
| `LearnableChannelAffine` | `models_v2/input_adapters.py` | Copied into `input_adapters_v3.py` |

### New v3 components

**`StainAugmentation`** (`models_v3/stain_augmentation.py`) — **training-only** stain-domain augmentation.

- Implemented as a **torchvision-style transform** in the **dataset pipeline** (NOT inside the model), so train/eval separation is explicit and auditable.
- Validation/test pipelines never include it.
- Zero trainable parameters.
- Configurable in YAML:

```yaml
stain_augmentation:
  enabled: true
  probability: 0.5
  concentration_range: [0.85, 1.15]   # H and DAB OD concentration multipliers
  brightness_range: [0.9, 1.1]        # RGB brightness, pre-deconvolution
  contrast_range: [0.9, 1.1]          # RGB contrast, pre-deconvolution
```

- Perturbs RGB brightness/contrast (pre-deconvolution) and H/DAB **stain concentration in OD space** (via the same Ruifrok matrix) — physically plausible histopathology variation. DAB information is never randomly destroyed (min multiplier 0.85).

**`CoarseScaleView`** (`models_v3/multiscale_v3.py`) — zero-parameter coarse view:

```python
224×224 → bilinear downsample → 112×112 → bilinear upsample → 224×224
```

- **This is NOT a larger spatial context and NOT a change of magnification.** The physical field of view remains the same; the coarse branch is a low-frequency / coarse multi-resolution view of the same 224×224 field.
- Zero trainable parameters.
- Both views pass through the **same** `SharedViTEncoder` (no duplicated ViT parameters).

**`StainGate`** (`models_v3/fusion_v3.py`) — token/channel-wise H vs DAB gate (v3 naming of the v2 `AdaptiveGate`):

```python
g_stain = σ(gate_mlp([H_e, D_e]))        # Linear(1536→192)→GELU→Linear(192→768)→Sigmoid
F_stain = g_stain ⊙ H_e + (1−g_stain) ⊙ D_e   # [B,197,768]
```

- Final Linear zero-initialized → `G_stain ≈ 0.5` at init (balanced start).
- **One shared instance** reused independently for fine and coarse branches.

**`ScaleGate`** (`models_v3/fusion_v3.py`) — NEW in v3, token/channel-wise fine vs coarse gate:

```python
g_scale = σ(scale_mlp([F_fine, F_coarse]))     # Linear(1536→192)→GELU→Linear(192→768)→Sigmoid
F       = g_scale ⊙ F_fine + (1−g_scale) ⊙ F_coarse   # [B,197,768]
```

- Final Linear zero-initialized → `G_scale ≈ 0.5` at init (balanced fine/coarse fusion; does not strongly prefer either scale).
- Answers per token/channel: *"rely on fine cellular information or coarse contextual information?"*

**`BidirectionalInteraction`** — unchanged from v2:

```python
delta_h = interaction_d_to_h(cat([h_tilde, d_tilde]))   # Linear(1536→192)→GELU→Linear(192→768)
delta_d = interaction_h_to_d(cat([d_tilde, h_tilde]))
h_e = h_tilde + delta_h
d_e = d_tilde + delta_d
```

- Final Linear zero-initialized → `ΔH = ΔD = 0` at init → `H_e = H̃`, `D_e = D̃` (pretrained representation preserved at init).
- **One shared instance** reused independently for fine and coarse branches.

---

## Parameter groups (exactly 5)

| Group | Params |
|---|---|
| `vit` | `encoder.parameters()` (shared ViT) |
| `existing_dsca` | `cross_attention.parameters()` + `refinement.parameters()` |
| `input_modules` | `norm_h, norm_dab, adapter_h, adapter_dab, channel_affine_h, channel_affine_dab` |
| `fusion_modules` | `interaction.parameters()` + `stain_gate.parameters()` + `scale_gate.parameters()` |
| `classifier` | `classifier.parameters()` |

- `CoarseScaleView` and `StainAugmentation` have **zero parameters** → not part of any group.
- Validation: no parameter appears twice; **every** model parameter (checked against `self.parameters()`, NOT requires_grad-filtered) belongs to exactly one group. Raises an error otherwise.

---

## Compatibility loading (v2 → v3)

`load_v2_weights(checkpoint_path, device)`:

- **Loaded preserved (from v2):** `encoder.*`, `cross_attention.*`, `refinement.*`, `classifier.*` — must ALL be present.
- **Fresh v3 (never loaded from v2):** `norm_h.*`, `norm_dab.*`, `adapter_h.*`, `adapter_dab.*`, `channel_affine_h.*`, `channel_affine_dab.*`, `coarse.*`, `interaction.*`, `stain_gate.*`, `scale_gate.*`.
- **Legacy ignored:** `proj_h.*`, `proj_d.*`, `fusion.*`.
- **Assertion:** any missing preserved parameter raises an exception (does NOT silently ignore).

The loader prints an explicit report:

```text
=== V3 COMPATIBILITY LOAD ===
Loaded preserved parameters (from v2):
  encoder.*             ✓ (N tensors)
  cross_attention.*     ✓ (N tensors)
  refinement.*          ✓ (N tensors)
  classifier.*          ✓ (N tensors)
Fresh v3 parameters (not loaded): ...
Preserved parameters missing: NONE
```

Checkpoint path (config): `/content/drive/MyDrive/HER2_Checkpoints/DSCA_ViT_v2/best_stage3.pt`

---

## Training protocol (3 stages)

Same design as v2: one persistent AdamW, 5 parameter groups with name keys, freeze/unfreeze via `requires_grad`, per-stage `CosineAnnealingLR` (LRs set first, then new scheduler per stage, T_max = stage epochs), weight decay 0.05 on weights only, gradient clipping 1.0, no AMP.

| Stage | Freeze | Train (LR) | Epochs |
|---|---|---|---|
| 1 | vit, existing_dsca | input_modules (2e-4), fusion_modules (2e-4), classifier (1e-5) | 4 |
| 2 | vit | input_modules (1e-4), existing_dsca (5e-5), fusion_modules (2e-4), classifier (1e-4) | 6 |
| 3 | — | vit (1e-5), existing_dsca (5e-5), input_modules (1e-4), fusion_modules (1e-4), classifier (1e-4) | 20 |

- Loss: **CrossEntropyLoss only** (ordinal loss intentionally REMOVED from v3 — tested separately later if needed).
- No focal loss, class weighting, TTA, extra attention, or extra losses.

---

## Data policy

- Same official dataset (train/test), same label mapping (`class_0/1+/2+/3+`).
- Deterministic stratified 10% validation holdout from train only (seed 42, indices saved to `split_indices.npz`).
- Validation/test **non-augmented** (no TTA, no test-time stain augmentation).
- Official test evaluated **exactly once** at the end; model selection is validation-based only.
- StainAugmentation is applied **only** to the training split, inside the training transform pipeline.

---

## Metrics & telemetry

**Metrics** (`utils/metrics_v3.py`): accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrix. Reported at **every validation epoch** (per-class recall highlighted for generalization monitoring) and for the single final test evaluation.

**Telemetry** (`utils/train_v3.py`, lightweight):
- Per new module: `grad_norm`, `parameter_delta` (‖p−p0‖), `relative_delta`.
- Interaction output norms: `mean(|ΔH|)`, `mean(|ΔD|)`.
- StainGate stats: mean/std/min/max/median.
- ScaleGate stats: mean/std/min/max/median.
- Gate/confidence correlations for both gates: `sample_gate = gate.mean(dim=(1,2))` vs predicted confidence (Pearson + Spearman).

---

## Checkpoints → `.../HER2_Checkpoints/DSCA_ViT_v3/`

`stage1_end.pt`, `stage2_end.pt`, `best_stage3.pt` (best on validation), `last.pt` — each with model/optimizer/scheduler state, epoch, stage, best val acc, resolved config, seed, split indices path.

---

## Files

```
models_v3/
├── __init__.py
├── dsca_vit_v3.py
├── input_adapters_v3.py
├── multiscale_v3.py
├── fusion_v3.py
├── stain_augmentation.py
├── color_deconv.py
├── shared_vit.py
└── cross_attention.py

utils/
├── train_v3.py
└── metrics_v3.py

configs/
└── dsca_v3_config.yaml

notebooks/
└── 06_DSCA_ViT_v3_Training.py  (+ .ipynb)
```

`convert_to_ipynb.py` includes `06_DSCA_ViT_v3_Training.py` in its `py_files` list.

---

## Sanity checks (notebook Cells 6/7/9/10)

- Parameter-group validation (5 groups, no duplicates, all assigned).
- Single-instance verification: exactly one `SharedViTEncoder`, one `BidirectionalCrossAttention`, one `BidirectionalInteraction`, one `StainGate`, one `ScaleGate`.
- Forward shapes: RGB→H/DAB→adapters→fine/coarse→tokens [B,197,768]→cross-attn→interaction→stain gate→scale gate→fused [B,197,768]→refined→logits [B,4].
- `loss.backward()` OK; no NaN/Inf; all params float32, correct device.
- Interaction residuals zero at init (`mean|ΔH|=0`, `mean|ΔD|=0`).
- Stain gate ≈ 0.5 at init; scale gate ≈ 0.5 at init.
- Fine and coarse representations have the same shape `[B,197,768]` and are **not exactly identical** (no minimum-difference threshold imposed).
- `models/` and `models_v2/` still import.