# DSCA-ViT v1 — Architecture & Implementation

## Overview

DSCA-ViT v1 is the **original** Dual-Stain Cross-Attention Vision Transformer for HER2 IHC scoring. It is the first architecture in the project: it explicitly separates Hematoxylin (morphology) and DAB (HER2 protein expression) stains from RGB images before processing them through a **shared Vision Transformer** with **spatially-biased bidirectional cross-attention** and **gated token fusion**.

```text
stain separation (fixed, physics-based)
        ↓
ViT-compatible representation (1→3 projections)
        ↓
H/DAB interaction (spatially-biased cross-attention)
        ↓
fusion (gated, per-token)
        ↓
classification (CLS + GAP)
```

It lives in the **`models/`** package and is the reference the successor experiments (v2 `models_v2/`, v3 `models_v3/`) build upon. The isolation rule applies: `models/` is never modified by later versions.

**Measured result:** original DSCA-ViT (**~92.26%**) did **not** beat the plain ViT-B/16 baseline (**95.02%**) on the official test split — this motivated v2 (`doc/dsca_v2_architecture.md`).

---

## High-level architecture

```
                       RGB IMAGE [B,3,224,224]
                              │
                              ▼
                 Fixed Color Deconvolution
                   ColorDeconvolution()
                      [no gradients]
                              │
                      ┌───────┴────────┐
                      │                │
                      ▼                ▼
                     H [B,1,H,W]    DAB [B,1,H,W]
                      │                │
                      ▼                ▼
                  proj_h 1→3       proj_d 1→3
                 (Conv2d(1,3,1×1))  (Conv2d(1,3,1×1))
                      │                │
                      └───────┬────────┘
                              │
                      ┌───────▼────────┐
                      │ Shared ViT-B/16 │
                      │ embed() + blocks 1–9 │
                      │ (100% shared,    │
                      │  batched 2B)     │
                      └───────┬────────┘
                      ┌───────┴────────┐
                      │                │
                      ▼                ▼
                 H tokens [B,197,768]   D tokens [B,197,768]
                      │                │
                      └───────┬────────┘
                      ┌───────▼─────────────────────────┐
                      │ SPATIALLY-BIASED BIDIRECTIONAL  │
                      │ CROSS-ATTENTION                  │
                      │ (bias B[197,197], β=1, γ=0.1)    │
                      └───────┬─────────────────────────┘
                      ┌───────┴────────┐
                      │                │
                      ▼                ▼
                 Ĥ tokens          D̂ tokens
                      │                │
                      └───────┬────────┘
                      ┌───────▼────────┐
                      │ Shared ViT-B/16 │
                      │ blocks 10–12    │
                      │ + final norm    │
                      └───────┬────────┘
                      ┌───────┴────────┐
                      │                │
                      ▼                ▼
                 H_final [B,197,768]   D_final [B,197,768]
                      │                │
                      └───────┬────────┘
                      ┌───────▼────────┐
                      │ GATED FUSION    │
                      │ F = g⊙H + (1−g)⊙D │
                      │ g = σ(Linear([H,D])) │
                      └───────┬────────┘
                              │  [B,197,768]
                              ▼
                   Refinement Block [B,197,768]
                              │
                              ▼
                   Classification Head [B,4]
                              │
                              ▼
                       HER2 Score {0, 1+, 2+, 3+}
```

---

## Components

### 5.1 Color Deconvolution — `models/color_deconv.py`

- **`ColorDeconvolution`** — fixed Ruifrok & Johnston H-DAB matrix:
  ```
  H:   [0.6500286, 0.7040310, 0.2860126]
  DAB: [0.2688606, 0.5700937, 0.7767574]
  Res: [0.7110272, 0.4234194, 0.5615672]
  ```
  - `OD = -log10(x_rgb + ε)`, ε = 1e-6.
  - `stains = OD @ M_inv` via `torch.linalg.inv` registered as a **buffer** (auto device move, never updated).
  - Returns `(h_channel, d_channel)` `[B,1,H,W]`, each **clamped to ≥ 0** (negative OD is non-physical).
  - Wrapped in `torch.no_grad()` in `dsca_vit.py` — no gradient flows through deconv.
  - `deconvolve_numpy()` — numpy variant for visualization/debugging.

### 5.2 Stain Channel Projection — `models/shared_vit.py`

- **`StainChannelProjection`** — `Conv2d(1, 3, kernel_size=1)` maps single-channel stain → 3-channel pseudo-RGB so the pretrained ViT patch embedding can be reused unchanged.
  - `init_mode="repeat"` (default): weight ≈ ones, bias ≈ zero → output ≈ `[x, x, x]` (identity-like, preserves pretrained ViT behavior early).
  - `init_mode="xavier"`: Xavier uniform weight, zero bias.
  - Separate instance per stream (`proj_h`, `proj_d`), **no sharing**. 6 params each.

### 5.3 Shared ViT Encoder — `models/shared_vit.py`

- **`SharedViTEncoder`** — wraps `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)`:
  - Extracts `patch_embed`, `cls_token [1,1,768]`, `pos_embed [1,197,768]`, `pos_drop`.
  - Splits the 12 blocks at `split_after=9` → `blocks_before` (blocks 1–9) and `blocks_after` (blocks 10–12), plus final `norm`.
  - `embed(x)` — patch embed + CLS + pos-embed + pos-drop → `[B,197,768]`.
  - `forward_before(x)` — blocks 1–9.
  - `forward_after(x)` — blocks 10–12 **+ final LayerNorm**.
  - `embed_dim=768`, `num_tokens=197`.
  - **Critical:** both H and D streams use the *same* parameter tensors — weight sharing is real, not simulated.
  - **Batched forward trick:** `cat([h_tokens, d_tokens], dim=0)` → one `[2B,197,768]` GPU call → ~40–60% wall-clock overhead instead of 2×.

### 5.4 Spatially-Biased Bidirectional Cross-Attention — `models/cross_attention.py`

- **`SpatialBiasMatrix`** — learnable `[197,197]` parameter initialized:
  ```
  B[i,i] = +β (default 1.0)
  B[i,j] = -γ · distance(i,j)  (default γ = 0.1)
  B[0,:] = B[:,0] = 0          (CLS has no spatial bias)
  ```
  - Distance = Euclidean distance between patch grid positions (14×14 grid).
  - `γ=0.1` → max penalty `-1.84` (soft locality prior, **not** a hard mask; `γ>0.3` would effectively hard-mask).
  - Prints initialization values at construction:
    ```
    Spatial Bias Initialization
    ---------------------------
    beta         : 1.0
    gamma        : 0.1
    max distance : 18.38
    max penalty  : -1.84
    ```
- **`CrossAttentionLayer`** — single-direction cross-attention, pre-norm:
  ```
  Q = W_q(LN(source)); K = W_k(LN(context)); V = W_v(LN(context))
  logits = QKᵀ/√d_k + B   (B added pre-softmax)
  out = source + W_o(attn @ V)
  ```
  - 12 heads, head_dim 64, stores `self.attn_weights` (detached) for visualization.
- **`BidirectionalCrossAttention`** — symmetric exchange computed in parallel from the *original* tokens (no ordering bias), each direction with its own `CrossAttentionLayer` + `CrossAttentionFFN` (GELU MLP, hidden 3072).
  - `embed_dim=768`, `num_heads=12`, `ffn_hidden_dim=3072`, `num_tokens=197`.

### 5.5 Gated Token Fusion — `models/fusion.py`

- **`GatedFusion`**:
  ```python
  # CLS token (index 0):
  fused_cls  = Linear(1536→768)([CLS_H || CLS_D])

  # Patch tokens (indices 1..196):
  g_i        = sigmoid(Linear(1536→768)([H_i || D_i]))     # [B,196,768]
  fused_pi   = g_i ⊙ H_i + (1 − g_i) ⊙ D_i
  ```
  - Gate is per-image, per-token, **per-channel** `[B,196,768]`.
  - Biologically interpretable: `g≈1` → morphology (H); `g≈0` → HER2 signal (DAB); `g≈0.5` → both.
  - Gate values stored as `_last_gate_values`, retrievable with `model.get_gate_values()`.

### 5.6 Refinement Block — `models/fusion.py`

- **`RefinementBlock`** — one standard pre-norm transformer block: `LayerNorm → MultiheadAttention(768, 12 heads) → residual`, then `LayerNorm → MLP(768→3072→768, GELU) → residual`.
  - Xavier-initialized (trained from scratch — fused stain representations have no ImageNet counterpart).
  - Lets the network reason over the fused representation (aggregate evidence across the patch).

### 5.7 Classification Head — `models/fusion.py`

- **`ClassificationHead`** — dual pooling:
  ```python
  cls = tokens[:, 0, :]            # CLS token [B,768]
  gap = tokens[:, 1:, :].mean(1)   # GAP of patches [B,768]
  z   = LN(concat([cls, gap]))     # [B,1536]
  logits = Linear(1536→768) → GELU → Dropout(0.1) → Linear(768→4)
  ```

### 5.8 Assembly — `models/dsca_vit.py`

- **`DSCAViT`** — the top-level module implementing the 9-step forward pass:
  `ColorDeconvolution → proj_h/proj_d → encoder.embed → forward_before → BidirectionalCrossAttention → forward_after → GatedFusion → RefinementBlock → ClassificationHead`.
- Constructor args: `num_classes=4, pretrained=True, split_after=9, proj_init="repeat", spatial_bias_beta=1.0, spatial_bias_gamma=0.1, classifier_dropout=0.1`.
- `get_gate_values()` — returns gates from last forward `[B,196,768]` (None if not run).
- `get_parameter_groups()` — returns 2 groups (see below).
- `count_parameters()` — per-component parameter counts.

---

## Parameter groups (exactly 2)

| Group | Params |
|---|---|
| `encoder` | `encoder.parameters()` — the shared ViT-B/16 (85,798,656) |
| `new` | `proj_h` + `proj_d` + `cross_attention` + `fusion` + `refinement` + `classifier` (24,853,031) |

### Parameter budget (~110.6M total, verified output)

| Component | Params | Share | Notes |
|---|---:|---:|---|
| `encoder` | 85,798,656 | 78% | Pretrained ImageNet, the heavy lifting |
| `cross_attention` | 14,217,625 | 13% | The novel contribution (bias + QKV + FFN ×2) |
| `refinement` | 7,087,872 | 6% | Post-fusion reasoning, from scratch |
| `fusion` | 2,360,832 | 2% | Lightweight, interpretable |
| `classifier` | 1,186,564 | 1% | CLS + GAP dual pooling |
| `proj_h` / `proj_d` | 6 / 6 | ~0% | Negligible |
| `color_deconv` | 0 | 0% | Fixed buffer, no learnable params |
| **total** | **110,651,561** | | |
| **trainable** | **110,651,561** | | (all trainable at init) |

Only **~22%** of parameters are new — the rest are pretrained.

---

## Training protocol (2 stages)

| Hyperparameter | Stage 1 | Stage 2 |
|---|---|---|
| Model | DSCA-ViT (ViT-B/16 backbone) | Same |
| Split point | `split_after=9` | Same |
| Spatial bias β / γ | 1.0 / 0.1 | Same |
| Batch size | 32 (drop to 16 if OOM) | 32 |
| Epochs | 30 | 30 |
| Optimizer | Adam | Adam |
| Encoder LR | **0 (frozen)** | 1e-5 |
| New components LR | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=30) | CosineAnnealingLR (T_max=30) |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Seed | 42 | 42 |
| Augmentation | Resize(224), HFlip, VFlip, Rotate(10°) | (val: Resize only) |

- **Stage 1** — freezes all encoder params, trains only `param_groups["new"]` at 1e-4. New components learn to interpret frozen pretrained features.
- **Stage 2** — loads best Stage 1 checkpoint, unfreezes everything, discriminative LRs (encoder 1e-5, new 1e-4).
- **Input normalization:** NO ImageNet Normalize (would corrupt Beer-Lambert OD computation); transforms are `Resize(224) → ToTensor()` (val) + HFlip/VFlip/Rotate(10°) (train). Augmentation applied on **RGB before deconvolution** so both stains see the identical geometric transform (preserves spatial correspondence).

---

## Data policy

- Official `train/` split; a validation portion held out from it (8,093 train / 1,847 val used in the actual run).
- Official `test/` split evaluated at the end (1,847 images: class_0 658, class_1+ 316, class_2+ 111, class_3+ 762).
- No ImageNet normalization; values in `[0,1]` via `ToTensor()`.
- `class_2+` (equivocal) is severely underrepresented (523 train / 111 test) — a known risk.

---

## Results & comparison

| Model | Official test accuracy |
|---|---|
| Plain ViT-B/16 baseline | **95.02%** |
| **DSCA-ViT v1 (original)** | **~92.26%** |

The original design underperformed the baseline. The v2 and v3 experiments (`doc/dsca_v2_architecture.md`, `doc/dsca_v3_architecture.md`) were created to close this gap by fixing the weak interfaces (stain input adapters, explicit interaction, gated fusion).

---

## Metrics & telemetry

- **Metrics** (`utils/metrics.py`): accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrix (via `sklearn`).
- **Telemetry:** `CrossAttentionLayer.attn_weights` (detached) for attention visualization; `DSCAViT.get_gate_values()` for fusion-gate heatmaps.
- **Visualizations:** `notebooks/visualize.py` (deconv sanity grid + gate/attention plots), `notebooks/deconv_sanity_check.py` (20-patch verification).

---

## Checkpoints

- `.../HER2_Checkpoints/DSCA_ViT/Stage1/` — best Stage 1 (`best_stage1_DSCA_ViT.pth`).
- `.../HER2_Checkpoints/DSCA_ViT/Stage2/` — best Stage 2 (`best_stage2_DSCA_ViT.pth`) + weights-only `weights_DSCA_ViT.pth` (this path is used by v2 for compatibility loading).
- `utils/checkpoint.py` — `save_checkpoint`, `load_checkpoint`.

---

## Files

```
models/
├── __init__.py            # Package exports
├── dsca_vit.py            # DSCAViT assembly (9-step forward)
├── shared_vit.py          # SharedViTEncoder + StainChannelProjection
├── color_deconv.py        # ColorDeconvolution (fixed Ruifrok) + deconvolve_numpy
├── cross_attention.py     # SpatialBiasMatrix + CrossAttentionLayer + BidirectionalCrossAttention
└── fusion.py              # GatedFusion + RefinementBlock + ClassificationHead

utils/
├── train.py               # train_one_epoch (calls model.train())
├── evaluate.py            # validate_one_epoch (calls model.eval())
├── metrics.py             # compute_metrics, print_metrics (sklearn)
└── checkpoint.py          # save_checkpoint, load_checkpoint

configs/
└── dsca_vit_b16.yaml      # Hyperparameters (reference)

notebooks/
├── train.py               # Colab 2-stage training
├── sanity_check.py        # 8 module unit tests
├── visualize.py           # Deconv + gate/attention visualization
└── deconv_sanity_check.py # Dedicated 20-patch deconv verification
```

---

## Verification & tests (all passed)

| Test | Result |
|---|---|
| `ColorDeconvolution` | `(B,1,224,224)`, non-negative, 0 learnable params |
| `StainChannelProjection` | `(B,3,224,224)` |
| `SharedViTEncoder` | embed/before/after all `(B,197,768)` |
| `BidirectionalCrossAttention` | H/D out `(B,197,768)`, attn_weights stored |
| `GatedFusion` | fused `(B,197,768)`, gates `(B,196,768)` in [0,1] |
| `RefinementBlock` | `(B,197,768)` |
| `ClassificationHead` | logits `(B,4)` |
| Full `DSCAViT` end-to-end | logits `(B,4)`, gates `(B,196,768)` |

- ✅ Spatial bias: diagonal +1.0, CLS row/col 0, max penalty −1.84 (soft prior).
- ✅ Gradient flow through 1→3 projections (deconv is no_grad, projections receive grads).
- ✅ 150/150 encoder params receive gradients in Stage 2.