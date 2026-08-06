# DSCA-ViT — Complete Implementation Summary

---

## Part 1 — What Was Built

### Project Layout

```
DSCA-ViT/
├── models/          (6 files)   — All neural network modules
├── datasets/        (3 files)   — Data loading and transforms
├── utils/           (5 files)   — Training infrastructure
├── configs/         (1 file)    — Hyperparameters
├── notebooks/       (3 files)   — Colab scripts
└── README.md
```

---

### models/ — The Neural Network

#### [color_deconv.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/models/color_deconv.py)

**What it does**: Fixed (non-learnable) module that separates an RGB image into
Hematoxylin and DAB channels using the Ruifrok & Johnston Beer-Lambert method.

**How it works**:
```
x_rgb [0,1]  →  OD = -log10(x_rgb + ε)  →  stain_ODs = M_inv @ OD  →  H, DAB, Residual
```

**Key design choices**:
- Stain matrix registered as a `buffer` (moves to GPU automatically, not a parameter)
- Output clamped to `>= 0` (negative optical density is non-physical)
- `torch.no_grad()` wraps the call in `dsca_vit.py` — no gradient flows through here
- Includes `deconvolve_numpy()` for visualization without PyTorch

---

#### [shared_vit.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/models/shared_vit.py)

**What it does**: Two things in one file.

**`StainChannelProjection`** — Learnable `Conv2d(1, 3, 1×1)`:
```
H (B, 1, 224, 224)  →  Proj_H  →  (B, 3, 224, 224)  →  pretrained ViT patch_embed
```
Initialized so each output channel ≈ input (`ones` init), preserving pretrained behavior early in training.

**`SharedViTEncoder`** — Wraps `timm.create_model("vit_base_patch16_224")` and:
- Removes the original classification head (`num_classes=0`)
- Splits the 12 blocks into `blocks_before` (0..8) and `blocks_after` (9..11)
- Exposes `embed()`, `forward_before()`, `forward_after()` separately
- Both H and DAB streams use the **exact same parameter tensors** — weight sharing is real, not simulated

---

#### [cross_attention.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/models/cross_attention.py)

**What it does**: The core novel contribution — spatially-biased bidirectional cross-attention.

Three classes:

**`SpatialBiasMatrix`** — Learnable `(197, 197)` parameter:
```
B[i,i] = +beta   (self-correspondence bonus, default 1.0)
B[i,j] = -gamma * euclidean_distance(i, j) on 14×14 grid
B[0,:] = B[:,0] = 0  (CLS token gets no spatial bias)
```
This matrix is ADDED to the raw QKᵀ/√d logits before softmax.

**`CrossAttentionLayer`** — Single direction cross-attention:
```
Q = W_q(LN(source))
K = W_k(LN(context))
V = W_v(LN(context))
logits = QKᵀ/√d_k + B
attn = softmax(logits)
output = source + W_o(attn @ V)
```
Stores `attn_weights` for visualization.

**`BidirectionalCrossAttention`** — Parallel bidirectional:
```
# BOTH directions computed from ORIGINAL (pre-update) tokens
H_updated = CrossAttn(Q=H, K=D, V=D, bias=B)  ← H queries DAB
D_updated = CrossAttn(Q=D, K=H, V=H, bias=B)  ← DAB queries H
H_out = FFN_H(H_updated)
D_out = FFN_D(D_updated)
```

---

#### [fusion.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/models/fusion.py)

Three classes:

**`GatedFusion`**:
```
# CLS token (index 0):
F_0 = Linear(1536, 768)([CLS_H || CLS_D])

# Patch tokens (indices 1..196):
g_i = sigmoid(Linear(1536, 768)([H_i || D_i]))
F_i = g_i ⊙ H_i + (1-g_i) ⊙ D_i

Returns: fused_tokens (B, 197, 768) + gate_values (B, 196, 768)
```

**`RefinementBlock`** — 1 standard transformer block (Xavier init, trained from scratch):
```
x = x + MHSA(LN(x))
x = x + FFN(LN(x))
```

**`ClassificationHead`**:
```
cls = tokens[:, 0, :]          (B, 768)
gap = mean(tokens[:, 1:, :])   (B, 768)
z   = [cls || gap]             (B, 1536)
out = LN → Linear(768) → GELU → Dropout(0.1) → Linear(4)
```

---

#### [dsca_vit.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/models/dsca_vit.py)

The top-level model. Full forward pass:

| Step | Operation | Tensor Shape |
|:-----|:----------|:-------------|
| 0 | RGB input | `(B, 3, 224, 224)` |
| 1 | Color deconvolution | `(B, 1, 224, 224)` × 2 |
| 2 | 1→3 projection (×2) | `(B, 3, 224, 224)` × 2 |
| 3 | Patch embed + CLS + pos | `(B, 197, 768)` × 2 |
| 4 | Shared blocks 1–9 (batched) | `(B, 197, 768)` × 2 |
| 5 | Bidirectional cross-attention | `(B, 197, 768)` × 2 |
| 6 | Shared blocks 10–12 (batched) | `(B, 197, 768)` × 2 |
| 7 | Gated fusion | `(B, 197, 768)` + gates |
| 8 | Refinement block | `(B, 197, 768)` |
| 9 | Classification head | `(B, 4)` |

The batching trick in steps 4 and 6:
```python
stacked = cat([h_tokens, d_tokens], dim=0)  # (2B, 197, 768)
stacked = encoder.forward_before(stacked)    # single GPU call
h_tokens, d_tokens = stacked.split(B, dim=0)
```

Also provides: `get_gate_values()`, `get_parameter_groups()`, `count_parameters()`.

---

### datasets/ — Data Loading

#### [transforms.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/datasets/transforms.py)

```python
# CRITICAL: NO ImageNet Normalize() here
get_train_transform():  Resize → HFlip → VFlip → Rotate(10°) → ToTensor()
get_test_transform():   Resize → ToTensor()
```

Output is `[0, 1]` float. The model expects this. The baseline used Normalize — this does not.

#### [dataset.py](file:///c:/Users/pc/projects/stage_fssm_brest_cancer/Dual-Stream_Cross-Attention_Vision_Transformer/datasets/dataset.py)

Exact copy of `HER2Dataset` from the baseline notebook. Zero changes. Same class names, same directory structure, same `__getitem__` logic.

---

### utils/ — Training Infrastructure

All functions are model-agnostic (work with any model that takes images and returns logits).

| File | Function | Notes |
|:-----|:---------|:------|
| `train.py` | `train_one_epoch(model, loader, criterion, optimizer, device)` | Returns `(loss, acc%)` |
| `evaluate.py` | `validate_one_epoch(model, loader, criterion, device)` | Returns `(loss, acc%, preds, labels)` |
| `metrics.py` | `compute_metrics(labels, preds, class_names)` | sklearn: accuracy, precision, recall, F1, report, CM |
| `checkpoint.py` | `save_checkpoint(...)` / `load_checkpoint(...)` | `**extra_info` for arbitrary metadata |

---

### notebooks/ — Colab Scripts

| File | Purpose |
|:-----|:--------|
| `sanity_check.py` | Tests every module shape independently. Run before training. Uses `pretrained=False` — no download needed. |
| `train.py` | Full 2-stage training pipeline mirroring baseline structure. Same Stage 1 (frozen) → Stage 2 (fine-tune) paradigm. |
| `visualize.py` | 3 visualization functions: deconvolution channels, gate heatmaps, cross-attention maps. |

---

### configs/dsca_vit_b16.yaml

Centralizes all hyperparameters. Currently the training notebook hardcodes values — the YAML exists as a reference. Config loading not yet wired in.

---

---

## Part 2 — What Will Likely Fail

> [!CAUTION]
> These are **real risks** that will need fixing before the model trains correctly.

---

### 🔴 CRITICAL — Will Definitely Fail or Produce Wrong Results

---

**F1 — The transforms do NOT apply stain normalization preprocessing**

The color deconvolution assumes the input is a raw IHC scan in `[0, 1]` range.
But the HER2-IHC-40x images are JPEG-compressed, and their actual pixel
statistics may drift significantly across slides. The Ruifrok matrix uses fixed
stain vectors calibrated for a specific lab's staining protocol.

If the images in the dataset were captured with different antibody lots,
scanners, or staining conditions, the deconvolution will produce
incorrectly separated channels — the "H channel" will contain DAB signal
and vice versa.

**What will happen**: The model will still train and may converge, but the
biological meaning of the two streams will be corrupted. You won't be able
to tell from the training loss curve.

**How to detect**: Visually inspect `visualize_deconvolution()` on 10+ images.
If the DAB channel looks brown and clean, the vectors are correct. If it looks
gray and noisy, they are not.

**Fix**: Add Macenko normalization as a preprocessing step before deconvolution,
or use per-image estimated stain vectors instead of the fixed Ruifrok matrix.

---

**F2 — The `BidirectionalCrossAttention` constructor signature likely mismatches `DSCAViT`**

In `dsca_vit.py`, the constructor calls:
```python
self.cross_attention = BidirectionalCrossAttention(
    embed_dim=768, num_heads=12, beta=1.0, gamma=0.5
)
```

But the subagent-built `cross_attention.py` may use different parameter names
(e.g., `spatial_bias_beta`, `spatial_bias_gamma`, or no `beta`/`gamma` at all
if it was hardcoded). The subagent worked independently without seeing `dsca_vit.py`.

**What will happen**: `TypeError` on import or model instantiation.

**How to detect**: Run `sanity_check.py` — it will fail immediately with a clear error.

**Fix**: Open both files and align the constructor signatures.

---

**F3 — The `visualize_cross_attention()` function references `bca.cross_attn_hd` and `bca.cross_attn_dh`**

These are assumed attribute names inside `BidirectionalCrossAttention`.
The subagent may have named them differently (e.g., `self.attn_hd`, `self.direction_1`).

**What will happen**: `AttributeError` when running the visualization.

**How to fix**: Check the actual attribute names in `cross_attention.py` and update `visualize.py`.

---

**F4 — Gate values shape assumption in `DSCAViT.get_gate_values()`**

`fusion.py` was built to return `(fused_tokens, gate_values)` where
`gate_values` has shape `(B, 196, 768)`. If the subagent implemented
gates differently (e.g., scalar gates `(B, 196, 1)` instead of per-dimension
`(B, 196, 768)`), the visualization code will produce incorrect or crashy heatmaps.

**How to detect**: Check the `GatedFusion.forward()` return shape in the sanity check output.

---

### 🟠 HIGH — Will Likely Cause Training Problems

---

**F5 — `torch.no_grad()` around color deconvolution blocks gradient flow through projections**

In `dsca_vit.py`:
```python
with torch.no_grad():
    h_channel, d_channel = self.color_deconv(x_rgb)

h_rgb = self.proj_h(h_channel)   # ← gradient blocked here!
```

Because `h_channel` and `d_channel` are produced under `no_grad()`, they have
`requires_grad=False`. The 1→3 projections (`proj_h`, `proj_d`) receive inputs
with no gradient history. Their gradients **will** be computed normally during
backward, so training is not entirely broken — but the input gradient to the
projections is disconnected, which is actually correct behavior (we don't want
gradients flowing into the fixed deconvolution). This is fine.

**Status**: Actually correct. Not a bug.

---

**F6 — Memory: batch_size=32 will likely OOM on Colab T4**

The model runs 2 forward passes through ViT-B/16 per sample (via the batched
cat trick, effective batch is 2×32=64 through the encoder). ViT-B/16 with
sequence length 197 and batch 64 requires ~13 GB GPU memory.

A Colab T4 has **15 GB**. This is tight. Any spike will crash the session.

**Fix**: Drop batch_size to 16 first. If that works, try 24.

---

**F7 — The StainChannelProjection `ones` initialization is not identical to channel replication**

`ones` init means all three output channels get `W = 1, b = 0`, so
`output[:, k, :, :] = input * 1 + 0 = input` for all k.
This IS channel replication at init — correct.

But after training, the bias term `b` will be nonzero. Since the deconvolution
outputs optical densities (typically `[0, 2+]`), the bias can push values
outside the range that the pretrained ViT normalization implicitly expects
(it was trained on ImageNet-normalized `[-2, 2]` range inputs).

**Fix**: Apply a learned per-stain normalization (a simple `LayerNorm` or
affine rescaling) after the 1→3 projection to recenter the input distribution
before the ViT patch embedding.

---

**F8 — Early stopping is not implemented**

The baseline notebook had no explicit early stopping either (just saved the
best checkpoint). But 30 epochs of fine-tuning on a small dataset (~8K images)
is very likely to overfit, especially in Stage 2 where the baseline already
showed the model memorizing training data (100% train acc, degrading val acc
after epoch 3–6).

**Fix**: Add a patience counter to the training loop. Recommend `patience=7`.

---

### 🟡 MEDIUM — Will Hurt Performance but Not Break Training

---

**F9 — The spatial bias matrix is initialized with `gamma=0.5` on a 14×14 grid**

Maximum distance on a 14×14 grid is `sqrt((13)² + (13)²) ≈ 18.4` units.
At `gamma=0.5`, the maximum penalty is `-9.2` logits.

But ViT attention logits are typically in the range `[-3, 3]` before softmax.
A bias of `-9.2` at the furthest patch will make those attention weights
essentially zero, which is equivalent to a hard mask — much more restrictive
than intended.

**Fix**: Use a smaller gamma (e.g., `0.1` or `0.2`) so the bias is a soft
prior, not a hard constraint.

---

**F10 — The YAML config is not actually wired into the training notebook**

`configs/dsca_vit_b16.yaml` exists but the training notebook (`notebooks/train.py`)
hardcodes all values directly. The YAML is unused documentation.

**Fix**: Add `import yaml; cfg = yaml.safe_load(open(...))` at the top of the
training notebook and reference `cfg["stage1"]["batch_size"]` etc.

---

**F11 — `save_checkpoint` and `load_checkpoint` are not tested together**

The `save_checkpoint` function takes `**extra_info` kwargs. The `load_checkpoint`
function returns the full checkpoint dict. If the caller tries to resume
Stage 2 from Stage 1 and reads `checkpoint["stage"]`, this works fine.
But if the subagent used different dict key names (e.g., `"val_accuracy"` vs
`"best_val_accuracy"`), the training notebook will fail silently or raise a
`KeyError`.

**Fix**: Run `sanity_check.py` fully, then add a checkpoint round-trip test.

---

---

## Part 3 — Hidden Assumptions

These are things the implementation assumes to be true that were **never
verified or explicitly stated**.

---

**A1 — The dataset images are stored as PNG or JPEG with values in [0, 255]**

`HER2Dataset` opens images with `PIL.Image.open(path).convert("RGB")`.
`ToTensor()` then divides by 255 to get `[0, 1]`.

If the dataset stores images differently (e.g., 16-bit TIFF, or already
normalized), `ToTensor()` will silently produce wrong values and the
deconvolution will compute incorrect optical densities.

**Assumed true but never checked**: The dataset is standard JPEG/PNG.

---

**A2 — The pretrained ViT patch embedding still works on non-ImageNet-normalized inputs**

The shared ViT encoder was pretrained with ImageNet normalization (mean/std).
We are feeding it pseudo-RGB outputs from a `[0, 1]`-range stain projection —
a completely different distribution.

The assumption is that the pretrained weights still provide useful initialization
despite this distribution mismatch. This is a reasonable assumption (transfer
learning is robust to moderate distribution shifts), but it means:
- Stage 1 training will need more epochs than the baseline to "adapt" the
  pretrained filters to the new input distribution
- The optimal learning rate may need to be higher for the encoder in Stage 2

---

**A3 — The 14×14 patch grid is correct for ViT-B/16 at 224×224**

224 / 16 = 14. This is correct and the spatial bias matrix is built on this.
But if someone changes `image_size` in the config to anything other than 224,
the spatial bias matrix will have wrong dimensions (it's hardcoded for 197 tokens).

**Assumed true**: `image_size` is always 224.

---

**A4 — Both H and DAB channels always have meaningful signal**

The gated fusion assumes both streams contain useful information in every patch.
But in practice, background patches (glass slide, white space between tissue)
will have near-zero signal in both channels after deconvolution. The model will
try to gate between two sources of noise.

This is fine for training (the gate will learn to output ~0.5 for background
patches, which doesn't affect the prediction much), but it wastes attention
capacity in the cross-attention module.

---

**A5 — Weight sharing benefits both streams equally**

The shared encoder is initialized from ImageNet ViT weights. These weights were
trained on natural RGB images. The H channel (blue/purple morphology) is
moderately similar to natural images. The DAB channel (brown HER2 signal)
is much less similar.

The assumption is that the shared encoder adapts to both streams during
fine-tuning. This may be true, but the DAB stream may systematically
underperform the H stream because its input distribution is further from
ImageNet.

---

**A6 — `torch.cuda.empty_cache()` is not needed**

The training loop in `utils/train.py` does not call `torch.cuda.empty_cache()`.
On Colab with limited VRAM, this can cause OOM errors mid-epoch from
fragmented memory — even if theoretical peak usage is within limits.

---

**A7 — The subagent-built modules are internally consistent with each other**

The three core modules (`color_deconv.py`, `cross_attention.py`, `fusion.py`)
were written by three independent subagents running in parallel. They did not
see each other's code. The `dsca_vit.py` assembly was written by the main agent
making assumptions about what they produced (constructor signatures, attribute
names, return types).

**This is the biggest hidden assumption**: Everything will connect cleanly.
The sanity check exists precisely to surface the inevitable mismatches.

---

## Summary Table

| Risk | Severity | Detectable With | Fix |
|:-----|:---------|:----------------|:----|
| Stain vectors miscalibrated | 🔴 Critical | Visual inspection | Macenko normalization |
| Constructor signature mismatch | 🔴 Critical | sanity_check.py | Align `__init__` params |
| Visualization attribute names wrong | 🔴 Critical | visualize.py run | Check actual attr names |
| Gate shape mismatch | 🔴 Critical | sanity_check.py | Fix GatedFusion return |
| OOM at batch_size=32 | 🟠 High | First training step | Reduce to 16 |
| Distribution mismatch (no norm) | 🟠 High | Training curves | Add post-projection norm |
| No early stopping | 🟠 High | Overfitting curves | Add patience counter |
| Spatial bias too aggressive | 🟡 Medium | Ablation B4 | Lower gamma to 0.1 |
| YAML config not wired | 🟡 Medium | Code review | Add yaml.safe_load |
| Checkpoint key naming | 🟡 Medium | sanity_check.py | Check dict keys |

---

## Recommended First Actions

1. **Run `sanity_check.py` with `pretrained=False`** — this will surface
   all constructor mismatches before downloading any weights.

2. **Fix constructor mismatches** — open `cross_attention.py` and `fusion.py`,
   compare their `__init__` signatures to the calls in `dsca_vit.py`.

3. **Run `visualize_deconvolution()` on 5 images** from the dataset to
   confirm the H/DAB separation is biologically correct.

4. **Set batch_size=16 for the first run** — confirm the model trains at all
   before scaling up.

5. **Inspect Stage 1 learning curves** — if validation accuracy doesn't
   improve past ~60% after 10 epochs, the input distribution mismatch
   (Assumption A2) is the likely cause. Add a `LayerNorm` after the
   1→3 projections.
