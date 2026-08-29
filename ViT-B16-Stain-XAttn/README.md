# ViT-B16-Stain-XAttn — Gated Cross-Attention Fusion of RGB, H, and DAB

**ViT-B16 fine-tuned on RGB, with optional spatial fusion of
Hematoxylin and DAB stain maps through a gated cross-attention
mechanism inserted into the last transformer blocks.**

Experiment: `vit_b16_stain_xattn_001`

## Model

- `ViTB16StainXAttn`: `timm.vit_base_patch16_224` (ImageNet-pretrained,
  `num_classes=0`) + a small stain pathway + `GatedCrossAttentionBlock`
  modules inserted into configurable last blocks (default `[8,9,10,11]`).
- A `Linear(768, 4)` head — identical to the baseline (no CORN, no
  dropout, no hidden layer).

### The zero-init gate guarantee (MOST IMPORTANT)

Each `GatedCrossAttentionBlock` has a scalar gate `alpha` initialized to
**exactly 0.0**. Inside `forward`, the module returns
`tanh(alpha) * attn_out` — a **gated delta** the caller adds to the RGB
residual stream. At initialization `tanh(0)=0`, so the stain pathway
contributes an **exact zero delta** and the model is **numerically
identical to the plain ViT-B16 baseline**. Verified by `--debug` check 10.4
(`assert torch.allclose(out_xattn, out_ref, atol=1e-5)` to a reference
plain-ViT with the same head). The stain pathway can only *add* signal
during training — it cannot make the model worse than the baseline at
step zero.

### Why this design over alternatives

- **Concat / small-CNN bottleneck** (`UNI-Stain-Attention`-style) was
  tried first: stain features go through a separate learnable encoder,
  producing a fixed embedding concatenated late. It both underperformed
  and injected a fixed-capacity bottleneck.
- **DSCA-ViT's ungated bidirectional cross-attention** was also tried —
  ungated, so it could inject large untrained updates into a working
  network at step zero.
- **Gated cross-attention (Flamingo-style)** lets RGB tokens attend to
  stain tokens *at the spatial level* while the gate keeps init outputs
  bit-identical to baseline and lets head + stain weights co-adapt.

### Why only the last 4 blocks (`[8,9,10,11]`)

Kept sparse and late-stage for parameter cost, and to resemble
`UNI-Stain-Attention`, the one prior fusion attempt that worked.

### H and DAB handling

- **Separate H and DAB patch embeddings** (`h_patch_embed`, `dab_patch_embed`
  — independent `Conv2d(1,768,16,16)`), deliberately NOT shared: parameter
  cost is trivial, and DAB's absolute intensity is clinically load-bearing
  (the `StainNorm1ch` lesson).
- **Fixed global normalization** from precomputed train-split stats
  (`h_norm=(h-h_mean)/h_std`, `dab_norm=(dab-dab_mean)/dab_std`) applied as
  a constant affine — **never per-instance**. Per-instance `GroupNorm`
  previously erased DAB's absolute-intensity signal (`StainNorm1ch` failure).
- Stain positional embedding is an **independent copy** of the backbone's
  patch positional embeddings (`pos_embed[:, 1:, :].clone()`): 14×14 = 196 grid.
- Two modality-type embeddings (`h_type_embed`, `dab_type_embed`), not
  three — RGB is always query, stain always key/value, so roles
  already disambiguate RGB.

### Stage-1 gradient-ramp dynamic

At `alpha=0`, gradient to `alpha` is nonzero (`sech²(0)·attn_out`), but
gradient to weights inside the cross-attention is `tanh(0)·(their grad) = 0`.
Under Adam, `alpha` typically moves off 0 within a few steps, then
gradient flows to the rest of the pathway — the Flamingo co-adaptation
mechanism. **This is logged empirically**: the first 50 Stage-1 steps log
`tanh(alpha)` and the layer-8 `out_proj.weight.grad` norm. If the grad norm
stays pinned near zero well into Stage 1 (past ~epoch 5), that's evidence
a longer ramp is needed and would justify freezing CA weights in Stage 1
(evidence-based follow-up, not a preemptive change).

### Deliberate data-policy deviation

The dataloader outputs **raw RGB `[0,1]` (no ImageNet Normalize)** — unlike
the plain ViT-B16 baseline — because color deconvolution `(-log10(x+eps))`
requires real pixel intensities; normalized (possibly negative) values
would silently produce meaningless H/DAB maps. ImageNet normalization is
applied **inside `forward()`**, only to the RGB pathway.
## Training recipe (matches plain ViT-B16 baseline exactly)

| Stage | Freeze | Trainable | LR | Epochs |
|---|---|---|---|---|
| 1 | `pretrained_backbone_parameters()` | `new_component_parameters()` | 1e-4 | 30 |
| 2 | None | Everything | Backbone 1e-5, new components 1e-4 | 30 |

- Adam, weight decay 0.0, batch 32, AMP off, no grad clipping
- CosineAnnealingLR per stage, no early stopping (fixed 30+30)
- Stage 2 initializes from `best_stage1.pt` (NEVER `last.pt`)
- Checkpoints: `best_stage1.pt`, `best_stage2.pt`, `last.pt`

## Data & split

- Load-only `split_indices_wsi.npz` (train 7283 | val 810 | test 1847;
  val_fraction 0.10, seed 42).
- Train transforms: Resize(224), HFlip, VFlip, Rot10, ToTensor (raw).
- Val/Test: Resize(224), ToTensor (raw).

## Commands

```bash
# Precompute H/DAB stats on the train split (required before training)
uv run python ViT-B16-Stain-XAttn/precompute_stain_stats.py \
    --config configs/vit_b16_stain_xattn_config.yaml

# Debug sanity checks (offline) - incl. the zero-contribution-at-init test
uv run python ViT-B16-Stain-XAttn/train_vit_b16_stain_xattn.py \
    --config configs/vit_b16_stain_xattn_config.yaml --debug

# Train (Stage 1 -> Stage 2; Stage 2 auto-inits from best_stage1.pt)
uv run python ViT-B16-Stain-XAttn/train_vit_b16_stain_xattn.py \
    --config configs/vit_b16_stain_xattn_config.yaml

# Official test (loads best_stage2.pt, NEVER last.pt)
uv run python ViT-B16-Stain-XAttn/evaluate_vit_b16_stain_xattn.py \
    --config configs/vit_b16_stain_xattn_config.yaml
```

## SLURM (Toubkal / UM6P, 1 A100)

```bash
sbatch slurm/slurm_vit_b16_stain_xattn/precompute_stain_stats.slurm
sbatch slurm/slurm_vit_b16_stain_xattn/train_vit_b16_stain_xattn.slurm
sbatch slurm/slurm_vit_b16_stain_xattn/evaluate_vit_b16_stain_xattn.slurm
```

## Reference results to beat

| Model | Test Acc | Bal Acc | Macro F1 |
|---|---|---|---|
| Proper ViT-B16 baseline | 94.69% | 93.58% | 91.11% |
| ViT-B16-CORN | 94.26% | 93.52% | 92.03% |
| Ensemble (best so far) | 95.61% | 94.59% | 93.14% |

## Logging per-layer gate values after training

After training, read each layer's `tanh(model.cross_attn_modules[str(i)].alpha)`
from the checkpoint. Moving away from zero indicates the network found the
stain signal useful at that depth; staying near zero is itself a valid,
informative negative result.
