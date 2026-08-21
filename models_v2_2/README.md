# DSS-ViT v2.2 — Dual-Stream Stain Vision Transformer

## 1. Overview

**DSS-ViT v2.2** is the regularized, reduced-capacity successor to
**DSS-ViT v2.1** for HER2 IHC breast cancer grading
(4 ordinal classes: `class_0`, `class_1+`, `class_2+`, `class_3+`).

### Why v2.2 exists

| Metric | v2.1 | Problem |
|--------|------|---------|
| Validation accuracy | ~99.6% | **Overfitting** — the model memorized the validation set |
| Test accuracy | 93.4% | Gap between val and test indicates leakage / poor generalization |

The two root causes addressed in v2.2:

1. **Random patch-level split** caused data leakage between train/val
   (the same WSI tiles appeared in both). v2.2 uses the **retrained
   baseline's WSI-aware split** (`split_indices.npz`) — no leakage.
2. **Excessive model capacity + light regularization** let the model
   memorize the training data. v2.2 reduces the StainEncoder capacity
   and adds stronger regularization (MixUp, higher weight decay,
   lower LRs, fewer fine-tuning epochs).

---

## 2. Architecture

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
 StainTokens [B,8,768]              └── x_patch [B,196,768]
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

### 2.1 Components

| Component | Detail |
|-----------|--------|
| **ColorDeconvolution** | Fixed Ruifrok & Johnston H-DAB matrix. Non-learnable, physics-based. Splits RGB → H and DAB channels. |
| **ViT-B16 backbone** | `timm vit_base_patch16_224`, ImageNet-1K pretrained. Runs on RGB (normalized internally). Produces 197 tokens [CLS + 196 patches], embed dim 768. |
| **StainEncoder** | 4× stride-2 convs (2→32→64→128→256) → BN → GELU → AdaptiveAvgPool2d(4,4) → flatten (4096) → Linear(4096, 256) → GELU → Linear(256, 8×768) → reshape [B, 8, 768]. **Reduced capacity (~3-4M params).** |
| **Cross-Attention Fusion** | `nn.MultiheadAttention(embed_dim=768, heads=12)`. Query = CLS token, Key/Value = 8 stain tokens. |
| **Gate MLP** | Linear(1536→768) → GELU → Linear(768→768), init bias 0 so gate ≈ 0.5 at init. `fused = x_cls + gate * attn_out`. |
| **OrdinalHead** | Linear(768 → 3 cutpoints). Sigmoid cutpoints → cumulative probabilities → 4 classes. |

### 2.2 Data flow

```python
h, dab = color_deconv(x_rgb)                     # (B,1,H,W) each
h_norm = (h - h_mean) / h_std
dab_norm = (dab - dab_mean) / dab_std
stain_input = cat([h_norm, dab_norm], dim=1)     # (B,2,H,W)
stain_tokens = stain_encoder(stain_input)        # (B,8,768)

rgb_norm = (x_rgb - imagenet_mean) / imagenet_std
features = vit.forward_features(rgb_norm)        # (B,197,768)
x_cls = features[:, 0]                            # (B,768)

attn_out, _ = cross_attn(query=x_cls, key=stain_tokens, value=stain_tokens)
gate = sigmoid(gate_mlp(cat([x_cls, attn_out])))
fused_cls = x_cls + gate * attn_out

logits = ordinal_head(fused_cls)                  # (B,3)
probs = cutpoints_to_probs(logits)                # (B,4)
```

---

## 3. Changes from v2.1 → v2.2

### 3.1 Model capacity (reduction)

| Component | v2.1 | v2.2 | Reason |
|-----------|------|------|--------|
| Stain tokens | 16 | **8** | Lower capacity → less memorization |
| Stain bottleneck | 512 | **256** | Lower capacity → less memorization |
| StainEncoder params | ~8-9M | **~3-4M** | ~50% reduction |

### 3.2 Regularization

| Regularization | v2.1 | v2.2 | Reason |
|----------------|------|------|--------|
| Weight decay (new modules) | 0.05 | **0.1** | Stronger penalty on new-module weights |
| Weight decay (ViT) | 0.05 | **0.05** | Kept (ViT is pretrained) |
| MixUp | disabled | **enabled (α=0.2)** | Linear interpolation of raw RGB + labels |
| Label smoothing | 0.1 | **0.1** | Kept |
| Gradient clipping | 1.0 | **1.0** | Kept |
| AMP | enabled | **enabled** | Kept (A100) |

### 3.3 Training recipe

| Stage | v2.1 | v2.2 |
|-------|------|------|
| 1 — epochs | 5 | **5** |
| 1 — LR (new) | 2e-4 | **2e-4** |
| 2 — epochs | 10 | **10** |
| 2 — LR (new) | 1e-4 | **1e-4** |
| 3 — epochs | 40 | **20** |
| 3 — ViT LR | 1e-5 | **5e-6** |
| 3 — new LR | 1e-4 | **5e-5** |
| Total epochs | 55 | **35** |

### 3.4 Data split (critical)

| Aspect | v2.1 | v2.2 |
|--------|------|------|
| Split type | Random patch-level | **WSI-aware (baseline's split)** |
| Split file | Generated by v2.1 | **Reuses** `.../ViT-Baseline/plain_vit_baseline_001/split_indices.npz` |
| Leakage | **YES** — same WSI in train+val | **NO** — clean WSI-aware split |

### 3.5 Summary of ALL changes

1. Stain tokens 16 → 8
2. Stain bottleneck 512 → 256
3. New-module weight decay 0.05 → 0.1
4. MixUp disabled → enabled (α=0.2)
5. Stage 3 epochs 40 → 20
6. Stage 3 ViT LR 1e-5 → 5e-6
7. Stage 3 new LR 1e-4 → 5e-5
8. Split: random → shared baseline WSI-aware split

---

## 4. Files in this package

| File | Type | Description |
|------|------|-------------|
| `__init__.py` | Code | Package exports |
| `color_deconv.py` | Code | Fixed H-DAB ColorDeconvolution (copy of v2.1) |
| `stain_encoder.py` | Code | **Reduced** StainEncoder (8 tokens, 256 bottleneck) |
| `ordinal_head.py` | Code | Ordinal head + CE/ordinal loss functions (copy of v2.1) |
| `stain_stats.py` | Code | Load/save global H/DAB statistics (copy of v2.1) |
| `dss_vit.py` | Code | Main DSSViT assembly (defaults 8 tokens / 256 bottleneck) |
| `README.md` | Doc | This documentation |

### Supporting files (outside this package)

| File | Purpose |
|------|---------|
| `configs/dss_vit_v2_2_config.yaml` | All hyperparameters + Toubkal paths |
| `utils/train_dss_vit_v2_2.py` | Training script (CLI, DDP, `--debug`, `--resume`) |
| `utils/evaluate_dss_vit_v2_2.py` | Official test evaluation script |
| `scripts/precompute_stain_stats_v2_2.py` | Precompute H/DAB stats on the training split |
| `slurm/slurm_dss_vit_v2_2/train_dss_vit_v2_2.slurm` | SLURM training job |
| `slurm/slurm_dss_vit_v2_2/evaluate_dss_vit_v2_2.slurm` | SLURM evaluation job |

---

## 5. Shared split (IMPORTANT)

DSS-ViT v2.2 **does NOT generate its own split**.
It loads the **retrained baseline's WSI-aware split**:

```
/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/ViT-Baseline/plain_vit_baseline_001/split_indices.npz
```

The `.npz` contains:

| Key | Description |
|-----|-------------|
| `train_indices` | Indices into the TRAIN set |
| `val_indices` | Indices into the TRAIN set (validation holdout) |
| `test_indices` | Indices into the TEST set |
| `val_fraction` | 0.0 (validate on the official TEST set) |
| `seed` | 42 |

Because this split is **shared by all models** (baseline + v2.2 + future),
every experiment uses the **exact same train/val/test images** → fair,
reproducible comparison.

---

## 6. Training recipe (full detail)

| Stage | Freeze | Trainable | LR | Epochs | Scheduler |
|-------|--------|-----------|-----|--------|-----------|
| 1 | RGB ViT | StainEncoder, cross_fusion_gate, ordinal_head | 2e-4 | 5 | Cosine (T_max=5) |
| 2 | RGB ViT | All new modules | 1e-4 | 10 | Cosine (T_max=10) |
| 3 | None | Entire model | ViT 5e-6, New 5e-5 | 20 | Cosine (T_max=20) |

### Hyperparameters

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Weight decay (new) | 0.1 |
| Weight decay (ViT) | 0.05 |
| Gradient clip | 1.0 |
| AMP | True (A100) |
| Batch size | 64 (per GPU) |
| Label smoothing | 0.1 |
| Ordinal loss weight | 0.1 |
| MixUp alpha | 0.2 |
| Loss | CE + 0.1 × ordinal BCE |

### Loss

```python
ce = cross_entropy(probs.log(), labels, label_smoothing=0.1)
ord = ordinal_bce(logits, labels)          # binary CE on cumulative cutpoints
total = ce + 0.1 * ord
```

With MixUp, the CE term is computed on both mixed label distributions:
`loss = λ * ce(y_a) + (1-λ) * ce(y_b) + 0.1 * ordinal_loss`.

---

## 7. How to run on Toubkal HPC

### 7.1 Precompute stain stats (CPU, once)

```bash
sbatch slurm/slurm_dss_vit_v2_2/precompute_stain_stats_v2_2.slurm
```

This computes global H/DAB mean & std on the **training portion of the
shared split** and saves `stain_stats.json`.

### 7.2 Debug check (interactive)

```bash
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
uv run python utils/train_dss_vit_v2_2.py \
    --config configs/dss_vit_v2_2_config.yaml \
    --debug
```

Validates:
- Shapes: stain_tokens `[B,8,768]`, x_cls `[B,768]`, fused_cls `[B,768]`,
  logits `[B,3]`, probs `[B,4]`
- `loss.backward()` works
- No NaN/Inf
- Gate initialization ≈ 0.5
- Prints per-group parameter counts (StainEncoder ~3-4M)

### 7.3 Full training (GPU, 35 epochs)

```bash
sbatch slurm/slurm_dss_vit_v2_2/train_dss_vit_v2_2.slurm
```

Monitor:

```bash
squeue -u amine.aitlaamim-ext
tail -f /home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT-v2.2/dss_vit_v2_2_001/slurm_<JOBID>.out
```

### 7.4 Evaluate on official test set

```bash
sbatch slurm/slurm_dss_vit_v2_2/evaluate_dss_vit_v2_2.slurm
```

Loads `best_stage3.pt` and reports accuracy, balanced accuracy,
macro/weighted F1, QWK, per-class metrics, and confusion matrix.

### 7.5 Resume training

```bash
# Uncomment --resume in the SLURM script, or run interactively:
uv run python utils/train_dss_vit_v2_2.py --config configs/dss_vit_v2_2_config.yaml --resume
```

---

## 8. Outputs

| Path | Contents |
|------|----------|
| `experiments/DSS-ViT-v2.2/dss_vit_v2_2_001/experiment_meta.json` | Identity + config snapshot |
| `checkpoints/DSS-ViT-v2.2/dss_vit_v2_2_001/stage1_end.pt` | End of Stage 1 |
| `checkpoints/DSS-ViT-v2.2/dss_vit_v2_2_001/stage2_end.pt` | End of Stage 2 |
| `checkpoints/DSS-ViT-v2.2/dss_vit_v2_2_001/best_stage3.pt` | Best Stage 3 (used by eval) |
| `checkpoints/DSS-ViT-v2.2/dss_vit_v2_2_001/last.pt` | Latest state (`--resume`) |
| `logs/DSS-ViT-v2.2/dss_vit_v2_2_001/train.log` | Training log |
| `logs/DSS-ViT-v2.2/dss_vit_v2_2_001/metrics.jsonl` | Per-epoch metrics |
| `results/DSS-ViT-v2.2/dss_vit_v2_2_001/test_results.json` | Evaluation results |
| `results/DSS-ViT-v2.2/dss_vit_v2_2_001/test_report.txt` | Human-readable report |

All paths are configured in `configs/dss_vit_v2_2_config.yaml`.

---

## 9. DDP / Multi-GPU

Edit `slurm/slurm_dss_vit_v2_2/train_dss_vit_v2_2.slurm`:

```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
```

And replace the training command:

```bash
srun uv run python utils/train_dss_vit_v2_2.py --config "${CONFIG}"
```

The script auto-detects `LOCAL_RANK/WORLD_SIZE/RANK`. Only rank 0
saves checkpoints and logs. `find_unused_parameters=True` is set in DDP.

---

## 10. Expected outcome

The goal of v2.2 is:

- **Reduce validation overfitting** (v2.1: val ~99.6% vs test 93.4%)
- **Improve test generalization** (target > 93.4% test accuracy, or at
  least reduce the val/test gap)
- **Fair comparison** with the baseline — both use the **same
  WSI-aware split**