# DSS-ViT Architecture

## Overview

DSS-ViT is a dual-stream Vision Transformer for HER2 IHC scoring:

- **RGB main stream**: pretrained ViT-B16 (ImageNet), input: raw RGB `[0,1]`
- **Stain auxiliary stream**: ColorDeconv (fixed Ruifrok H/DAB) → normalized → StainEncoder → tokens `[B,16,768]`
- **Fusion**: cross-attention (CLS query × stain tokens) + gated residual
- **Head**: ordinal cutpoints → class probabilities (0/1+/2+/3+)
- **Loss**: CE + α·ordinal-BCE (α=0.1)

## Files

```
models_v2_1/
├── __init__.py
├── color_deconv.py
├── stain_encoder.py
├── ordinal_head.py
├── stain_stats.py
└── dss_vit.py

utils/
├── train_dss_vit.py
├── metrics_dss_vit.py
├── evaluate_dss_vit.py
└── split_utils.py
```

`scripts/` contains `precompute_stain_stats.py`.

`configs/` contains `dss_vit_config.yaml`.

`slurm/` contains:
- `train_dss_vit.slurm`
- `evaluate_dss_vit.slurm`
- `precompute_stain_stats.slurm`

## Train

```bash
sbatch slurm/train_dss_vit.slurm
```

Or interactive debug:

```bash
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug
```

Resume from latest:

```bash
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --resume
```

Third stage selection: **accuracy** (best validation accuracy in Stage 3 → `best_stage3.pt`).

## Evaluate on official test

```bash
sbatch slurm/evaluate_dss_vit.slurm
```

Or:

```bash
python utils/evaluate_dss_vit.py --config configs/dss_vit_config.yaml
```

## Precompute stain stats

```bash
sbatch slurm/precompute_stain_stats.slurm
```

Or:

```bash
python scripts/precompute_stain_stats.py --config configs/dss_vit_config.yaml
```

## Model forward

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

## Training stages (55 epochs)

| Stage | Freeze | Trainable | LR | Epochs |
|---|---|---|---|---|
| 1 | `vit` | StainEncoder, CrossFusion, OrdinalHead | 2e-4 | 5 |
| 2 | `vit` | All new modules | 1e-4 | 10 |
| 3 | — | Entire model | ViT 1e-5, new 1e-4 | 40 |

- AdamW, weight decay 0.05 (weights only), clip 1.0, AMP on A100.
- Per-stage `CosineAnnealingLR`; optimizer param groups rebuilt at each stage.
- MixUp/CutMix off by default (`mixup_alpha: 0.0`).

## DDP

- Auto-inits when `LOCAL_RANK`/`WORLD_SIZE`/`RANK` are set (SLURM `srun`).
- `DistributedSampler` for training; validation rank 0 only.
- `find_unused_parameters=True`; optimizer rebuilt after `requires_grad` changes.
- Only rank 0 saves checkpoints/logs.
- Batch size in config is **per GPU**.