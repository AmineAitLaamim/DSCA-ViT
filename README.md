# DSCA-ViT — Dual-Stain Cross-Attention Vision Transformer

A biologically-informed Vision Transformer for automated HER2 IHC scoring.

## Architecture

```
RGB Patch (224×224×3)
        │
        ▼
Color Deconvolution (Ruifrok, fixed)
        │
  ┌─────┴─────┐
  │           │
  ▼           ▼
Hematoxylin  DAB
  │           │
1→3 Proj    1→3 Proj
  │           │
  └─────┬─────┘
        │ (shared ViT Blocks 1–9)
        │
Bidirectional Cross-Attention
  (spatially-biased)
        │
        │ (shared ViT Blocks 10–12)
        │
  Gated Token Fusion
        │
  Refinement Block
        │
Classification Head
        │
  HER2 Score {0, 1+, 2+, 3+}
```

**Core novelty**: Spatially-biased bidirectional cross-attention that explicitly
exploits the pixel-level spatial correspondence between H and DAB channels — a
property unique to color deconvolution.

## Project Structure

```
DSCA-ViT/
│
├── models/
│   ├── __init__.py
│   ├── dsca_vit.py          # Main model assembly
│   ├── shared_vit.py        # Shared ViT encoder + 1→3 projection
│   ├── color_deconv.py      # Fixed Ruifrok color deconvolution
│   ├── cross_attention.py   # Spatially-biased bidirectional cross-attention
│   └── fusion.py            # Gated fusion, refinement block, classifier
│
├── datasets/
│   ├── __init__.py
│   ├── dataset.py           # HER2Dataset (unchanged from baseline)
│   └── transforms.py        # Transforms WITHOUT ImageNet normalization
│
├── utils/
│   ├── __init__.py
│   ├── train.py             # train_one_epoch
│   ├── evaluate.py          # validate_one_epoch
│   ├── metrics.py           # compute_metrics, print_metrics
│   └── checkpoint.py        # save_checkpoint, load_checkpoint
│
├── configs/
│   └── dsca_vit_b16.yaml    # Hyperparameters and paths
│
├── notebooks/
│   ├── train.py             # Colab training script (paste as cells)
│   ├── sanity_check.py      # Module unit tests (run before training)
│   └── visualize.py         # Deconvolution, gates, attention maps
│
└── README.md
```

## Environment

- Python 3.11
- PyTorch 2.x
- timm
- Google Colab (GPU)

## Getting Started (Colab)

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/DSCA-ViT.git /content/DSCA-ViT

# 2. Install dependencies
!pip install timm pyyaml --quiet

# 3. Run sanity check
!python /content/DSCA-ViT/notebooks/sanity_check.py

# 4. Run training (paste cells from notebooks/train.py)
```

## Key Design Decisions

| Decision | Rationale |
|:---|:---|
| Fixed color deconvolution | Physics-based, interpretable, no data dependency |
| Shared encoder (100%) | Parameter efficiency; prevents 2× model size |
| Learnable 1→3 projection | Preserves pretrained ViT patch embedding |
| Bidirectional cross-attention | Both H→DAB and DAB→H directions carry clinical meaning |
| **Spatially-biased attention** | **Exploits guaranteed pixel registration between stains** |
| Gated fusion | Per-patch adaptive weighting; interpretable |
| Post-fusion refinement block | Reasons over fused stain representation |

## Important Notes

### No ImageNet Normalization

The transforms in `datasets/transforms.py` do **not** apply ImageNet
normalization. This is intentional:

> Color deconvolution requires raw RGB values in [0, 1] range. Applying
> ImageNet normalization before deconvolution would corrupt the optical
> density computation (Beer-Lambert law).

### Augmentation Order

Data augmentation must be applied to the RGB image **before** it enters the
model. Both H and DAB channels must see the same geometric transforms, or the
spatial correspondence between streams is broken.

## Ablation Studies

See `implementation_plan.md` for the full ablation plan (18 experiments).
Priority order: A-series (architecture) → B-series (attention) → C-series
(placement) → D-series (fusion).

## Citation

[To be added upon publication]
