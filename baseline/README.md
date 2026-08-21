# Plain ViT-B16 Baseline — HER2-IHC-40x

This folder contains a **self-contained reproduction** of the original
Colab baseline (`baseline_vit_model/model_HER2_ViT.ipynb`) as a set of
CLI scripts ready to run on the **Toubkal HPC** (UM6P) via SLURM.

---

## 1. What this baseline is

| Item | Value |
|------|-------|
| Task | HER2 IHC breast cancer grading (4 classes) |
| Classes | `class_0`, `class_1+`, `class_2+`, `class_3+` |
| Dataset | HER2-IHC-40x (WSI split) |
| Model | `timm vit_base_patch16_224` (ImageNet-1K pretrained) |
| Input | 224 × 224 RGB, ImageNet normalization |
| Expected accuracy | **95.02%** (original Colab result) |

### Training protocol (identical to the original notebook)

| Stage | Strategy | Optimizer | LR | Epochs | Scheduler |
|-------|----------|-----------|-----|--------|-----------|
| 1 | Frozen backbone, train head only | Adam | 1e-4 | 30 | CosineAnnealingLR |
| 2 | Full fine-tuning | Adam | backbone 1e-5, head 1e-4 | 30 | CosineAnnealingLR |

- **Loss**: plain `CrossEntropyLoss` (no label smoothing)
- **Batch size**: 32
- **Validation**: on the **official TEST set** (as in the notebook)
- **No AMP**, no gradient clipping, no mixup, no weight decay

### Data augmentation (train only)

```
Resize(224, 224)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(10°, bilinear, fill=0)
ToTensor()
Normalize(ImageNet mean/std)
```

---

## 2. Files in this folder

| File | Purpose |
|------|---------|
| `models_baseline.py` | `PlainViTB16` — timm ViT-B/16 with a 4-class head |
| `baseline_data.py` | Dataset loader, transforms, and shared split utility |
| `metrics_baseline.py` | Accuracy, balanced accuracy, macro/weighted F1, QWK, confusion matrix |
| `train_baseline_vit.py` | 2-stage training CLI (HPC/DDP-ready) |
| `evaluate_baseline_vit.py` | Official test-set evaluation CLI |
| `__init__.py` | Package exports |
| `baseline_vit_model/` | Original Colab notebooks + trained weights + inference/GradCAM scripts |

---

## 3. How to run on Toubkal HPC

### 3.1 Submit training

```bash
sbatch slurm/slurm_baseline_vit/train_baseline_vit.slurm
```

### 3.2 Submit evaluation

```bash
sbatch slurm/slurm_baseline_vit/evaluate_baseline_vit.slurm
```

### 3.3 Interactive debug (optional)

```bash
uv run python baseline/train_baseline_vit.py --config configs/plain_vit_baseline_config.yaml --debug
```

### 3.4 Monitor

```bash
squeue -u amine.aitlaamim-ext
tail -f /home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-Baseline/plain_vit_baseline_001/slurm_JOBID.out
```

---

## 4. ⭐ Shared split — `split_indices.npz` (IMPORTANT)

The baseline training **always creates** a shared split file that
**ALL future models** (DSS-ViT, DSCA-ViT, etc.) should reuse so every
experiment uses the **exact same train/val/test split**.

### Path to the split file

```
/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/ViT-Baseline/plain_vit_baseline_001/split_indices.npz
```

### What the file contains

| Key | Description |
|-----|-------------|
| `train_indices` | Indices into the TRAIN set (full train set when `val_fraction=0.0`) |
| `val_indices` | Indices into the TRAIN set (validation holdout; empty when `val_fraction=0.0`) |
| `test_indices` | Indices into the TEST set |
| `val_fraction` | The validation fraction used (0.0 for this baseline) |
| `seed` | The random seed used (42) |

### How to load it in a future model

```python
import numpy as np

data = np.load(
    "/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/"
    "ViT-Baseline/plain_vit_baseline_001/split_indices.npz"
)

train_indices = data["train_indices"]
val_indices   = data["val_indices"]
test_indices  = data["test_indices"]
```

### Why this matters

- **Fair comparison**: every model is trained and evaluated on the same images.
- **Reproducibility**: the split is deterministic (seed 42) and saved once.
- **No accidental leakage**: the test set is never used for training or
  hyperparameter selection.

---

## 5. Outputs produced by training

| Path | Contents |
|------|----------|
| `checkpoints/ViT-Baseline/plain_vit_baseline_001/best_stage1.pt` | Best Stage 1 model |
| `checkpoints/ViT-Baseline/plain_vit_baseline_001/best_stage2.pt` | Best Stage 2 model (used by evaluation) |
| `checkpoints/ViT-Baseline/plain_vit_baseline_001/last.pt` | Latest state (used by `--resume`) |
| `logs/ViT-Baseline/plain_vit_baseline_001/train.log` | Training log |
| `logs/ViT-Baseline/plain_vit_baseline_001/metrics.jsonl` | Per-epoch metrics |
| `results/ViT-Baseline/plain_vit_baseline_001/test_results.json` | Evaluation results |
| `results/ViT-Baseline/plain_vit_baseline_001/test_report.txt` | Human-readable report |
| `experiments/ViT-Baseline/plain_vit_baseline_001/split_indices.npz` | **Shared split (train/val/test)** |

---

## 6. Config

All hyperparameters and paths live in:

```
configs/plain_vit_baseline_config.yaml
```

This config exactly matches the original Colab notebook settings
(Adam, batch 32, no label smoothing, no AMP, validation on the test set).