# ViT-B16-CORN — Ordinal Regression Baseline

A baseline model using a **ViT-B16 backbone with CORN** (Conditional
Ordinal Regression for Neural networks) loss for ordinal HER2
classification.

This is a **single-variable change** from the existing proper ViT-B16
baseline: only the output layer and loss are changed from standard 4-way
cross-entropy to CORN ordinal regression.

The hypothesis: explicitly encoding the HER2 severity order
(`0 < 1+ < 2+ < 3+`) improves boundary-class performance, especially:

- `class_1+` recall
- `class_2+` precision
- Balanced accuracy / macro-F1

## Reference results

| Model | Test Accuracy |
|-------|---------------|
| Proper ViT-B16 baseline | 94.69% |
| UNI Stage 2 fine-tuned | 94.69% |
| DSS-ViT v2.2 | 93.18% |
| UNI-Stain-MLP | 92.91% |
| UNI-RGB-MLP | 93.23% |
| **ViT-B16-CORN (this)** | **TBD** |

---

## 1. Why CORN?

Standard 4-way cross-entropy treats `class_0`, `class_1+`, `class_2+`,
`class_3+` as unrelated nominal categories. CORN instead models the
ordinal structure directly: the head outputs `num_classes - 1 = 3`
logits, each representing `P(y > k | y > k-1)`, and reconstructs genuine
unconditional class probabilities from those sigmoids.

### Model

```
features = ViT-B16(ImageNet-normalized RGB)   # [B, 768]
logits   = Linear(768, 3)                      # [B, 3]  (num_classes - 1)
s = sigmoid(logits)                            # [B, 3]
p_gt0, p_gt1, p_gt2 = s[:,0], s[:,1], s[:,2]
p0 = 1 - p_gt0
p1 = p_gt0 * (1 - p_gt1)
p2 = p_gt0 * p_gt1 * (1 - p_gt2)
p3 = p_gt0 * p_gt1 * p_gt2
probs = stack([p0, p1, p2, p3])               # [B, 4], sums to 1
```

Predictions always use `corn_label_from_logits(logits)` — the official
coral-pytorch function. `torch.argmax` never appears in this package.

## 2. Training recipe (identical to proper ViT-B16 baseline)

| Stage | Freeze | Trainable | LR | Epochs |
|-------|--------|-----------|-----|--------|
| 1 | ViT backbone | CORN head only | 1e-4 | 30 |
| 2 | None | Entire model | Backbone 1e-5, Head 1e-4 | 30 |

- Optimizer: Adam, wd 0.0, batch 32, AMP off, no gradient clipping
- Loss: `corn_loss(logits, labels, num_classes=4)`
- Scheduler: CosineAnnealingLR(T_max=epochs), recreated per stage
- Validation: every epoch, select best by validation accuracy

## 3. Data

- Load `split_indices_wsi.npz` (train 7,283 / val 810 / test 1,847).
- ImageNet normalization **in the dataloader** (matches proper baseline).
- Dataloader returns integer labels `0,1,2,3`.

## 4. Commands

```bash
uv run python ViT-B16-CORN/train_vit_b16_corn.py --config configs/vit_b16_corn_config.yaml
uv run python ViT-B16-CORN/train_vit_b16_corn.py --config configs/vit_b16_corn_config.yaml --debug
uv run python ViT-B16-CORN/train_vit_b16_corn.py --config configs/vit_b16_corn_config.yaml --resume
uv run python ViT-B16-CORN/evaluate_vit_b16_corn.py --config configs/vit_b16_corn_config.yaml
```

On HPC:
```bash
sbatch slurm/slurm_vit_b16_corn/train_vit_b16_corn.slurm
sbatch slurm/slurm_vit_b16_corn/evaluate_vit_b16_corn.slurm
```

## 5. Outputs

| Path | Contents |
|------|----------|
| `checkpoints/ViT-B16-CORN/vit_b16_corn_001/best_stage1.pt` | Best val acc Stage 1 |
| `checkpoints/ViT-B16-CORN/vit_b16_corn_001/best_stage2.pt` | Best val acc Stage 2 (eval uses this) |
| `checkpoints/ViT-B16-CORN/vit_b16_corn_001/last.pt` | Latest state (`--resume`) |
| `logs/ViT-B16-CORN/vit_b16_corn_001/...` | Training logs |
| `results/ViT-B16-CORN/vit_b16_corn_001/test_results.json` | Test metrics (JSON) |
| `results/ViT-B16-CORN/vit_b16_corn_001/test_report.txt` | Human-readable report |

## 6. Notes

- Uses the official `coral-pytorch` (already installed).
- `torch.argmax` banned in the package (debug static scan enforces this).
- MAE reported alongside the standard metric suite.
- No existing packages modified.