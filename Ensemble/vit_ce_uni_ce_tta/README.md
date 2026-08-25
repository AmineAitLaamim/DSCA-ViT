# Ensemble — vit_ce_uni_ce_tta

**Equal-weight probability averaging of ViT-B16 CE + UNI CE with test-time augmentation (TTA).**

| Model | Class | Checkpoint |
|---|---|---|
| ViT-B16 CE | `PlainViTB16` | `checkpoints/ViT-Baseline/plain_vit_baseline_001/best_stage2.pt` (94.69%) |
| UNI CE | `UNIBaselineModel` | `checkpoints/UNI-Baseline/uni_baseline_001/best_stage2.pt` (94.69%) |

## TTA — ON
Each model's per-sample probabilities are averaged over **6 augmentations**:
1. Original
2. Horizontal flip
3. Vertical flip
4. Rotate 90°
5. Rotate 180°
6. Rotate 270°

The per-model augmentations happen **before** normalization:
- ViT-B16 CE: `Resize -> flip/rotate -> ToTensor -> Normalize(ImageNet)`
- UNI CE: `Resize -> flip/rotate -> ToTensor` (raw RGB)

Only **probabilities** are averaged across augmentations (never logits),
then averaged across models.

## Ensemble
```python
# per model: probs_tta = mean over 6 augs of probs
ensemble_probs = (probs_vit_tta + probs_uni_tta) / 2
ensemble_preds  = argmax(ensemble_probs, dim=1)
```

## How to run
Validation (default):
```bash
uv run python Ensemble/vit_ce_uni_ce_tta/evaluate_vit_ce_uni_ce_tta.py \
    --config configs/ensemble_config.yaml
```
Official test (once, locked):
```bash
uv run python Ensemble/vit_ce_uni_ce_tta/evaluate_vit_ce_uni_ce_tta.py \
    --config configs/ensemble_config.yaml --eval-test
```

Artifacts:
```
logs/Ensemble/vit_ce_uni_ce_tta/vit_ce_uni_ce_tta_001/
results/Ensemble/vit_ce_uni_ce_tta/vit_ce_uni_ce_tta_001/{val,test}_results.{json,txt}
```

## Why UNI-CORN is excluded
UNI-CORN showed a negative interaction with under-regularized UNI
fine-tuning: CORN's conditional training doubled the 1+ → 2+
false-positive rate on that backbone. It is therefore left out of all
four ensemble variants.

## Reference
Best single-model test accuracy to beat: **94.69%**.
