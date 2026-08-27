# Ensemble — vit_corn_uni_ce_tta

**Equal-weight probability averaging of ViT-B16 CORN + UNI CE with test-time augmentation (TTA).**

This variant was **added after the original four ensembles** to test a
different combination: the ViT-B16 CORN paired with the
under-regularized UNI CE baseline, with TTA.

| Model | Class | Checkpoint |
|---|---|---|
| ViT-B16 CORN | `ViTB16CORN` | `checkpoints/ViT-B16-CORN/vit_b16_corn_001/best_stage2.pt` (94.26%) |
| UNI CE | `UNIBaselineModel` | `checkpoints/UNI-Baseline/uni_baseline_001/best_stage2.pt` (94.69%) |

## TTA — ON
Each model's per-sample probabilities are averaged over **6 augmentations**:
1. Original
2. Horizontal flip
3. Vertical flip
4. Rotate 90°
5. Rotate 180°
6. Rotate 270°

Per-model augmentations happen **before** normalization:
- ViT-B16 CORN: `Resize -> flip/rotate -> ToTensor -> Normalize(ImageNet)`
- UNI CE: `Resize -> flip/rotate -> ToTensor` (raw RGB)

Only **probabilities** are averaged across augmentations, then across models.

## Use only `probs`
Ensembling uses **only probabilities — never raw logits**. For ViT-B16
CORN this is its already-implemented chain-rule-reconstructed `[B,4]`
probs, **not** its raw `[B,3]` conditional logits.

## Ensemble
```python
# per model: probs_tta = mean over 6 augs of probs
ensemble_probs = (probs_corn_tta + probs_uni_tta) / 2
ensemble_preds  = argmax(ensemble_probs, dim=1)
```

## How to run
Validation (default):
```bash
uv run python Ensemble/vit_corn_uni_ce_tta/evaluate_vit_corn_uni_ce_tta.py \
    --config configs/ensemble_config.yaml
```
Official test (once, locked):
```bash
uv run python Ensemble/vit_corn_uni_ce_tta/evaluate_vit_corn_uni_ce_tta.py \
    --config configs/ensemble_config.yaml --eval-test
```

Artifacts:
```
logs/Ensemble/vit_corn_uni_ce_tta/vit_corn_uni_ce_tta_001/
results/Ensemble/vit_corn_uni_ce_tta/vit_corn_uni_ce_tta_001/{val,test}_results.{json,txt}
```

## Reference
Best single-model test accuracy to beat: **94.69%**.