# Ensemble — vit_ce_uni_ce_vit_corn

**Equal-weight probability averaging of three complementary models, no TTA.**

| Model | Class | Checkpoint |
|---|---|---|
| ViT-B16 CE | `PlainViTB16` | `checkpoints/ViT-Baseline/plain_vit_baseline_001/best_stage2.pt` (94.69%) |
| UNI CE | `UNIBaselineModel` | `checkpoints/UNI-Baseline/uni_baseline_001/best_stage2.pt` (94.69%) |
| ViT-B16 CORN | `ViTB16CORN` | `checkpoints/ViT-B16-CORN/vit_b16_corn_001/best_stage2.pt` (94.26%) |

## TTA
**Off.**

## Use only `probs`
Ensembling uses **only probabilities — never raw logits**. For CORN
this is its already-implemented chain-rule-reconstructed `[B,4]` probs,
**not** its raw `[B,3]` conditional logits.

## Transforms
- ViT-B16 CE / ViT-B16 CORN: `Resize(224) -> ToTensor -> Normalize(ImageNet)`
- UNI CE: `Resize(224) -> ToTensor` (raw RGB; UNI normalizes internally)

## Ensemble
```python
ensemble_probs = (probs_vit + probs_uni + probs_corn) / 3
ensemble_preds  = argmax(ensemble_probs, dim=1)
```

## How to run
Validation (default):
```bash
uv run python Ensemble/vit_ce_uni_ce_vit_corn/evaluate_vit_ce_uni_ce_vit_corn.py \
    --config configs/ensemble_config.yaml
```
Official test (once, locked):
```bash
uv run python Ensemble/vit_ce_uni_ce_vit_corn/evaluate_vit_ce_uni_ce_vit_corn.py \
    --config configs/ensemble_config.yaml --eval-test
```

Artifacts:
```
logs/Ensemble/vit_ce_uni_ce_vit_corn/vit_ce_uni_ce_vit_corn_001/
results/Ensemble/vit_ce_uni_ce_vit_corn/vit_ce_uni_ce_vit_corn_001/{val,test}_results.{json,txt}
```

## Why UNI-CORN is excluded
UNI-CORN showed a negative interaction with under-regularized UNI
fine-tuning: CORN's conditional training doubled the 1+ → 2+
false-positive rate on that backbone. It is therefore left out of all
four ensemble variants.

## Reference
Best single-model test accuracy to beat: **94.69%**.
