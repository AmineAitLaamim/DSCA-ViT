# Ensemble — vit_ce_uni_ce

**Equal-weight probability averaging of two complementary CE-trained models, no TTA.**

| Model | Class | Checkpoint |
|---|---|---|
| ViT-B16 CE | `PlainViTB16` | `checkpoints/ViT-Baseline/plain_vit_baseline_001/best_stage2.pt` (94.69%) |
| UNI CE | `UNIBaselineModel` | `checkpoints/UNI-Baseline/uni_baseline_001/best_stage2.pt` (94.69%) |

## TTA
**Off.** Only `probs` are averaged — never logits.

## Transforms
- ViT-B16 CE: `Resize(224) -> ToTensor -> Normalize(ImageNet)`
- UNI CE: `Resize(224) -> ToTensor` (raw RGB; UNI normalizes internally)

## Ensemble
```python
ensemble_probs = (probs_vit + probs_uni) / 2
ensemble_preds  = argmax(ensemble_probs, dim=1)
```

## How to run
Validation (default):
```bash
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config configs/ensemble_config.yaml
```
Official test (once, locked):
```bash
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config configs/ensemble_config.yaml --eval-test
```
Debug sanity checks:
```bash
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config configs/ensemble_config.yaml --debug
```

Artifacts (nested two levels):
```
logs/Ensemble/vit_ce_uni_ce/vit_ce_uni_ce_001/
results/Ensemble/vit_ce_uni_ce/vit_ce_uni_ce_001/{val,test}_results.{json,txt}
```

## Why UNI-CORN is excluded
UNI-CORN showed a negative interaction with under-regularized UNI
fine-tuning: CORN's conditional training doubled the 1+ → 2+
false-positive rate on that backbone. It is therefore left out of all
four ensemble variants.

## Reference
Best single-model test accuracy to beat: **94.69%**.
