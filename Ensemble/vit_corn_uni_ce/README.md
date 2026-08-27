# Ensemble — vit_corn_uni_ce

**Equal-weight probability averaging of ViT-B16 CORN + UNI CE, no TTA.**

This variant was **added after the original four ensemble variants** to
test a different model combination: the ordinal ViT-B16 CORN paired with
the under-regularized UNI CE baseline, without the ViT-B16 CE member.

| Model | Class | Checkpoint |
|---|---|---|
| ViT-B16 CORN | `ViTB16CORN` | `checkpoints/ViT-B16-CORN/vit_b16_corn_001/best_stage2.pt` (94.26%) |
| UNI CE | `UNIBaselineModel` | `checkpoints/UNI-Baseline/uni_baseline_001/best_stage2.pt` (94.69%) |

## TTA
**Off.**

## Use only `probs`
Ensembling uses **only probabilities — never raw logits**. For ViT-B16
CORN this is its already-implemented chain-rule-reconstructed `[B,4]`
probs, **not** its raw `[B,3]` conditional logits.

## Transforms
- ViT-B16 CORN: `Resize(224) -> ToTensor -> Normalize(ImageNet)`
- UNI CE: `Resize(224) -> ToTensor` (raw RGB; UNI normalizes internally)

## Ensemble
```python
ensemble_probs = (probs_corn + probs_uni) / 2
ensemble_preds  = argmax(ensemble_probs, dim=1)
```

## How to run
Validation (default):
```bash
uv run python Ensemble/vit_corn_uni_ce/evaluate_vit_corn_uni_ce.py \
    --config configs/ensemble_config.yaml
```
Official test (once, locked):
```bash
uv run python Ensemble/vit_corn_uni_ce/evaluate_vit_corn_uni_ce.py \
    --config configs/ensemble_config.yaml --eval-test
```
Debug sanity checks (offline):
```bash
uv run python Ensemble/vit_corn_uni_ce/evaluate_vit_corn_uni_ce.py \
    --config configs/ensemble_config.yaml --debug
```

Artifacts (nested two levels):
```
logs/Ensemble/vit_corn_uni_ce/vit_corn_uni_ce_001/
results/Ensemble/vit_corn_uni_ce/vit_corn_uni_ce_001/{val,test}_results.{json,txt}
```

## Reference
Best single-model test accuracy to beat: **94.69%**.