# UNI-CORN - Ordinal UNI Baseline

Combines the UNI histopathology foundation model with CORN (Conditional
Ordinal Regression for Neural networks) loss. Tests whether ordinal
supervision improves boundary classes (class_1+ recall, class_2+
precision, macro-F1).

| Model | Test Acc | Bal Acc | Macro F1 |
|-------|----------|---------|----------|
| Proper ViT-B16 baseline | 94.69% | 93.58% | 91.11% |
| UNI original fine-tune | 94.69% | 92.77% | 91.35% |
| ViT-B16 CORN | 94.26% | 93.52% | 92.03% |
| UNI-Regularized | 93.23% | 90.96% | 87.43% |
| UNI-Stain-Attention | 93.77% | 90.60% | 88.87% |
| **UNI-CORN (this)** | **TBD** | **TBD** | **TBD** |

## Model

- UNI backbone (exact recipe, strict local load) + `nn.Linear(1024, 3)` head.
- Internal ImageNet normalization; dataloader returns raw RGB [0,1].
- probs [B,4] reconstructed from CORN sigmoids (sums to 1).
- Predictions always via `corn_label_from_logits`; `torch.argmax` banned.

## Training (original UNI-baseline recipe)

- Stage 1: frozen UNI, head-only Adam 1e-4, 30 ep.
- Stage 2: full fine-tune, Adam backbone 1e-5 / head 1e-4, 30 ep.
- Adam (NOT AdamW), wd 0.0, batch 32, AMP off, no early stopping.
- Loss: `corn_loss(logits, labels, num_classes=4)`.
- Stage 2 initializes from `best_stage1.pt` (logged).

## Commands

```bash
uv run python UNI-CORN/train_uni_corn.py --config configs/uni_corn_config.yaml
uv run python UNI-CORN/evaluate_uni_corn.py --config configs/uni_corn_config.yaml
```

## Differs from

- UNI-baseline: adds CORN ordinal loss (3-logit head).
- ViT-B16-CORN: uses UNI backbone plus internal normalization.