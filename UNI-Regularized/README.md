# UNI-Regularized — Regularized Fine-Tuning Baseline

A new self-contained experiment using the **UNI backbone** with the same
architecture as the existing `UNI-baseline`, but with a **regularized
fine-tuning recipe** (AdamW + patience-based early stopping on validation
loss) and a **corrected Stage 2 initialization**.

This experiment isolates one question:

> Does a fine-tuning recipe better suited to UNI's 303M-parameter size
> produce a genuine test accuracy above the current UNI-baseline 94.69%,
> or was 94.69% already close to UNI's ceiling and the overfitting mostly
> cosmetic?

## Reference results

| Model | Test Accuracy |
|-------|---------------|
| Proper ViT-B16 baseline | 94.69% |
| UNI-baseline Stage 2 (Adam, no wd) | 94.69% |
| DSS-ViT v2.2 | 93.18% |
| UNI-Stain-MLP | 92.91% |
| UNI-RGB-MLP | 93.23% |
| **UNI-Regularized (this)** | **TBD** |

---

## 1. What changes (bundled as one change this run)

1. **AdamW instead of Adam** — with a decay/no-decay split: weight decay
   is excluded from all 1D params (LayerNorm weights/biases, other
   biases).
2. **Patience-based early stopping on validation loss** — patience 7,
   min_delta 1e-4. Checkpoint selection remains keyed to **validation
   accuracy** (unchanged). Counters reset at the Stage 1 → Stage 2
   transition.

## 2. Pre-flight Stage 2 initialization correction

Inspecting `baseline/UNI-baseline/train_uni_baseline.py` revealed the
original UNI-baseline **does not reload `best_stage1.pt`** at the start
of Stage 2. It simply carries the model over from the final Stage-1 epoch
(equivalent to `stage1_end.pt` / `last.pt` weights). `best_stage1.pt` is
written during Stage 1 but never read again.

This build **corrects that**: Stage 2 explicitly loads
`checkpoint_dir/best_stage1.pt` (key `"model_state_dict"`, verified from
the actual `save_checkpoint()` source) and logs
`"Stage 2 initialized from best_stage1.pt"`. Never `last.pt`.

This ties back to the original overfitting diagnosis with a second,
more concrete mechanism: Stage 2 wasn't just following an ill-suited
LR/epoch recipe — it was also initialized from drifted late-epoch Stage-1
weights rather than the actual best Stage-1 checkpoint. Even if val
accuracy plateaued at 95.80% from epoch 18 in the original run, the
weights themselves could still differ at matching val accuracy, and this
fix removes that uncertainty.

## 3. Model

Same architecture as `UNI-baseline/uni_baseline_model.py` (renamed
`UNIRegularizedModel`): exact UNI recipe, strict local checkpoint load,
internal ImageNet normalization in `forward()`, `nn.Linear(1024, 4)` head,
returns `{"logits": [B,4], "probs": [B,4]}`. No components added/removed.
Added only helper methods `freeze_backbone()` / `unfreeze_backbone()`.

## 4. Training recipe

| Stage | Freeze | Trainable | Optimizer | LR | Weight decay | Epochs | Early stop |
|-------|--------|-----------|-----------|-----|--------------|--------|-----------|
| 1 | UNI backbone | Head only | AdamW | 1e-4 | 0.0 | max 30 | patience 7, min_delta 1e-4 on val loss |
| 2 | None | Entire model | AdamW | Backbone 1e-5, Head 1e-4 | 0.05 non-1D only | max 30 | same |

Early stopping uses **validation loss**; checkpoint selection uses
**validation accuracy**. `stage1_epochs: 30` / `stage2_epochs: 30` are
hard caps only.

## 5. Commands

```bash
uv run python UNI-Regularized/train_uni_regularized.py --config configs/uni_regularized_config.yaml
uv run python UNI-Regularized/train_uni_regularized.py --config configs/uni_regularized_config.yaml --debug
uv run python UNI-Regularized/train_uni_regularized.py --config configs/uni_regularized_config.yaml --resume
uv run python UNI-Regularized/evaluate_uni_regularized.py --config configs/uni_regularized_config.yaml
```

On HPC:
```bash
sbatch slurm/slurm_uni_regularized/train_uni_regularized.slurm
sbatch slurm/slurm_uni_regularized/evaluate_uni_regularized.slurm
```

## 6. Outputs

| Path | Contents |
|------|----------|
| `checkpoints/UNI-Regularized/uni_regularized_001/best_stage1.pt` | Best val acc Stage 1 |
| `checkpoints/UNI-Regularized/uni_regularized_001/best_stage2.pt` | Best val acc Stage 2 (eval uses this) |
| `checkpoints/UNI-Regularized/uni_regularized_001/last.pt` | Latest state (`--resume`) |
| `logs/UNI-Regularized/uni_regularized_001/...` | Training logs (incl. stop reasons) |
| `results/UNI-Regularized/uni_regularized_001/test_results.json` | Test metrics (JSON) |
| `results/UNI-Regularized/uni_regularized_001/test_report.txt` | Human-readable report |

## 7. `--debug`

- `HF_HUB_OFFLINE=1` before imports; static scan for forbidden strings.
- Random raw RGB [B,3,224,224]; logits/probs [B,4]; `loss.backward()`;
  no NaN/Inf.
- Asserts `freeze_backbone()` / `unfreeze_backbone()` behavior.
- Prints backbone / head / total trainable counts.
- Mocked Stage-2 init check: saves `best_stage1.pt` with current weights,
  zeroes the head, reloads `best_stage1.pt`, and verifies exact tensor
  restoration via `torch.allclose` — no epochs required.

## 8. Notes

- Split: load-only `split_indices_wsi.npz` (7,283 / 810 / 1,847).
- Fully offline: local UNI checkpoint only.
- No existing packages modified.