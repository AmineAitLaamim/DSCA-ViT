# UNI-Regularized-v2 — Stronger Fine-Tuning Regularization

Follow-up to [`UNI-Regularized`](../UNI-Regularized/) (v1), targeting the same
diagnosed problem:

> Train loss still collapsed to near-zero too early, and the previous
> regularization did not change the overfitting curve enough.

This run bundles **three regularization changes together** (per the experiment
spec, they are **not** disentangled in this run):

| # | Change | v1 | v2 |
|---|--------|----|----|
| 1 | Stochastic depth (`drop_path_rate`) | absent | `0.2` |
| 2 | Head dropout | none | `nn.Dropout(0.3)` before the `Linear` head |
| 3 | Stage 2 backbone weight decay | `0.05` | `0.5` |

Everything else is **identical to v1**: the 2-stage recipe, the shared WSI split
file, transforms, early stopping, and checkpoint selection.

## 1. The three regularization changes

### 1.1 Stochastic depth — `drop_path_rate=0.2`

The UNI backbone is created with

```python
self.backbone = timm.create_model(
    "vit_large_patch16_224",
    img_size=224,
    patch_size=16,
    init_values=1e-5,
    num_classes=0,
    dynamic_img_size=True,
    drop_path_rate=0.2,          # NEW in v2 — configurable
)
```

`drop_path_rate` adds `DropPath` modules that randomly drop residual
sub-blocks during training. It does **not** add or remove parameters, so the
strict `load_state_dict(state_dict, strict=True)` from the official UNI
checkpoint remains unaffected.

### 1.2 Head dropout

The simple `nn.Linear(1024, num_classes)` head is replaced with:

```python
self.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1024, num_classes),
)
```

No extra hidden layers — no change in capacity.

### 1.3 Rebalanced backbone weight decay

Stage 2 uses a **different** weight decay for backbone and head so that the
effective decay strength (`lr * wd`) is equalized:

```text
backbone: lr 1e-5  × wd 0.5  = 5.00e-06
head    : lr 1e-4  × wd 0.05 = 5.00e-06
```

Both products are `5.00e-06` — parity. Without this, the much larger head
learning rate would have given the head disproportionately strong decay
relative to its update magnitude; the rebalancing makes the relative
regularization pressure consistent across the whole model. Stage 1 retains
`weight_decay_stage1 = 0.0` exactly as v1.

## 2. Why backbone wd = 0.5 (LR × wd parity)

AdamW decouples weight decay from the gradient step: each step decays weights
by `wd` regardless of the learning rate schedule. In this recipe the head runs
at `1e-4` and the backbone at `1e-5` (10× difference), so if both used
`wd = 0.05`, the head would receive 10× more decay per optimizer step than the
backbone relative to its update scale. Raising backbone wd to `0.5` gives both
`lr * wd = 5.00e-06`, i.e. equal regularization strength relative to the
per-update step size.

## 3. DropPath + head dropout stay active in Stage 1

`DropPath` and `nn.Dropout` are stochastic layers with **no parameters**.
Freezing the backbone (`requires_grad=False`) does not disable them — they
remain stochastically active throughout both stages.

This is **expected**: frozen UNI weights simply do not receive gradient
updates, but dropout / stochastic depth still impose random sub-network
sampling and feature-zeroing on the forward pass. Head dropout regularizes the
trainable head; backbone DropPath regularizes the frozen feature extractor's
forward path. Neither updates any backbone weights.

## 4. Logging — `Train Acc` is intentional

v1's console log omitted `Train Acc`. This run **adds** it by design, with the
exact per-epoch format:

```
Stage {stage} | Epoch [{epoch:02d}/{stage_epochs}] | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | QWK {qwk:.4f} | {elapsed:.1f}s
```

This makes the memorization pattern (train loss collapsing while train acc
saturates) directly visible in the console, which is the whole point of the
comparison against v1. After each stage the stop reason is logged:

- `"Stage N stopped: early stopping triggered at epoch K"`
- or `"Stage N stopped: reached max_epochs cap"`

At startup the effective decay strengths are logged so the `5.00e-06` parity is
auditable from the training log itself:

```
Effective backbone decay strength (lr*wd): 5.00e-06
Effective head decay strength (lr*wd): 5.00e-06
```

## 5. Success signal — the curve shape, not just the final accuracy

The primary evaluation criterion for this experiment is the **train/val loss
curve shape** compared against v1:

- Did train loss stop collapsing to near-zero too early?
- Did the gap between train loss and val loss shrink?
- Did train accuracy still saturate to ~100% while train loss drops to ~0?

Final test accuracy is secondary. If the curves still look like v1
(memorization), the regularization bundle did not address the diagnosed
problem.

## 6. Files

```
UNI-Regularized-v2/
├── __init__.py
├── uni_regularized_v2_model.py        # UNI backbone (drop_path 0.2) + dropout head
├── train_uni_regularized_v2.py        # Training CLI (--resume / --debug / --distributed)
├── evaluate_uni_regularized_v2.py     # Official-test evaluation (best_stage2.pt only)
└── README.md

configs/uni_regularized_v2_config.yaml
slurm/slurm_uni_regularized_v2/
├── train_uni_regularized_v2.slurm
└── evaluate_uni_regularized_v2.slurm
```

## 7. Usage

```bash
# Train (2 stages; Stage 2 auto-initializes from best_stage1.pt)
uv run python UNI-Regularized-v2/train_uni_regularized_v2.py \
    --config configs/uni_regularized_v2_config.yaml

# Debug sanity checks (DropPath active, head dropout, Stage 1/2 requires_grad)
uv run python UNI-Regularized-v2/train_uni_regularized_v2.py \
    --config configs/uni_regularized_v2_config.yaml --debug

# Resume from last.pt
uv run python UNI-Regularized-v2/train_uni_regularized_v2.py \
    --config configs/uni_regularized_v2_config.yaml --resume

# Official test evaluation (loads best_stage2.pt, NEVER last.pt)
uv run python UNI-Regularized-v2/evaluate_uni_regularized_v2.py \
    --config configs/uni_regularized_v2_config.yaml
```

HPC (Toubkal / UM6P, A100):

```bash
sbatch slurm/slurm_uni_regularized_v2/train_uni_regularized_v2.slurm
sbatch slurm/slurm_uni_regularized_v2/evaluate_uni_regularized_v2.slurm
```

## 8. Data & split

Identical to v1:

- Split file (load-only, never regenerated):
  `.../plain_vit_baseline_001/split_indices_wsi.npz`
  (train 7283 | val 810 | test 1847, `val_fraction=0.10`, `seed=42`)
- Dataloader returns raw RGB `[0,1]`; the model applies ImageNet normalization
  internally.
- Train transforms: `Resize(224)` + h/v flips + `RandomRotation(10, bilinear,
  fill=0)`.
- Val/test transforms: `Resize(224)` only.

## 9. Naming

| Item | Value |
|------|-------|
| Experiment | `uni_regularized_v2_001` |
| Checkpoint dir | `checkpoints/UNI-Regularized-v2/uni_regularized_v2_001/` |
| Log dir | `logs/UNI-Regularized-v2/uni_regularized_v2_001/` |
| Results dir | `results/UNI-Regularized-v2/uni_regularized_v2_001/` |