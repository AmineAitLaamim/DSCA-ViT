# Ensemble — Probability-Averaging Evaluation of Trained HER2 Models

This self-contained package evaluates **ensembles of already-trained
models** on the HER2-IHC-40x dataset. **No training is performed.** It
loads three existing `best_stage2.pt` checkpoints and answers one
question:

> Does simple probability averaging of complementary models, optionally
> with TTA, beat the best single-model test accuracy of **94.69%**?

Only **probabilities** are averaged — never raw logits. For CORN this
means its chain-rule-reconstructed `[B,4]` probs, not its raw `[B,3]`
conditional logits.

## Four variants

| Variant (code folder) | Constituents | TTA | Experiment |
|---|---|---|---|
| [`vit_ce_uni_ce/`](./vit_ce_uni_ce/) | ViT-B16 CE + UNI CE | No | `vit_ce_uni_ce_001` |
| [`vit_ce_uni_ce_tta/`](./vit_ce_uni_ce_tta/) | ViT-B16 CE + UNI CE | Yes | `vit_ce_uni_ce_tta_001` |
| [`vit_ce_uni_ce_vit_corn/`](./vit_ce_uni_ce_vit_corn/) | ViT-B16 CE + UNI CE + ViT-B16 CORN | No | `vit_ce_uni_ce_vit_corn_001` |
| [`vit_ce_uni_ce_vit_corn_tta/`](./vit_ce_uni_ce_vit_corn_tta/) | ViT-B16 CE + UNI CE + ViT-B16 CORN | Yes | `vit_ce_uni_ce_vit_corn_tta_001` |

## Layout

```
Ensemble/
├── common/            # shared model loader, transforms/TTA, metrics,
│                      #   dataset utils, inference, test lock, debug
├── vit_ce_uni_ce/            evaluate_vit_ce_uni_ce.py
├── vit_ce_uni_ce_tta/        evaluate_vit_ce_uni_ce_tta.py
├── vit_ce_uni_ce_vit_corn/   evaluate_vit_ce_uni_ce_vit_corn.py
├── vit_ce_uni_ce_vit_corn_tta/ evaluate_vit_ce_uni_ce_vit_corn_tta.py
├── compare_val_results.py    optional cross-variant comparison table
└── README.md
```

Each variant builds its own **nested two-level** artifacts:
```
logs/Ensemble/<variant_name>/<experiment_name>/
results/Ensemble/<variant_name>/<experiment_name>/
   ├── val_results.json / val_report.txt     (default run)
   └── test_results.json / test_report.txt   (--eval-test)
```

## Validation-first, test-once protocol

1. **Validation first (default).** Each run evaluates on the 810
   `val_indices` images held out from `train_dir` and writes
   `val_results.json` / `val_report.txt`.
2. **Official test once (`--eval-test`).** The full `test_dir` is loaded

## Why UNI-CORN is excluded

All four variants deliberately omit `UNI-CORN`. On the **original,
under-regularized UNI fine-tune** backbone, CORN's conditional training
**doubled the 1+ → 2+ false-positive rate**. Since every ensemble here
uses that under-regularized UNI backbone, including UNI-CORN would drag
down the ensemble via that pathological confusion, so it is left out
rather than left as an unexplained omission.

## Why the original UNI-baseline checkpoint (94.69%), not UNI-Regularized-v2 (93.39%)

The UNI member in all four variants is the **original** UNI-baseline
checkpoint (94.69%), even though it is known to come from an overfit
selection process, rather than `UNI-Regularized-v2` (93.39%, healthier
training dynamics but lower solo accuracy).

This is a **deliberate decision, not an oversight**: ensembles can
benefit from a strong individual member regardless of how it got there.
Voting/averaging is robust to a member's idiosyncrasies as long as its
confidence is well-calibrated on its correct classes; the goal here is
to test whether complementary strong signals (ViT-B16 CE + under-regularized
UNI CE, plus the ordinal ViT-B16 CORN) combine to beat 94.69%.

## Data-loading notes

### `val_indices` double-dip (mild, disclosed)

The same 810 `val_indices` images are used both as part of each
constituent model's own checkpoint selection during training **and**
here to choose among the four ensemble variants. This is a mild
double-dip on 810 images. It is low-risk given only 4 discrete options
are being compared, but it is stated explicitly rather than left
implicit.

### `test_indices` is the identity mapping (why `test_dir` is loaded directly)

The official test set is loaded by passing `test_dir` **directly** to
`HER2Dataset` — it is **not indexed by `test_indices`**. This is valid
because `test_indices` in `split_indices_wsi.npz` is confirmed to be the
identity mapping `[0, 1, ..., 1846]` over the 1847 images in `test_dir`,
so loading the folder directly and indexing it with `test_indices` are
mathematically identical. Loading the folder directly is simpler and
matches the pattern in `evaluate_baseline_vit.py` /
`evaluate_uni_baseline.py`.

A **permanent guard** in `common/dataset_utils.py::load_split()` asserts
`np.array_equal(test_indices, np.arange(len(test_indices)))` and logs the
result. If the split file is ever regenerated differently, this fails
immediately instead of silently reintroducing the ambiguity that the

## Commands

```bash
# Validation (default) — one command per variant
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py --config configs/ensemble_config.yaml
uv run python Ensemble/vit_ce_uni_ce_tta/evaluate_vit_ce_uni_ce_tta.py --config configs/ensemble_config.yaml
uv run python Ensemble/vit_ce_uni_ce_vit_corn/evaluate_vit_ce_uni_ce_vit_corn.py --config configs/ensemble_config.yaml
uv run python Ensemble/vit_ce_uni_ce_vit_corn_tta/evaluate_vit_ce_uni_ce_vit_corn_tta.py --config configs/ensemble_config.yaml

# Official test (once, locked) — add --eval-test
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config configs/ensemble_config.yaml --eval-test
# ... (same pattern for the other three)

# Compare validation results across all four variants
uv run python Ensemble/compare_val_results.py --config configs/ensemble_config.yaml

# Debug sanity checks (offline)
uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config configs/ensemble_config.yaml --debug
```

HPC (Toubkal / UM6P, 1 A100 each):
```bash
sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce.slurm
sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce_tta.slurm
sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce_vit_corn.slurm
sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce_vit_corn_tta.slurm
```

## Reference

Best single-model test accuracy to beat: **94.69%** (see
`PROJECT_DEVELOPMENT_LOG.md`).

direct-loading decision rules out.

## Checkpoint key

All checkpoints store weights under **`"model_state_dict"`** (confirmed
from the project's `save_checkpoint()` sources). `common/model_loader.py`
hardcodes that key and asserts it at load time so a format change fails
loudly.

## Offline guarantee

`HF_HUB_OFFLINE=1` is set at the very top of every variant script before
any imports, and `--debug` performs a static scan of `Ensemble/` for
forbidden tokens: `hf_hub_download`, `huggingface_hub.login`, `HF_TOKEN`,
`from_pretrained`. No network calls.

   directly and evaluated. Before running, a shared lock file
   `results/Ensemble/.test_eval_lock.json` is checked:
   - if it names a **different** variant, the run **refuses** unless
     `--force` is passed;
   - if it does not exist, or already names the current variant, the run
     proceeds and writes/updates the lock.
3. `--debug` runs offline sanity checks instead of a full evaluation
   (see `common/debug_checks.py`).

Constants for each run are passed from the variant script; the shared
driver is `common/evaluator.py::run_variant()`.
