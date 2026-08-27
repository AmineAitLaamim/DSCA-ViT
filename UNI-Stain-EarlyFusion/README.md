# UNI-Stain-EarlyFusion — Full Fine-Tuning with 5-Channel Input

**UNI (ViT-L/16, DINOv2, mass100k) modified to accept a 5-channel
input `[RGB + Hematoxylin + DAB]` via early fusion, then FULL
fine-tuning (Stage 2).**

Experiment: `uni_stain_earlyfusion_001`

## Model

- UNI backbone with the patch-embed projection **replaced** by a
  `Conv2d(5, ...)` so it ingests:
  ```
  [ normalized RGB | normalized H | normalized DAB ]   # [B, 5, 224, 224]
  ```
- RGB is ImageNet-normalized; H and DAB are deconvolved (fixed Ruifrok
  H-DAB matrix) and normalized with global train-split stats.
- 4-class linear head `Linear(1024, 4)`.
- `forward` returns `{"logits", "probs"}`.

### Early-fusion design
Stain information is fused **at the very first layer** and then flows
through every transformer block. No separate branch, no late fusion.

### Why no cross-attention is needed
Once the 5-channel image is patch-embedded, UNI's **self-attention**
fuses morphology and stain in every layer. An explicit cross-attention
module would add parameters for no new capability — the fusion already
happens inside the shared transformer.

### Critical loading order (strict-load-before-replace)
1. Create the ORIGINAL UNI backbone.
2. `load_state_dict(..., strict=True)` from the local UNI checkpoint.
3. Replace `patch_embed.proj` with `Conv2d(5, ...)`.
4. Copy pretrained RGB weights into channels `[0:3]`, copy bias,
   zero-init channels `[3:5]` (H/DAB).

After replacement the model can't be strict-loaded with the original
checkpoint. For inference we **rebuild the same** modified projection,
then load the saved **training** checkpoint (`best_stage2.pt`).

### H/DAB zero-init at start
At construction the H/DAB projection channels are zero. So in Stage 1
(frozen backbone) the model effectively sees RGB only — fine, since
Stage 1 trains only the head. In Stage 2 the whole backbone is
unfrozen and the stain channels become trainable, letting the model
learn to use stain information.

## Training recipe (original UNI baseline, no early stopping)

| Stage | Freeze | Trainable | LR | Epochs |
|---|---|---|---|---|
| 1 | Backbone | Head only | 1e-4 | 30 |
| 2 | None | Entire model | Backbone 1e-5, Head 1e-4 | 30 |

- Optimizer: **Adam**, weight decay 0.0
- Loss: CrossEntropyLoss (no label smoothing)
- Batch size: 32/GPU, AMP disabled, no grad clipping
- Scheduler: CosineAnnealingLR per stage
- Stage 2 initializes from `best_stage1.pt` (never `last.pt`)
- Checkpoints: `best_stage1.pt`, `best_stage2.pt`, `last.pt`,
  `stage1_end.pt`, `stage2_end.pt`

## Data policy

- Split file LOAD-ONLY: `split_indices_wsi.npz` (train 7283 | val 810 |
  test 1847; val_fraction 0.10, seed 42).
- Dataloader returns raw RGB `[0,1]`; NO normalization in dataloader.
- Train transforms: Resize(224), HFlip, VFlip, Rot10, ToTensor.
- Val/Test transforms: Resize(224), ToTensor.

## Commands

```bash
# Precompute H/DAB stats on the train split (required before training)
uv run python UNI-Stain-EarlyFusion/precompute_stain_stats.py \
    --config configs/uni_stain_earlyfusion_config.yaml

# Debug sanity checks (offline) — verifies strict-load-before-replace,
# 5-channel proj, RGB-slice == original UNI, H/DAB zero-init, forward/backward
uv run python UNI-Stain-EarlyFusion/train_uni_stain_earlyfusion.py \
    --config configs/uni_stain_earlyfusion_config.yaml --debug

# Train (Stage 1 → Stage 2; Stage 2 auto-inits from best_stage1.pt)
uv run python UNI-Stain-EarlyFusion/train_uni_stain_earlyfusion.py \
    --config configs/uni_stain_earlyfusion_config.yaml

# Official test evaluation (loads best_stage2.pt, NEVER last.pt)
uv run python UNI-Stain-EarlyFusion/evaluate_uni_stain_earlyfusion.py \
    --config configs/uni_stain_earlyfusion_config.yaml
```

## SLURM (Toubkal / UM6P, 1 A100)

```bash
sbatch slurm/slurm_uni_stain_earlyfusion/precompute_stain_stats.slurm
sbatch slurm/slurm_uni_stain_earlyfusion/train_uni_stain_earlyfusion.slurm
sbatch slurm/slurm_uni_stain_earlyfusion/evaluate_uni_stain_earlyfusion.slurm
```

## Reference

Best single-model test accuracy to beat: **94.69%**.