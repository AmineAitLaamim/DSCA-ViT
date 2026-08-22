# UNI-Stain-MLP — Frozen UNI + H/DAB Stain Side-Information

A model that uses a **frozen UNI histopathology foundation model**
(ViT-L/16, DINOv2, mass100k) as a feature extractor and adds
**H/DAB stain side-information** through a small trainable
`StainEncoder` branch. The final classifier is an MLP head.

The hypothesis: adding stain information to frozen UNI features can
match/beat the baselines while avoiding full fine-tuning overfitting.

## Reference numbers

| Model | Test accuracy |
|-------|---------------|
| Proper ViT-B16 baseline | 94.69% |
| UNI Stage 2 (fine-tuned) | 94.69% (* reference only) |
| DSS-ViT v2.2 (ViT) | 93.18% |
| UNI-Stain-MLP (this) | TBD |

(* The UNI Stage 2 number was selected on an overfit validation process;
treat it as a reference, not a trusted final baseline.)

---

## 1. How this differs from previous attempts

| Model | Backbone | Stain info | Training |
|-------|----------|-----------|----------|
| UNI baseline (`baseline/UNI-baseline/`) | Frozen UNI then Stage-2 fine-tune | none | 2-stage, fine-tunes UNI |
| DSS-ViT v2.2 | ViT-B/16 (trainable) | Stain tokens + cross-attention | multi-stage, complex |
| **UNI-Stain-MLP (this)** | **Frozen UNI forever** | **H/DAB → StainEncoder → 512-dim vector** | **1-stage, train StainEncoder + MLP head only** |

This version avoids full UNI fine-tuning entirely:
- UNI features are deterministic (backbone forced to eval mode via a
  `train()` override).
- Only a small `StainEncoder` (~0.5M params) + MLP head are trained.
- Stain features are self-consistent: H/DAB stats are recomputed by
  `UNI_v2/precompute_stain_stats.py` (not reusing the old DSS-ViT
  v2.2 `stain_stats.json`).

## 2. Model architecture

```
RGB [B,3,224,224] (raw [0,1])
├── ImageNet normalize inside model
├── UNI forward_features → tokens [B,197,1024]
│     cls = tokens[:,0]        [B,1024]
│     gap = tokens[:,1:].mean  [B,1024]
│     rgb_feat = cat([cls,gap]) [B,2048]
├── ColorDeconvolution → H, DAB [B,1,H,W] each
│   → normalize with train-split stats
│   → stain_input = cat([h_norm, dab_norm]) [B,2,H,W]
│   → StainEncoder [B,512]
│   → stain_feat
└── combined = cat([rgb_feat, stain_feat]) [B,2560]
    → head: Linear(2560,1024) → GELU → Dropout(0.3) → Linear(1024,4)
    → {"logits": [B,4], "probs": [B,4]}
```

The UNI backbone is frozen forever. A `train()` override keeps the
backbone in `eval()` mode even when the outer model is in train mode.

## 3. Package layout

| File | Purpose |
|------|---------|
| `color_deconv.py` | Self-contained Ruifrok ColorDeconvolution |
| `stain_stats.py` | Load/save global H/DAB stats |
| `stain_encoder.py` | Small CNN [B,2,H,W] → [B,512] |
| `uni_stain_mlp.py` | Main model (frozen UNI + stain branch + MLP head) |
| `precompute_stain_stats.py` | Recompute H/DAB stats for this package |
| `train_uni_stain_mlp.py` | 1-stage training CLI |
| `evaluate_uni_stain_mlp.py` | Official test-set evaluation CLI |
| `README.md` | This file |

## 4. Training recipe (single stage, UNI frozen)

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Optimizer | AdamW (stain_encoder + head only) |
| LR | 1e-3 |
| Weight decay | 0.05 (weights only) |
| Loss | CrossEntropyLoss(label_smoothing=0.1) |
| Batch size | 64 (per GPU) |
| AMP | enabled |
| Grad clip | 1.0 |
| Scheduler | CosineAnnealingLR(T_max=30) |
| Validation | every epoch, best by val accuracy |

The optimizer is constructed ONLY with `stain_encoder` and `head`
parameters — backbone.parameters() are never included.

## 5. Commands

```bash
# 1) Precompute stain stats (uses train_indices from split, saves to configs/uni_stain_mlp_stain_stats.json)
uv run python UNI_v2/precompute_stain_stats.py --config configs/uni_stain_mlp_config.yaml

# 2) Train (single GPU)
uv run python UNI_v2/train_uni_stain_mlp.py --config configs/uni_stain_mlp_config.yaml

# 2b) Debug / sanity check
uv run python UNI_v2/train_uni_stain_mlp.py --config configs/uni_stain_mlp_config.yaml --debug

# 2c) Resume
uv run python UNI_v2/train_uni_stain_mlp.py --config configs/uni_stain_mlp_config.yaml --resume

# 3) Evaluate (loads best.pt, never last.pt)
uv run python UNI_v2/evaluate_uni_stain_mlp.py --config configs/uni_stain_mlp_config.yaml
```

On HPC:
```bash
sbatch slurm/slurm_uni_stain_mlp/train_uni_stain_mlp.slurm
sbatch slurm/slurm_uni_stain_mlp/evaluate_uni_stain_mlp.slurm
```

## 6. `--debug` sanity checks

- Sets `HF_HUB_OFFLINE=1` at the very top before any imports.
- Static-scans `UNI_v2/` for:
  `hf_hub_download`, `huggingface_hub.login`, `HF_TOKEN`, `from_pretrained`.
- Asserts UNI backbone `requires_grad=False` for all params.
- Asserts trainable params > 0.
- Random raw-RGB [B,3,224,224] in [0,1]; checks logits/probs [B,4].
- Checks no NaN/Inf in logits/probs; `loss.backward()` succeeds.
- After `model.train()`, asserts `model.backbone.training == False`.
- Prints frozen backbone / stain encoder / head / total trainable counts.

## 7. Outputs

| Path | Contents |
|------|----------|
| `checkpoints/UNI-Stain-MLP/uni_stain_mlp_001/best.pt` | Best val accuracy |
| `checkpoints/UNI-Stain-MLP/uni_stain_mlp_001/last.pt` | Latest state (`--resume`) |
| `logs/UNI-Stain-MLP/uni_stain_mlp_001/train.log` | Training log |
| `logs/UNI-Stain-MLP/uni_stain_mlp_001/metrics.jsonl` | Per-epoch metrics |
| `results/UNI-Stain-MLP/uni_stain_mlp_001/test_results.json` | Test metrics (JSON) |
| `results/UNI-Stain-MLP/uni_stain_mlp_001/test_report.txt` | Human-readable report |

## 8. Notes

- Split: load-only `split_indices_wsi.npz` (7,283 / 810 / 1,847;
  `val_fraction=0.10`, `seed=42`); never regenerated.
- Stain stats: `configs/uni_stain_mlp_stain_stats.json`, self-consistent
  with this package's `color_deconv.py`.
- Fully offline: local UNI checkpoint only, no Hugging Face.
- Use `uv run python` on HPC (no conda).
- No existing packages modified (`baseline/`, `models_*`, `utils/`, etc.).