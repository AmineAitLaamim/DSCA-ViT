# UNI-Stain-Attention — Frozen UNI + Stain-Conditioned Attention Pooling

A model that uses a **frozen UNI histopathology foundation model**
(ViT-L/16, DINOv2, mass100k) as a feature extractor, adds **H/DAB stain
features**, and uses **stain-conditioned attention pooling** over UNI
patch tokens to fuse the modalities more richly than simple
concatenation.

This experiment tests whether a more advanced fusion mechanism improves
over the previous `UNI-Stain-MLP` model while keeping UNI frozen.

## Reference numbers

| Model | Test accuracy |
|-------|---------------|
| Proper ViT-B16 baseline | 94.69% |
| UNI Stage 2 (fine-tuned) | 94.69% (* reference only) |
| DSS-ViT v2.2 (ViT) | 93.18% |
| UNI-Stain-MLP (previous) | pending |
| **UNI-Stain-Attention (this)** | **TBD** |

(* The UNI Stage 2 number was selected on an overfit validation process;
treat as a reference, not a trusted final baseline.)

---

## 1. How this differs from previous attempts

| Model | Fusion | Stain info | Training |
|-------|--------|-----------|----------|
| UNI baseline (`baseline/UNI-baseline/`) | simple | none | 2-stage, fine-tunes UNI |
| UNI-RGB-MLP (`UNI-RGB-MLP/`) | simple concat | none | 1-stage, train head only |
| UNI-Stain-MLP (`UNI_v2/`) | simple concat | H/DAB → 512-dim vector | 1-stage, frozen UNI |
| **UNI-Stain-Attention (this)** | **stain-conditioned attention pooling** | **H/DAB → 512-dim vector + [B,256] attention summary** | **1-stage, frozen UNI** |

The key difference from `UNI_v2/` is **stain-conditioned attention
pooling** over UNI patch tokens: the stain feature queries spatially
meaningful UNI patch tokens via `nn.MultiheadAttention` (embed_dim=256,
4 heads), producing a stain-conditioned summary of tissue architecture
`[B, 256]`, instead of simple concatenation of `rgb_feat` + `stain_feat`.

## 2. Model architecture

```
RGB [B,3,224,224] (raw [0,1])
├── ImageNet normalize inside model
├── with torch.no_grad():
│     UNI forward_features → tokens [B,197,1024]
│     cls = tokens[:,0]        [B,1024]
│     patch_tokens = tokens[:,1:] [B,196,1024]
│     gap = patch_tokens.mean  [B,1024]
│     rgb_feat = cat([cls,gap]) [B,2048]
├── ColorDeconvolution → H, DAB [B,1,H,W]
│   → normalize with train-split stats
│   → stain_input = cat([h_norm, dab_norm]) [B,2,H,W]
│   → StainEncoder [B,512] (stain_feat)
├── Stain-conditioned attention pooling:
│     stain_query = stain_query_proj(stain_feat)   [B,256]
│     patch_keys  = patch_key_proj(patch_tokens)   [B,196,256]
│     patch_vals  = patch_value_proj(patch_tokens) [B,196,256]
│     attn_out, _ = MultiheadAttention(query=[B,1,256], key, value) → [B,1,256]
│     stain_attended = attn_out.squeeze(1)         [B,256]
└── combined = cat([rgb_feat, stain_feat, stain_attended]) [B,2816]
    → head: Linear(2816,1024) → GELU → Dropout(0.3) → Linear(1024,4)
    → {"logits": [B,4], "probs": [B,4]}
```

The UNI backbone is frozen forever. A `train()` override keeps the
backbone in `eval()` mode even when the outer model is in train mode.
The backbone forward pass is wrapped in an explicit `torch.no_grad()`
block.

## 3. Package layout

| File | Purpose |
|------|---------|
| `color_deconv.py` | Self-contained Ruifrok ColorDeconvolution |
| `stain_stats.py` | Load/save global H/DAB stats |
| `stain_encoder.py` | Small CNN [B,2,H,W] → [B,512] |
| `uni_stain_attn.py` | Main model (frozen UNI + stain + attention pooling + MLP head) |
| `precompute_stain_stats.py` | Recompute H/DAB stats for this package |
| `train_uni_stain_attn.py` | 1-stage training CLI |
| `evaluate_uni_stain_attn.py` | Official test-set evaluation CLI |
| `README.md` | This file |

The folder name contains hyphens, so it is **not** importable as a
Python package. Scripts are run **by path** with direct sibling imports.

## 4. Training recipe (single stage, UNI frozen)

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Optimizer | AdamW (trainable params only) |
| LR | 1e-3 |
| Weight decay | 0.05 (weights only) |
| Loss | CrossEntropyLoss(label_smoothing=0.1) |
| Batch size | 64 (per GPU) |
| AMP | enabled |
| Grad clip | 1.0 |
| Scheduler | CosineAnnealingLR(T_max=30) |
| Validation | every epoch, best by val accuracy |

Trainable modules: `stain_encoder`, `stain_query_proj`, `patch_key_proj`,
`patch_value_proj`, `attention` (MultiheadAttention), `head`.
The optimizer never includes `backbone.parameters()`.

## 5. Commands

```bash
# 1) Precompute stain stats (train split only, self-consistent)
uv run python UNI-Stain-Attention/precompute_stain_stats.py --config configs/uni_stain_attn_config.yaml

# 2) Train (single GPU)
uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config configs/uni_stain_attn_config.yaml

# 2b) Debug / sanity check
uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config configs/uni_stain_attn_config.yaml --debug

# 2c) Resume
uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config configs/uni_stain_attn_config.yaml --resume

# 3) Evaluate (loads best.pt, never last.pt)
uv run python UNI-Stain-Attention/evaluate_uni_stain_attn.py --config configs/uni_stain_attn_config.yaml
```

On HPC:
```bash
sbatch slurm/slurm_uni_stain_attn/precompute_stain_stats.slurm
sbatch slurm/slurm_uni_stain_attn/train_uni_stain_attn.slurm
sbatch slurm/slurm_uni_stain_attn/evaluate_uni_stain_attn.slurm
```

## 6. Precompute stain stats

- Loads `split_indices_wsi.npz`, uses **only** `train_indices`.
- Uses the SAME `ColorDeconvolution` from `UNI-Stain-Attention/color_deconv.py`.
- Saves to `configs/uni_stain_attn_stain_stats.json`.
- Training raises if this file is missing.

## 7. `--debug` sanity checks

- Sets `HF_HUB_OFFLINE=1` at the very top before any imports.
- Static-scans `UNI-Stain-Attention/` for:
  `hf_hub_download`, `huggingface_hub.login`, `HF_TOKEN`, `from_pretrained`.
- Asserts UNI backbone `requires_grad=False` for all params.
- Asserts trainable params > 0.
- Random raw-RGB [B,3,224,224] in [0,1]; checks logits/probs [B,4].
- Checks no NaN/Inf in logits/probs; `loss.backward()` succeeds.
- After `model.train()`, asserts `model.backbone.training == False`.
- Asserts attention output shape `[B,256]` before fusion.
- Confirms backbone forward runs inside `torch.no_grad()` (code inspection).
- Prints per-group counts (frozen backbone / stain_encoder / projections /
  attention / head / total trainable).

## 8. Outputs

| Path | Contents |
|------|----------|
| `checkpoints/UNI-Stain-Attention/uni_stain_attn_001/best.pt` | Best val accuracy |
| `checkpoints/UNI-Stain-Attention/uni_stain_attn_001/last.pt` | Latest state (`--resume`) |
| `logs/UNI-Stain-Attention/uni_stain_attn_001/train.log` | Training log |
| `logs/UNI-Stain-Attention/uni_stain_attn_001/metrics.jsonl` | Per-epoch metrics |
| `results/UNI-Stain-Attention/uni_stain_attn_001/test_results.json` | Test metrics (JSON) |
| `results/UNI-Stain-Attention/uni_stain_attn_001/test_report.txt` | Human-readable report |

## 9. Notes

- Split: load-only `split_indices_wsi.npz` (7,283 / 810 / 1,847;
  `val_fraction=0.10`, `seed=42`); never regenerated.
- Stain stats: `configs/uni_stain_attn_stain_stats.json`, self-consistent
  with this package's `color_deconv.py`.
- Fully offline: local UNI checkpoint only, no Hugging Face.
- Use `uv run python` on HPC (no conda).
- No existing packages modified (`baseline/`, `models_*`, `UNI_v2/`,
  `UNI-RGB-MLP/`, etc.).