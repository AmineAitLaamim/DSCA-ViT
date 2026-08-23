# UNI-RGB-MLP — Frozen UNI + RGB-Only MLP Ablation

A **control ablation** that uses a **frozen UNI histopathology
foundation model** (ViT-L/16, DINOv2, mass100k) as a feature extractor
and an **MLP head on RGB features only** — with **no stain branch**.

This isolates whether the stain side-information in `UNI-Stain-MLP`
(`UNI_v2/`) actually helps.

## Reference numbers

| Model | Test accuracy |
|-------|---------------|
| Proper ViT-B16 baseline | 94.69% |
| UNI Stage 2 (fine-tuned) | 94.69% |
| DSS-ViT v2.2 (ViT) | 93.18% |
| UNI-Stain-MLP (previous) | 92.91% |
| **UNI-RGB-MLP (this)** | **TBD** |

## Interpretation

- If **RGB-only < UNI-Stain-MLP** → the stain branch helps.
- If **RGB-only ≈ UNI-Stain-MLP** → the stain branch adds little; frozen
  UNI itself is the limiting factor.
- If **RGB-only > UNI-Stain-MLP** → the stain branch may be hurting.

---

## 1. Model architecture

```
RGB [B,3,224,224] (raw [0,1])
├── ImageNet normalize inside model
├── with torch.no_grad():
│     UNI forward_features → tokens [B,197,1024]
│     cls = tokens[:,0]        [B,1024]
│     gap = tokens[:,1:].mean  [B,1024]
│     rgb_feat = cat([cls,gap]) [B,2048]
└── head: Linear(2048,1024) → GELU → Dropout(0.3) → Linear(1024,4)
    → {"logits": [B,4], "probs": [B,4]}
```

The UNI backbone is frozen forever. A `train()` override keeps the
backbone in `eval()` mode even when the outer model is in train mode.
The backbone forward pass is wrapped in an explicit `torch.no_grad()`
block.

## 2. Package layout

| File | Purpose |
|------|---------|
| `uni_rgb_mlp.py` | Main model (frozen UNI + RGB-only MLP head) |
| `train_uni_rgb_mlp.py` | 1-stage training CLI |
| `evaluate_uni_rgb_mlp.py` | Official test-set evaluation CLI |
| `README.md` | This file |

The folder name contains hyphens, so it is **not** importable as a
Python package. Scripts are run **by path** with direct sibling imports:

```bash
uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config configs/uni_rgb_mlp_config.yaml
```

## 3. Training recipe (single stage, UNI frozen)

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Optimizer | AdamW (head params only) |
| LR | 1e-3 |
| Weight decay | 0.05 (weights only) |
| Loss | CrossEntropyLoss(label_smoothing=0.1) |
| Batch size | 64 (per GPU) |
| AMP | enabled |
| Grad clip | 1.0 |
| Scheduler | CosineAnnealingLR(T_max=30) |
| Validation | every epoch, best by val accuracy |

The optimizer is constructed ONLY with `head` parameters —
backbone.parameters() are never included.

## 4. Commands

```bash
# Train (single GPU)
uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config configs/uni_rgb_mlp_config.yaml

# Debug / sanity check
uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config configs/uni_rgb_mlp_config.yaml --debug

# Resume
uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config configs/uni_rgb_mlp_config.yaml --resume

# Evaluate (loads best.pt, never last.pt)
uv run python UNI-RGB-MLP/evaluate_uni_rgb_mlp.py --config configs/uni_rgb_mlp_config.yaml
```

On HPC:
```bash
sbatch slurm/slurm_uni_rgb_mlp/train_uni_rgb_mlp.slurm
sbatch slurm/slurm_uni_rgb_mlp/evaluate_uni_rgb_mlp.slurm
```

## 5. `--debug` sanity checks

- Sets `HF_HUB_OFFLINE=1` at the very top before any imports.
- Static-scans `UNI-RGB-MLP/` for:
  `hf_hub_download`, `huggingface_hub.login`, `HF_TOKEN`, `from_pretrained`.
- Asserts UNI backbone `requires_grad=False` for all params.
- Asserts trainable params > 0.
- Random raw-RGB [B,3,224,224] in [0,1]; checks logits/probs [B,4].
- Checks no NaN/Inf in logits/probs; `loss.backward()` succeeds.
- After `model.train()`, asserts `model.backbone.training == False`.
- Confirms backbone forward runs inside `torch.no_grad()` (code inspection).
- Prints frozen backbone / head / total trainable counts.

## 6. Outputs

| Path | Contents |
|------|----------|
| `checkpoints/UNI-RGB-MLP/uni_rgb_mlp_001/best.pt` | Best val accuracy |
| `checkpoints/UNI-RGB-MLP/uni_rgb_mlp_001/last.pt` | Latest state (`--resume`) |
| `logs/UNI-RGB-MLP/uni_rgb_mlp_001/train.log` | Training log |
| `logs/UNI-RGB-MLP/uni_rgb_mlp_001/metrics.jsonl` | Per-epoch metrics |
| `results/UNI-RGB-MLP/uni_rgb_mlp_001/test_results.json` | Test metrics (JSON) |
| `results/UNI-RGB-MLP/uni_rgb_mlp_001/test_report.txt` | Human-readable report |

## 7. Notes

- Split: load-only `split_indices_wsi.npz` (7,283 / 810 / 1,847;
  `val_fraction=0.10`, `seed=42`); never regenerated.
- Fully offline: local UNI checkpoint only, no Hugging Face.
- Use `uv run python` on HPC (no conda).
- No existing packages modified (`baseline/`, `models_*`, `UNI_v2/`, etc.).