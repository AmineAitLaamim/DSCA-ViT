# DSS-ViT Project — Complete Status and Handoff Document

> **Purpose:** This document explains everything done to prepare the
> DSS-ViT project for HPC execution on Toubkal (UM6P). Read this first
> before touching any file. Written for a new agent or developer continuing
> this work.

---

## 1. Project Overview

### Repository

```
Local PC:   DSCA-ViT/
GitHub:     AmineAitLaamim/DSCA-ViT
HPC:        /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT/
```

### What the repository contains

The repo contains **multiple model generations**:

| Directory | Model | Status |
|-----------|-------|--------|
| `models/` | DSCA-ViT v1 | **DO NOT TOUCH** |
| `models_v2/` | DSCA-ViT v2 | **DO NOT TOUCH** |
| `models_v3/` | DSCA-ViT v3 | **DO NOT TOUCH** |
| `models_v2_1/` | **DSS-ViT** | Active development target |

### ⚠️ Critical scope rule

**This project is currently focused exclusively on DSS-ViT (`models_v2_1/`).**

Do NOT modify, refactor, rename, or delete anything in:
- `models/`, `models_v2/`, `models_v3/`
- `utils/train.py`, `utils/train_v2.py`, `utils/train_v3.py`
- `utils/metrics.py`, `utils/metrics_v2.py`

Only the following files are in scope for DSS-ViT infrastructure work:
- `configs/dss_vit_config.yaml`
- `utils/train_dss_vit.py`
- `utils/evaluate_dss_vit.py`
- `scripts/precompute_stain_stats.py`
- `slurm/train_dss_vit.slurm`
- `slurm/evaluate_dss_vit.slurm`
- `slurm/precompute_stain_stats.slurm`
- `.gitignore`
- `README_HPC.md`

---

## 2. DSS-ViT — What It Is

**Dual-Stream Stain Vision Transformer** for HER2 IHC breast cancer grading.

### Architecture (do not change)

| Component | Detail |
|-----------|--------|
| RGB branch | Pretrained ViT-B16 (ImageNet) via `timm` |
| Stain branch | ColorDeconvolution (H/DAB) → normalize → StainEncoder → 16 tokens |
| Fusion | Cross-attention (CLS query × stain tokens) + gated residual |
| Head | Ordinal cutpoints → 4-class probabilities |
| Loss | CrossEntropy + 0.1 × ordinal BCE |

### Training stages (do not change)

| Stage | ViT | LR (new modules) | ViT LR | Epochs |
|-------|-----|------------------|--------|--------|
| 1 | frozen | 2e-4 | — | 5 |
| 2 | frozen | 1e-4 | — | 10 |
| 3 | unfrozen | 1e-4 | 1e-5 | 40 |

### Dataset — HER2-IHC-40x

4-class ordinal grading: `class_0`, `class_1+`, `class_2+`, `class_3+`.
Downloaded from Zenodo as `her2-ihc-40x-wsi.zip`, extracted and reorganized.
Already present on Toubkal — **never download again, never put in Git**.

---

## 3. Key Files to Read (in this order)

| File | Purpose |
|------|---------|
| `configs/dss_vit_config.yaml` | All hyperparameters + all HPC paths. The single source of truth. |
| `README_HPC.md` | Complete HPC workflow guide. Every command you need. |
| `models_v2_1/dss_vit.py` | DSSViT model assembly |
| `utils/train_dss_vit.py` | CLI training: DDP-ready, AMP, staged, `--debug`, `--resume` |
| `utils/evaluate_dss_vit.py` | CLI test evaluation |
| `utils/split_utils.py` | Deterministic stratified 90/10 train/val split (seed 42) |
| `scripts/precompute_stain_stats.py` | Global H/DAB mean+std over training split |

### SLURM scripts

| File | Partition | GPU | Time |
|------|-----------|-----|------|
| `slurm/precompute_stain_stats.slurm` | `compute` | none | 2h |
| `slurm/train_dss_vit.slurm` | `gpu` | 1 A100 | 24h |
| `slurm/evaluate_dss_vit.slurm` | `gpu` | 1 A100 | 2h |

---

## 4. What Was Done — Summary of All Changes

### Problem

Project was originally developed in Google Colab. Paths were hardcoded for
Colab, config keys were scattered (`stain_stats.path`, `dataset.split_indices_path`),
and there were no SLURM scripts or HPC documentation.

### Goal

Prepare DSS-ViT execution infrastructure for Toubkal HPC without touching
the model architecture, training logic, or v1/v2/v3 code.

---

### `configs/dss_vit_config.yaml`

Removed two stale config sections (`stain_stats.path`, `dataset.split_indices_path`).
All DSS-ViT paths unified under a single `paths:` block with 9 keys:

```yaml
paths:
  experiment_name:   "dss_vit_001"
  train_dir:         "/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train"
  test_dir:          "/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test"
  experiment_dir:    "/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/DSS-ViT/dss_vit_001"
  checkpoint_dir:    "/home/amine.aitlaamim-ext/projects/DSCA-ViT/checkpoints/DSS-ViT/dss_vit_001"
  log_dir:           "/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001"
  results_dir:       "/home/amine.aitlaamim-ext/projects/DSCA-ViT/results/DSS-ViT/dss_vit_001"
  stain_stats_path:  "/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/DSS-ViT/dss_vit_001/stain_stats.json"
  split_indices_path:"/home/amine.aitlaamim-ext/projects/DSCA-ViT/experiments/DSS-ViT/dss_vit_001/split_indices.npz"
```

> All paths are literal absolute strings — YAML performs no interpolation.
> When changing workspace or experiment, update all 8 paths together in this block.

---

### `utils/train_dss_vit.py` (lines ~544–576)

Fixed two stale config key lookups:
```python
# Before (broken):
split_indices_path = config["dataset"]["split_indices_path"]
stain_stats_path   = config["stain_stats"]["path"]

# After (correct):
split_indices_path = config["paths"]["split_indices_path"]
stain_stats_path   = config["paths"]["stain_stats_path"]
```

Added (rank 0 only):
- `makedirs` for `experiment_dir` and `results_dir`
- Writes `experiment_meta.json` at training start with: `experiment_name`,
  `started_at`, `git_commit`, `seed`, full `config`

Training logic, DDP, AMP, `--debug`, `--resume`, staged training — **all unchanged**.

---

### `utils/evaluate_dss_vit.py` (lines ~81–86, 190, 196)

Fixed one stale key lookup. Added `results_dir`. Redirected:
- `test_results.json` → `results_dir` (not `log_dir`)
- `test_report.txt`  → `results_dir` (not `log_dir`)

`eval.log` stays in `log_dir` (correct).

---

### `scripts/precompute_stain_stats.py` (lines ~57–62)

Fixed two stale key lookups. Added `makedirs` for `experiment_dir` before
writing `stain_stats.json` and `split_indices.npz`.

---

### SLURM scripts (all three — full rewrites)

All three scripts now have:
- Real absolute `WORKSPACE` (no `<USER>` placeholder)
- Real `--partition` (no `<PARTITION>` placeholder)
- `REPO_DIR="${WORKSPACE}/code/DSCA-ViT"` derived cleanly
- Literal absolute `#SBATCH --output/--error` paths
- `uv run python` instead of `python` (Python not on PATH on Toubkal)

| Script | Partition | GPU | Key command |
|--------|-----------|-----|-------------|
| `precompute_stain_stats.slurm` | `compute` | `gpu:0` | `uv run python scripts/precompute_stain_stats.py` |
| `train_dss_vit.slurm` | `gpu` | `gpu:1` | `uv run python utils/train_dss_vit.py` |
| `evaluate_dss_vit.slurm` | `gpu` | `gpu:1` | `uv run python utils/evaluate_dss_vit.py` |

---

### `.gitignore`

Added:
```gitignore
*.jsonl          # metrics files
*.npz            # split index files
stain_stats.json
*.log  *.out  *.err
experiments/
results/
jobs/
```

`.venv/` was already present — no change needed.

---

### `README_HPC.md`

Full rewrite. See that file for the complete workflow documentation.

---

## 5. HPC Workspace Layout

```
/home/amine.aitlaamim-ext/projects/DSCA-ViT/
├── code/DSCA-ViT/          ← Git repository
│   └── .venv/              ← uv env (git-ignored)
├── data/HER2/              ← dataset (NOT in Git)
├── experiments/DSS-ViT/dss_vit_001/
│   ├── experiment_meta.json
│   ├── split_indices.npz
│   └── stain_stats.json
├── checkpoints/DSS-ViT/dss_vit_001/
│   ├── stage1_end.pt / stage2_end.pt
│   ├── best_stage3.pt
│   └── last.pt
├── logs/DSS-ViT/dss_vit_001/
│   ├── train.log / metrics.jsonl
│   └── slurm_JOBID.out / .err
└── results/DSS-ViT/dss_vit_001/
    ├── test_results.json
    └── test_report.txt
```

---

## 6. Confirmed Toubkal Information

| Item | Value |
|------|-------|
| Username | `amine.aitlaamim-ext` |
| Workspace root | `/home/amine.aitlaamim-ext/projects/DSCA-ViT` |
| GPU partition | `gpu` |
| CPU partition | `compute` |
| CUDA (A100) | `12.1.1` |
| cuDNN (A100) | `8.9.2.26-CUDA-12.1.1` |
| Python env | `uv`-managed `.venv` inside repo |
| Python invocation | `uv run python` |
| GPU access | Working |

---

## 7. What Has NOT Been Done Yet

| Task | Notes |
|------|-------|
| Actual training run | Pipeline is ready; training not yet submitted |
| GPU allocation verification | Run `srun --partition=gpu --gres=gpu:1 --time=00:05:00 nvidia-smi` |
| DSS-ViT import check on HPC | Run the import check from `README_HPC.md` Step F |
| `models_v2_1/` bug review | Architecture not reviewed during this session |
| Multi-GPU DDP test | Documented but not tested |

---

## 8. Quick Reference

```bash
# ── LOCAL PC ──────────────────────────────────────────────────────────────────
git add . && git commit -m "..." && git push

# ── TOUBKAL ───────────────────────────────────────────────────────────────────
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
git pull

# Verify imports
uv run python -c "from models_v2_1 import DSSViT; print('OK')"

# Step 1 — precompute (CPU, once per experiment)
sbatch slurm/precompute_stain_stats.slurm

# Step 2 — debug check (interactive)
uv run python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug

# Step 3 — full training (GPU, 55 epochs)
sbatch slurm/train_dss_vit.slurm

# Step 4 — evaluate
sbatch slurm/evaluate_dss_vit.slurm

# Monitor
squeue -u amine.aitlaamim-ext
sacct -j JOBID
tail -f /home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_JOBID.out
```

## 9. Starting a New Experiment (dss_vit_002, etc.)

1. Edit `configs/dss_vit_config.yaml` — change `experiment_name` and all 8 paths.
2. Edit all three `slurm/*.slurm` — change `EXPERIMENT=` and the two `#SBATCH --output/--error` literal paths.
3. Commit, push, pull on HPC. Run precompute → debug → train → evaluate.
