# DSS-ViT on Toubkal (UM6P HPC)

Complete guide for running **DSS-ViT** on the Toubkal cluster using SLURM.
Covers local development → GitHub → HPC synchronization and the full
precompute → debug → train → evaluate workflow.

---

## Model overview

| Component | Detail |
|-----------|--------|
| RGB branch | Pretrained ViT-B16 (ImageNet) |
| Stain branch | ColorDeconvolution (H/DAB) → normalize → StainEncoder → 16 tokens |
| Fusion | Cross-attention (CLS query × stain tokens) + gated residual |
| Head | Ordinal cutpoints → class probabilities |
| Loss | CE + 0.1 × ordinal BCE |

Three-stage training (5 + 10 + 40 = 55 epochs):

| Stage | ViT | Trainable modules | ViT LR | New LR | Epochs |
|-------|-----|-------------------|--------|--------|--------|
| 1 | frozen | StainEncoder, CrossFusion, OrdinalHead | — | 2e-4 | 5 |
| 2 | frozen | StainEncoder, CrossFusion, OrdinalHead | — | 1e-4 | 10 |
| 3 | unfrozen | all | 1e-5 | 1e-4 | 40 |

---

## HPC workspace layout

```
/home/amine.aitlaamim-ext/projects/DSCA-ViT/       ← workspace root (NOT in Git)
│
├── code/
│   └── DSCA-ViT/                                  ← Git repository  ← git pull here
│       ├── configs/dss_vit_config.yaml
│       ├── datasets/
│       ├── models_v2_1/
│       ├── utils/
│       ├── scripts/
│       ├── slurm/
│       ├── doc/
│       ├── requirements.txt
│       └── .venv/                                 ← uv-managed env (git-ignored)
│
├── data/
│   └── HER2/                                      ← dataset — already on Toubkal, NOT in Git
│       ├── train/{class_0, class_1+, class_2+, class_3+}
│       └── test/ {class_0, class_1+, class_2+, class_3+}
│
├── experiments/
│   └── DSS-ViT/
│       └── dss_vit_001/
│           ├── experiment_meta.json               ← identity + config snapshot
│           ├── split_indices.npz                  ← deterministic train/val split
│           └── stain_stats.json                   ← H/DAB mean + std
│
├── checkpoints/
│   └── DSS-ViT/
│       └── dss_vit_001/
│           ├── stage1_end.pt
│           ├── stage2_end.pt
│           ├── best_stage3.pt                     ← used by evaluation
│           └── last.pt                            ← used by --resume
│
├── logs/
│   └── DSS-ViT/
│       └── dss_vit_001/
│           ├── train.log
│           ├── metrics.jsonl
│           ├── slurm_JOBID.out
│           └── slurm_JOBID.err
│
└── results/
    └── DSS-ViT/
        └── dss_vit_001/
            ├── test_results.json
            └── test_report.txt
```

### What is NOT in Git

`.gitignore` enforces this — none of the following can be accidentally committed:

| Pattern | Reason |
|---------|--------|
| `.venv/` | uv virtual environment — cluster-local |
| `data/` | dataset |
| `experiments/` | generated metadata |
| `checkpoints/` | model weights |
| `logs/` | training/SLURM logs |
| `results/` | evaluation outputs |
| `*.pt`, `*.pth` | checkpoints |
| `*.npz` | split index files |
| `*.jsonl` | metrics files |
| `*.log`, `*.out`, `*.err` | log files |
| `stain_stats.json` | generated stain statistics |

---

## Git synchronization workflow

```
YOUR LOCAL PC
  │  edit code / configs
  │  git add . && git commit -m "..."
  │  git push
  ▼
GITHUB
  │  git pull
  ▼
/home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT/
  │
  ├── reads   .../data/HER2/          (existing — not from Git)
  ├── writes  .../experiments/DSS-ViT/
  ├── writes  .../checkpoints/DSS-ViT/
  ├── writes  .../logs/DSS-ViT/
  └── writes  .../results/DSS-ViT/
```

---

## Confirmed Toubkal information

| Item | Value |
|------|-------|
| Username | `amine.aitlaamim-ext` |
| Workspace root | `/home/amine.aitlaamim-ext/projects/DSCA-ViT` |
| Repository path | `/home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT` |
| GPU partition | `gpu` |
| CPU partition | `compute` |
| Other partitions | `himem`, `gpu_h100` |
| GPU access | working |
| Dataset location | `/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2` |
| CUDA version (A100) | `12.1.1` (confirmed via `module avail CUDA`) |
| cuDNN (A100) | `8.9.2.26-CUDA-12.1.1` |
| Python environment | `uv`-managed `.venv` inside the repository |
| Environment activation | `source .venv/bin/activate` |

---

## Environment setup (already done)

The `.venv` was created with `uv` inside the repository using `requirements.txt`:

```bash
# Already done — do not repeat unless recreating from scratch
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Dependencies installed (`requirements.txt`):
- `torch >= 2.0`, `torchvision >= 0.15`, `timm >= 0.9`
- `numpy`, `scipy`, `scikit-learn`
- `PyYAML`, `Pillow`, `matplotlib`, `seaborn`

The `.venv/` directory is git-ignored and stays on the cluster. Every SLURM
script activates it with:

```bash
source "${REPO_DIR}/.venv/bin/activate"
```

If you need to recreate the environment (e.g. after a full workspace reset):

```bash
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Experiment naming convention

```
dss_vit_001   ← first experiment (current)
dss_vit_002   ← next experiment
```

To start a new experiment:
1. Change `experiment_name` in `configs/dss_vit_config.yaml` and update all
   derived paths in the same `paths:` block.
2. Update `EXPERIMENT=` in all three `slurm/*.slurm` scripts.
3. Update the `#SBATCH --output` / `--error` literal paths in those scripts
   (SLURM evaluates those before shell variables expand).
4. Commit and push from local PC. Pull on Toubkal.

---

## First-run verification on Toubkal

Before submitting jobs, confirm the environment is correct:

### A · Pull and activate

```bash
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
git pull
source .venv/bin/activate
```

### B · Verify GPU visibility

```bash
nvidia-smi
```

Expected: A100 GPU listed with memory information.

### C · Verify SLURM GPU allocation

```bash
srun --partition=gpu --gres=gpu:1 --time=00:05:00 nvidia-smi
```

Expected: same A100 GPU information, confirming SLURM can allocate a GPU.

### D · Verify PyTorch

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

Expected: `CUDA available: True` and `NVIDIA A100` (or similar).

### E · Verify dataset structure

```bash
find /home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2 -maxdepth 2 -type d | sort
```

Expected output:
```
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test/class_0
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test/class_1+
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test/class_2+
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/test/class_3+
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train/class_0
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train/class_1+
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train/class_2+
/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/train/class_3+
```

### F · Verify DSS-ViT imports

```bash
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
python -c "
from models_v2_1 import DSSViT
from utils.metrics_dss_vit import compute_metrics
from utils.split_utils import get_or_create_split_indices
print('DSS-ViT imports OK')
"
```

---

## Normal pipeline

### Step 1 — Precompute stain statistics (CPU job, run once per experiment)

```bash
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
sbatch slurm/precompute_stain_stats.slurm
```

Monitor:
```bash
squeue -u amine.aitlaamim-ext
sacct -j JOBID
```

Creates in `experiments/DSS-ViT/dss_vit_001/`:
- `split_indices.npz` — deterministic stratified 90/10 train/val split (seed 42)
- `stain_stats.json` — global H/DAB mean + std over the training split

If the split file already exists it is reloaded, not regenerated.

### Step 2 — Debug check (interactive, before full training)

```bash
source .venv/bin/activate
cd /home/amine.aitlaamim-ext/projects/DSCA-ViT/code/DSCA-ViT
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --debug
```

`--debug` runs only 3 batches per epoch per stage. Use this to confirm the
config, dataset loading, and model construction work before submitting the
full SLURM job.

### Step 3 — Full training (GPU job, 55 epochs)

```bash
sbatch slurm/train_dss_vit.slurm
```

Monitor:
```bash
squeue -u amine.aitlaamim-ext
tail -f /home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_JOBID.out
```

Training creates:

| Path | Stage | Contents |
|------|-------|----------|
| `checkpoints/.../stage1_end.pt` | end of stage 1 | model + optimizer + config |
| `checkpoints/.../stage2_end.pt` | end of stage 2 | model + optimizer + config |
| `checkpoints/.../best_stage3.pt` | stage 3 best val acc | model + optimizer + config |
| `checkpoints/.../last.pt` | every epoch | latest state (used by `--resume`) |
| `logs/.../train.log` | all stages | timestamped training log |
| `logs/.../metrics.jsonl` | all stages | per-epoch metrics |
| `experiments/.../experiment_meta.json` | training start | experiment identity snapshot |

### Step 4 — Evaluate (GPU job)

```bash
sbatch slurm/evaluate_dss_vit.slurm
```

Creates:

| Path | Contents |
|------|----------|
| `results/.../test_results.json` | accuracy, QWK, F1, per-class metrics |
| `results/.../test_report.txt` | human-readable report |
| `logs/.../eval.log` | evaluation log |

---

## Resume training

```bash
# Uncomment --resume in slurm/train_dss_vit.slurm, then:
sbatch slurm/train_dss_vit.slurm

# Or interactively:
python utils/train_dss_vit.py --config configs/dss_vit_config.yaml --resume
```

`--resume` loads `last.pt` and restores stage, epoch, optimizer, and scheduler.

---

## SLURM reference

```bash
# Check your running/pending jobs
squeue -u amine.aitlaamim-ext

# Check a completed job
sacct -j JOBID

# Tail a running job's output
tail -f /home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_JOBID.out

# Cancel a job
scancel JOBID
```

---

## Checkpoint reference

Each `.pt` file is a `torch.save` dict:

```python
{
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch":                int,
    "stage":                int,   # 1, 2, or 3
    "metrics":              dict,
    "config":               dict,
    "split_indices_path":   str,
}
```

---

## Multi-GPU DDP

Edit `slurm/train_dss_vit.slurm`:

```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
```

Replace the `python` training line with:

```bash
srun python utils/train_dss_vit.py --config "${CONFIG}"
```

Only rank 0 writes checkpoints, logs, and metrics.

---

## Source code layout (reference)

```
models_v2_1/          ← DSS-ViT model package
utils/
  ├── train_dss_vit.py
  ├── evaluate_dss_vit.py
  ├── metrics_dss_vit.py
  └── split_utils.py
scripts/
  └── precompute_stain_stats.py
configs/
  └── dss_vit_config.yaml
slurm/
  ├── train_dss_vit.slurm
  ├── evaluate_dss_vit.slurm
  └── precompute_stain_stats.slurm
requirements.txt
.venv/                ← uv venv (git-ignored, cluster-local)
```