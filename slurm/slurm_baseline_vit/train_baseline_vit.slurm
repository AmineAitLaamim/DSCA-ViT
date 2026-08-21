#!/bin/bash
# ============================================================
# SLURM Job Script — Train Plain ViT-B16 Baseline
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: plain_vit_baseline_001
# ============================================================
# Reproduces the 2-stage baseline protocol (same logic as the
# Colab notebook, but on HPC):
#   Stage 1: frozen backbone, train classification head only (30 epochs)
#   Stage 2: full fine-tuning, backbone lr=1e-5, head lr=1e-4 (30 epochs)
#
# Usage:
#   sbatch slurm/slurm_baseline_vit/train_baseline_vit.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must be literal absolute
# strings matching log_dir in configs/plain_vit_baseline_config.yaml.
# ============================================================

#SBATCH --job-name=baseline_vit_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-Baseline/plain_vit_baseline_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-Baseline/plain_vit_baseline_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/plain_vit_baseline_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="plain_vit_baseline_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/plain_vit_baseline_config.yaml"
LOG_DIR="${WORKSPACE}/logs/ViT-Baseline/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv inside the repository
# uv run picks up .venv automatically; no activation needed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python baseline/train_baseline_vit.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python baseline/train_baseline_vit.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python baseline/train_baseline_vit.py --config "${CONFIG}" --resume
