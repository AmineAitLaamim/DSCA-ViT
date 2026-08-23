#!/bin/bash
# ============================================================
# SLURM Job Script - Train ViT-B16-CORN
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_b16_corn_001
# ============================================================
# Trains the ViT-B16 + CORN ordinal regression baseline using the
# same 2-stage protocol as the proper ViT-B16 baseline:
#   Stage 1: frozen backbone, train CORN head only (30 epochs)
#   Stage 2: full fine-tuning, backbone lr=1e-5, head lr=1e-4 (30 epochs)
#
# Usage:
#   sbatch slurm/slurm_vit_b16_corn/train_vit_b16_corn.slurm
# ============================================================

#SBATCH --job-name=vit_b16_corn_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-B16-CORN/vit_b16_corn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-B16-CORN/vit_b16_corn_001/slurm_%j.err

# ============================================================
# Workspace - must match configs/vit_b16_corn_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="vit_b16_corn_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/vit_b16_corn_config.yaml"
LOG_DIR="${WORKSPACE}/logs/ViT-B16-CORN/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment - uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python ViT-B16-CORN/train_vit_b16_corn.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python ViT-B16-CORN/train_vit_b16_corn.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python ViT-B16-CORN/train_vit_b16_corn.py --config "${CONFIG}" --resume