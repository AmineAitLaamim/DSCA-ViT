#!/bin/bash
# ============================================================
# SLURM Job Script — Train UNI-RGB-MLP
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_rgb_mlp_001
# ============================================================
# Trains the frozen UNI feature extractor + RGB-only MLP head.
# Single stage, UNI frozen forever.
#
# Usage:
#   sbatch slurm/slurm_uni_rgb_mlp/train_uni_rgb_mlp.slurm
# ============================================================

#SBATCH --job-name=uni_rgb_mlp_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-RGB-MLP/uni_rgb_mlp_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-RGB-MLP/uni_rgb_mlp_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_rgb_mlp_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_rgb_mlp_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_rgb_mlp_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-RGB-MLP/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python UNI-RGB-MLP/train_uni_rgb_mlp.py --config "${CONFIG}" --resume