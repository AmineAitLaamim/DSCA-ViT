#!/bin/bash
# ============================================================
# SLURM Job Script — Train UNI-Regularized
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_regularized_001
# ============================================================
# Trains the UNI backbone with a regularized fine-tuning recipe
# (AdamW + early stopping on val loss) and corrected Stage 2
# initialization from best_stage1.pt.
#
# Usage:
#   sbatch slurm/slurm_uni_regularized/train_uni_regularized.slurm
# ============================================================

#SBATCH --job-name=uni_regularized_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Regularized/uni_regularized_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Regularized/uni_regularized_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_regularized_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_regularized_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_regularized_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Regularized/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python UNI-Regularized/train_uni_regularized.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python UNI-Regularized/train_uni_regularized.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python UNI-Regularized/train_uni_regularized.py --config "${CONFIG}" --resume