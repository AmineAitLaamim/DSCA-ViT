#!/bin/bash
# ============================================================
# SLURM Job Script — Train UNI-Regularized-v2
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_regularized_v2_001
# ============================================================
# Follow-up to UNI-Regularized (v1) with THREE regularization
# changes bundled together:
#   1. Stochastic depth  : UNI backbone drop_path_rate=0.2
#   2. Head dropout      : nn.Dropout(0.3) before the Linear head
#   3. Rebalanced wd     : Stage 2 backbone wd=0.5, head wd=0.05
#                          -> lr*wd parity = 5.00e-06 both sides
#
# Usage:
#   sbatch slurm/slurm_uni_regularized_v2/train_uni_regularized_v2.slurm
# ============================================================

#SBATCH --job-name=uni_regularized_v2_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Regularized-v2/uni_regularized_v2_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Regularized-v2/uni_regularized_v2_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_regularized_v2_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_regularized_v2_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_regularized_v2_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Regularized-v2/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python UNI-Regularized-v2/train_uni_regularized_v2.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python UNI-Regularized-v2/train_uni_regularized_v2.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python UNI-Regularized-v2/train_uni_regularized_v2.py --config "${CONFIG}" --resume