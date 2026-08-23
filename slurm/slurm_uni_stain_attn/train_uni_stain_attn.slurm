#!/bin/bash
# ============================================================
# SLURM Job Script — Train UNI-Stain-Attention
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_stain_attn_001
# ============================================================
# Trains the frozen UNI feature extractor + H/DAB stain branch
# fused via stain-conditioned attention pooling over UNI patch
# tokens + MLP head. UNI frozen forever.
#
# Usage:
#   sbatch slurm/slurm_uni_stain_attn/train_uni_stain_attn.slurm
#
# Prerequisite: run precompute_stain_stats.slurm first.
# ============================================================

#SBATCH --job-name=uni_stain_attn_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-Attention/uni_stain_attn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-Attention/uni_stain_attn_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_stain_attn_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_stain_attn_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_stain_attn_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Stain-Attention/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Single GPU (default):
uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python UNI-Stain-Attention/train_uni_stain_attn.py --config "${CONFIG}" --resume