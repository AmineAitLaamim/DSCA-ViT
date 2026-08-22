#!/bin/bash
# ============================================================
# SLURM Job Script — Precompute Stain Stats (UNI-Stain-MLP)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_stain_mlp_001
# ============================================================
# Computes global H/DAB mean/std over the TRAINING split only,
# using the SAME ColorDeconvolution implementation in UNI_v2/.
#
# Usage:
#   sbatch slurm/slurm_uni_stain_mlp/precompute_stain_stats.slurm
# ============================================================

#SBATCH --job-name=uni_stain_mlp_precompute
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-MLP/uni_stain_mlp_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-MLP/uni_stain_mlp_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_stain_mlp_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_stain_mlp_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_stain_mlp_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Stain-MLP/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run precompute
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python UNI_v2/precompute_stain_stats.py --config "${CONFIG}"