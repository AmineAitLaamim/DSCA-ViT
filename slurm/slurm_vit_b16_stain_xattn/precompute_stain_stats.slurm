#!/bin/bash
# ============================================================
# SLURM Job Script — Precompute Stain Stats (ViT-B16-Stain-XAttn)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_b16_stain_xattn_001
# ============================================================
# Computes global H/DAB mean/std over the TRAINING split only,
# using the SAME ColorDeconvolution implementation in
# ViT-B16-Stain-XAttn/. Saves to
#   configs/vit_b16_stain_xattn_stain_stats.json
#
# Usage:
#   sbatch slurm/slurm_vit_b16_stain_xattn/precompute_stain_stats.slurm
# ============================================================

#SBATCH --job-name=vit_xattn_precompute
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-B16-Stain-XAttn/vit_b16_stain_xattn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-B16-Stain-XAttn/vit_b16_stain_xattn_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/vit_b16_stain_xattn_config.yaml
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="vit_b16_stain_xattn_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/vit_b16_stain_xattn_config.yaml"
LOG_DIR="${WORKSPACE}/logs/ViT-B16-Stain-XAttn/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Offline guarantee — no network calls
# ------------------------------------------------------------
export HF_HUB_OFFLINE=1

# ------------------------------------------------------------
# Run precompute
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python ViT-B16-Stain-XAttn/precompute_stain_stats.py --config "${CONFIG}"