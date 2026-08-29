#!/bin/bash
# ============================================================
# SLURM Job Script — Train ViT-B16-Stain-XAttn
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_b16_stain_xattn_001
# ============================================================
# Trains ViT-B16 (ImageNet-pretrained) with gated cross-attention
# fusion of H/DAB stain maps, 2-stage (Adam; Stage-2 unfreezes all).
#
# Usage:
#   sbatch slurm/slurm_vit_b16_stain_xattn/train_vit_b16_stain_xattn.slurm
#
# Prerequisite: run precompute_stain_stats.slurm first.
# ============================================================

#SBATCH --job-name=vit_xattn_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
# Offline guarantee — pretrained weights must be HF-cached
# (see pre-implementation verification section 2.2)
# ------------------------------------------------------------
export HF_HUB_OFFLINE=1

# ------------------------------------------------------------
# Run training (Stage 1 -> Stage 2)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python ViT-B16-Stain-XAttn/train_vit_b16_stain_xattn.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python ViT-B16-Stain-XAttn/train_vit_b16_stain_xattn.py --config "${CONFIG}" --resume