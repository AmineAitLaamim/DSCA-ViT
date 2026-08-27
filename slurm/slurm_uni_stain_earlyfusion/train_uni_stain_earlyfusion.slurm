#!/bin/bash
# ============================================================
# SLURM Job Script — Train UNI-Stain-EarlyFusion
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_stain_earlyfusion_001
# ============================================================
# Trains the UNI backbone with a replaced 5-channel patch-embed
# projection (RGB + H + DAB early fusion), 2-stage (Adam; Stage 2
# unfreezes the whole backbone).
#
# Usage:
#   sbatch slurm/slurm_uni_stain_earlyfusion/train_uni_stain_earlyfusion.slurm
#
# Prerequisite: run precompute_stain_stats.slurm first.
# ============================================================

#SBATCH --job-name=uni_earlyfusion_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-EarlyFusion/uni_stain_earlyfusion_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-EarlyFusion/uni_stain_earlyfusion_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_stain_earlyfusion_config.yaml
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_stain_earlyfusion_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_stain_earlyfusion_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Stain-EarlyFusion/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run training (Stage 1 -> Stage 2)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python UNI-Stain-EarlyFusion/train_uni_stain_earlyfusion.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python UNI-Stain-EarlyFusion/train_uni_stain_earlyfusion.py --config "${CONFIG}" --resume