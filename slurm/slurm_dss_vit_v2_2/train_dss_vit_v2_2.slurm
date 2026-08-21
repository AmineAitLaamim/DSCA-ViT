#!/bin/bash
# ============================================================
# SLURM Job Script — Train DSS-ViT v2.2
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: dss_vit_v2_2_001
# ============================================================
# DSS-ViT v2.2: Strong regularization + reduced capacity.
# Uses the retrained baseline's WSI-aware split.
#
# Usage:
#   sbatch slurm/slurm_dss_vit_v2_2/train_dss_vit_v2_2.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must be literal absolute
# strings matching log_dir in configs/dss_vit_v2_2_config.yaml.
# ============================================================

#SBATCH --job-name=dss_vit_v2_2_train
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT-v2.2/dss_vit_v2_2_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT-v2.2/dss_vit_v2_2_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/dss_vit_v2_2_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="dss_vit_v2_2_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_v2_2_config.yaml"
LOG_DIR="${WORKSPACE}/logs/DSS-ViT-v2.2/${EXPERIMENT}"
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
uv run python utils/train_dss_vit_v2_2.py --config "${CONFIG}"

# Multi-GPU DDP (set --gres=gpu:N and --ntasks=N above, then uncomment):
# srun uv run python utils/train_dss_vit_v2_2.py --config "${CONFIG}"

# Resume from last checkpoint:
# uv run python utils/train_dss_vit_v2_2.py --config "${CONFIG}" --resume