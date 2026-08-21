#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate DSS-ViT v2.2 on Official Test Set
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: dss_vit_v2_2_001
# ============================================================
# Loads best_stage3.pt and evaluates on the official test set.
#
# Usage:
#   sbatch slurm/slurm_dss_vit_v2_2/evaluate_dss_vit_v2_2.slurm
# ============================================================

#SBATCH --job-name=dss_vit_v2_2_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT-v2.2/dss_vit_v2_2_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT-v2.2/dss_vit_v2_2_001/slurm_%j.err

# ============================================================
# Workspace
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="dss_vit_v2_2_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_v2_2_config.yaml"
LOG_DIR="${WORKSPACE}/logs/DSS-ViT-v2.2/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

# Evaluate using best_stage3.pt (default):
uv run python utils/evaluate_dss_vit_v2_2.py --config "${CONFIG}"

# Evaluate a specific checkpoint:
# uv run python utils/evaluate_dss_vit_v2_2.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/DSS-ViT-v2.2/${EXPERIMENT}/best_stage3.pt"