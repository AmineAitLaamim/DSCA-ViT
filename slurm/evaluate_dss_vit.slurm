#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate DSS-ViT on Official Test Set
# Toubkal / UM6P HPC  |  experiment: dss_vit_001
# ============================================================
# Usage:
#   sbatch slurm/evaluate_dss_vit.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must therefore be written
# as literal absolute strings matching log_dir in the config.
# ============================================================

#SBATCH --job-name=dss_vit_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/dss_vit_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/DSCA-ViT"
EXPERIMENT="dss_vit_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_config.yaml"
LOG_DIR="${WORKSPACE}/logs/DSS-ViT/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment
# Check available modules with: module avail CUDA && module avail Python
# Then load the appropriate versions and activate the conda env.
# ------------------------------------------------------------
# module load CUDA/<version>
# module load Python/<version>
# conda activate dss_vit_env

# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Evaluate using best_stage3.pt (default):
python utils/evaluate_dss_vit.py --config "${CONFIG}"

# Evaluate a specific checkpoint:
# python utils/evaluate_dss_vit.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/DSS-ViT/${EXPERIMENT}/best_stage3.pt"