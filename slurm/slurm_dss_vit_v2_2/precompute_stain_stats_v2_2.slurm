#!/bin/bash
# ============================================================
# SLURM Job Script — Precompute Global H/DAB Stain Statistics (v2.2)
# Toubkal / UM6P HPC  |  partition: compute (CPU)  |  experiment: dss_vit_v2_2_001
# ============================================================
# Computes global H/DAB mean/std on the TRAINING split only
# (from the WSI-aware split). Creates split_indices_wsi.npz on
# first run.
#
# Usage:
#   sbatch slurm/slurm_dss_vit_v2_2/precompute_stain_stats_v2_2.slurm
# ============================================================

#SBATCH --job-name=dss_vit_v2_2_precompute
#SBATCH --account=manapy-um6p-st-msda-1wabcjwe938-low-cpu
#SBATCH --qos=low-cpu
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run precompute
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python scripts/precompute_stain_stats_v2_2.py --config "${CONFIG}"