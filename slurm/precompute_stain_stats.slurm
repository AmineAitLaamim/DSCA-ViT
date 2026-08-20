#!/bin/bash
# ============================================================
# SLURM Job Script — Precompute H/DAB Stain Statistics
# Toubkal / UM6P HPC  |  partition: compute (CPU)  |  experiment: dss_vit_001
# ============================================================
# Runs on CPU only. H/DAB deconvolution is a lightweight
# preprocessing step — A100 GPU nodes are not needed here.
#
# Usage:
#   sbatch slurm/precompute_stain_stats.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must be literal absolute
# strings matching log_dir in configs/dss_vit_config.yaml.
# ============================================================

#SBATCH --job-name=dss_vit_precompute
#SBATCH --account=manapy-um6p-st-msda-1wabcjwe938-low-cpu
#SBATCH --qos=low-cpu
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/DSS-ViT/dss_vit_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/dss_vit_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="dss_vit_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/dss_vit_config.yaml"
LOG_DIR="${WORKSPACE}/logs/DSS-ViT/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv inside the repository
# uv run picks up .venv automatically; no activation needed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run precompute
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python scripts/precompute_stain_stats.py --config "${CONFIG}"