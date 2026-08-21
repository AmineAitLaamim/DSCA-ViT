#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate Plain ViT-B16 Baseline
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: plain_vit_baseline_001
# ============================================================
# Usage:
#   sbatch slurm/slurm_baseline/evaluate_plain_vit_baseline.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must be literal absolute
# strings matching log_dir in configs/plain_vit_baseline_config.yaml.
# ============================================================

#SBATCH --job-name=vit_baseline_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-Baseline/plain_vit_baseline_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/ViT-Baseline/plain_vit_baseline_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/plain_vit_baseline_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="plain_vit_baseline_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/plain_vit_baseline_config.yaml"
LOG_DIR="${WORKSPACE}/logs/ViT-Baseline/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv inside the repository
# uv run picks up .venv automatically; no activation needed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------
cd "${REPO_DIR}"

# Evaluate using best_stage2.pt (default):
uv run python utils/evaluate_plain_vit_baseline.py --config "${CONFIG}"

# Evaluate a specific checkpoint:
# uv run python utils/evaluate_plain_vit_baseline.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/ViT-Baseline/${EXPERIMENT}/best_stage2.pt"