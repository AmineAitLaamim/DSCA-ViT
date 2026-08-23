#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate UNI-Stain-Attention on Official Test Set
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_stain_attn_001
# ============================================================
# Loads the best checkpoint (best.pt — NEVER last.pt) and
# evaluates on the official test set exactly once.
#
# Usage:
#   sbatch slurm/slurm_uni_stain_attn/evaluate_uni_stain_attn.slurm
# ============================================================

#SBATCH --job-name=uni_stain_attn_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-Attention/uni_stain_attn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Stain-Attention/uni_stain_attn_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_stain_attn_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_stain_attn_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_stain_attn_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Stain-Attention/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv (no conda)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run evaluation — uses best.pt (NEVER last.pt)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python UNI-Stain-Attention/evaluate_uni_stain_attn.py --config "${CONFIG}"

# Optional: evaluate a specific checkpoint explicitly:
# uv run python UNI-Stain-Attention/evaluate_uni_stain_attn.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/UNI-Stain-Attention/${EXPERIMENT}/best.pt"