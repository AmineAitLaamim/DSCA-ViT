#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate UNI-Stain-EarlyFusion on Official Test Set
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_stain_earlyfusion_001
# ============================================================
# Loads best_stage2.pt (NEVER last.pt) and evaluates on the official
# test set exactly once.
#
# Usage:
#   sbatch slurm/slurm_uni_stain_earlyfusion/evaluate_uni_stain_earlyfusion.slurm
# ============================================================

#SBATCH --job-name=uni_earlyfusion_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
# Context
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run evaluation — uses best_stage2.pt (NEVER last.pt)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python UNI-Stain-EarlyFusion/evaluate_uni_stain_earlyfusion.py --config "${CONFIG}"

# Optional: evaluate a specific checkpoint explicitly:
# uv run python UNI-Stain-EarlyFusion/evaluate_uni_stain_earlyfusion.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/UNI-Stain-EarlyFusion/${EXPERIMENT}/best_stage2.pt"