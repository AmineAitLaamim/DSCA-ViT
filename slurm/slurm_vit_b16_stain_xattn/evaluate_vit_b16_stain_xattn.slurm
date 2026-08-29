#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate ViT-B16-Stain-XAttn on Official Test Set
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_b16_stain_xattn_001
# ============================================================
# Loads best_stage2.pt (NEVER last.pt) and evaluates on the official
# test set exactly once.
#
# Usage:
#   sbatch slurm/slurm_vit_b16_stain_xattn/evaluate_vit_b16_stain_xattn.slurm
# ============================================================

#SBATCH --job-name=vit_xattn_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
# Offline guarantee — no network calls
# ------------------------------------------------------------
export HF_HUB_OFFLINE=1

# ------------------------------------------------------------
# Run evaluation — uses best_stage2.pt (NEVER last.pt)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python ViT-B16-Stain-XAttn/evaluate_vit_b16_stain_xattn.py --config "${CONFIG}"

# Optional: evaluate a specific checkpoint explicitly:
# uv run python ViT-B16-Stain-XAttn/evaluate_vit_b16_stain_xattn.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/ViT-B16-Stain-XAttn/${EXPERIMENT}/best_stage2.pt"