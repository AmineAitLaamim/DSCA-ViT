#!/bin/bash
# ============================================================
# SLURM Job Script — Ensemble: ViT CE + UNI CE + ViT CORN + TTA (official test)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_ce_uni_ce_vit_corn_tta_001
# ============================================================
# Evaluates the 3-model ensemble (with TTA) on the official test set
# once (locked). Probabilities are averaged across 6 augmentations,
# then across models. Never logits.
#
# Usage:
#   sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce_vit_corn_tta.slurm
# ============================================================

#SBATCH --job-name=ens_vit_uni_corn_tta
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce_vit_corn_tta/vit_ce_uni_ce_vit_corn_tta_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce_vit_corn_tta/vit_ce_uni_ce_vit_corn_tta_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/ensemble_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/ensemble_config.yaml"
LOG_DIR="${WORKSPACE}/logs/Ensemble/vit_ce_uni_ce_vit_corn_tta/vit_ce_uni_ce_vit_corn_tta_001"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Run official-test evaluation with TTA
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python Ensemble/vit_ce_uni_ce_vit_corn_tta/evaluate_vit_ce_uni_ce_vit_corn_tta.py \
    --config "${CONFIG}" --eval-test
