#!/bin/bash
# ============================================================
# SLURM Job Script — Ensemble: ViT-B16 + UNI CE + TTA (validation)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_corn_uni_ce_tta_001
# ============================================================
# Runs VALIDATION (with TTA) on the default path (no --eval-test)
# to protect the official test set until validation is reviewed.
# Only probabilities are averaged, never logits.
#
# Usage:
#   sbatch slurm/slurm_ensemble/evaluate_vit_corn_uni_ce_tta.slurm
# ============================================================

#SBATCH --job-name=ens_corn_uni_tta
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_corn_uni_ce_tta/vit_corn_uni_ce_tta_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_corn_uni_ce_tta/vit_corn_uni_ce_tta_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/ensemble_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/ensemble_config.yaml"
LOG_DIR="${WORKSPACE}/logs/Ensemble/vit_corn_uni_ce_tta/vit_corn_uni_ce_tta_001"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Run VALIDATION with TTA (default; no --eval-test).
# Leave default; enable --eval-test only after validation review.
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python Ensemble/vit_corn_uni_ce_tta/evaluate_vit_corn_uni_ce_tta.py \
    --config "${CONFIG}"

# Official test (once, locked) — enable only after review:
# uv run python Ensemble/vit_corn_uni_ce_tta/evaluate_vit_corn_uni_ce_tta.py \
#     --config "${CONFIG}" --eval-test