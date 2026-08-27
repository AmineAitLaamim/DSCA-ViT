#!/bin/bash
# ============================================================
# SLURM Job Script — Ensemble: ViT-B16 CORN + UNI CE (validation)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_corn_uni_ce_001
# ============================================================
# Runs VALIDATION only (default, no --eval-test) to keep the
# official test set protected until validation results are
# reviewed. Only probabilities are averaged, never logits.
#
# Usage:
#   sbatch slurm/slurm_ensemble/evaluate_vit_corn_uni_ce.slurm
# ============================================================

#SBATCH --job-name=ens_corn_uni
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_corn_uni_ce/vit_corn_uni_ce_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_corn_uni_ce/vit_corn_uni_ce_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/ensemble_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/ensemble_config.yaml"
LOG_DIR="${WORKSPACE}/logs/Ensemble/vit_corn_uni_ce/vit_corn_uni_ce_001"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Run VALIDATION (default; no --eval-test).
# Official test is COMMENTED OUT until validation is reviewed.
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python Ensemble/vit_corn_uni_ce/evaluate_vit_corn_uni_ce.py \
    --config "${CONFIG}"

# Official test (once, locked) — enable only after review:
# uv run python Ensemble/vit_corn_uni_ce/evaluate_vit_corn_uni_ce.py \
#     --config "${CONFIG}" --eval-test