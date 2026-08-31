#!/bin/bash
# ============================================================
# SLURM Job Script — Ensemble: ViT-B16 CE + UNI CE (official test)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_ce_uni_ce_001
# ============================================================
# Evaluates the ensemble on the official test set once (locked).
# Only probabilities are averaged, never logits.
#
# --force is included because the shared test lock already names another
# variant (vit_ce_uni_ce_vit_corn_tta); running vit_ce_uni_ce officially
# deliberately overrides it. Remove --force to respect the lock default.
#
# Usage:
#   sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce.slurm
# ============================================================

#SBATCH --job-name=ens_vit_uni
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce/vit_ce_uni_ce_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce/vit_ce_uni_ce_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/ensemble_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/ensemble_config.yaml"
LOG_DIR="${WORKSPACE}/logs/Ensemble/vit_ce_uni_ce/vit_ce_uni_ce_001"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Run official-test evaluation (validation-first protocol in code)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python Ensemble/vit_ce_uni_ce/evaluate_vit_ce_uni_ce.py \
    --config "${CONFIG}" --eval-test --force
