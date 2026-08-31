#!/bin/bash
# ============================================================
# SLURM Job Script — Ensemble: ViT CE + UNI CE + ViT CORN (official test)
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: vit_ce_uni_ce_vit_corn_001
# ============================================================
# Evaluates the 3-model ensemble on the official test set once
# (locked). Only probs are averaged; CORN uses its [B,4] probs.
#
# --force is included because the shared test lock already names
# another variant; running vit_ce_uni_ce_vit_corn officially
# deliberately overrides it. Remove --force to respect the lock.
#
# Usage:
#   sbatch slurm/slurm_ensemble/evaluate_vit_ce_uni_ce_vit_corn.slurm
# ============================================================

#SBATCH --job-name=ens_vit_uni_corn
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce_vit_corn/vit_ce_uni_ce_vit_corn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/Ensemble/vit_ce_uni_ce_vit_corn/vit_ce_uni_ce_vit_corn_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/ensemble_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/ensemble_config.yaml"
LOG_DIR="${WORKSPACE}/logs/Ensemble/vit_ce_uni_ce_vit_corn/vit_ce_uni_ce_vit_corn_001"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Run official-test evaluation
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python Ensemble/vit_ce_uni_ce_vit_corn/evaluate_vit_ce_uni_ce_vit_corn.py \
    --config "${CONFIG}" --eval-test --force
