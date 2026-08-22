#!/bin/bash
# ============================================================
# SLURM Job Script — Evaluate UNI Baseline on Official Test Set
# Toubkal / UM6P HPC  |  partition: gpu (A100)  |  experiment: uni_baseline_001
# ============================================================
# Loads the best Stage 2 checkpoint (best_stage2.pt — NEVER
# last.pt) and evaluates on the official test set exactly once.
#
# Usage:
#   sbatch slurm/slurm_uni_baseline/evaluate_uni_baseline.slurm
#
# Note: #SBATCH directives are evaluated before shell variables
# expand.  The --output/--error paths must be literal absolute
# strings matching log_dir in configs/uni_baseline_config.yaml.
# ============================================================

#SBATCH --job-name=uni_baseline_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Baseline/uni_baseline_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-Baseline/uni_baseline_001/slurm_%j.err

# ============================================================
# Workspace — must match configs/uni_baseline_config.yaml paths
# ============================================================
WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
EXPERIMENT="uni_baseline_001"

REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_baseline_config.yaml"
LOG_DIR="${WORKSPACE}/logs/UNI-Baseline/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------
# Environment — uv-managed .venv inside the repository
# uv run picks up .venv automatically; no activation needed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Run evaluation — uses best_stage2.pt (NEVER last.pt)
# ------------------------------------------------------------
cd "${REPO_DIR}"

uv run python baseline/UNI-baseline/evaluate_uni_baseline.py --config "${CONFIG}"

# Optional: evaluate a specific checkpoint explicitly:
# uv run python baseline/UNI-baseline/evaluate_uni_baseline.py --config "${CONFIG}" \
#     --checkpoint "${WORKSPACE}/checkpoints/UNI-Baseline/${EXPERIMENT}/best_stage2.pt"