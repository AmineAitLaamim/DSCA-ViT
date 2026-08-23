#!/bin/bash
# SLURM eval -- UNI-CORN
#SBATCH --job-name=uni_corn_eval
#SBATCH --account=manapy-1wabcjwe938-default-gpu
#SBATCH --qos=default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-CORN/uni_corn_001/slurm_%j.out
#SBATCH --error=/home/amine.aitlaamim-ext/projects/DSCA-ViT/logs/UNI-CORN/uni_corn_001/slurm_%j.err

WORKSPACE="/home/amine.aitlaamim-ext/projects/DSCA-ViT"
REPO_DIR="${WORKSPACE}/code/DSCA-ViT"
CONFIG="${REPO_DIR}/configs/uni_corn_config.yaml"
mkdir -p "${WORKSPACE}/logs/UNI-CORN/uni_corn_001"
cd "${REPO_DIR}"
uv run python UNI-CORN/evaluate_uni_corn.py --config "${CONFIG}"