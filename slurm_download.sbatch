#!/bin/bash
#SBATCH --job-name=fft_dl_square
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data manifests

python download_square_matrices.py \
  --output-root data \
  --manifest manifests/suitesparse_square_manifest.csv \
  --full-limit 100 \
  --case-limit 20
