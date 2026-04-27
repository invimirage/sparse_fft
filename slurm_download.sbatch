#!/bin/bash
#SBATCH --job-name=fft_dl_square
#SBATCH --account=m4243
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data manifests

python download_square_matrices.py \
  --output-root data \
  --manifest manifests/suitesparse_square_manifest.csv \
  --full-limit 100 \
  --case-limit 20 \
  --workers 16
