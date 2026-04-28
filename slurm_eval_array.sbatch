#!/bin/bash
#SBATCH --job-name=fft_density_eval
#SBATCH --account=m4243_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-119%8

set -euo pipefail
module load conda
conda activate clasp
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/raw full_fft_reference_cache

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CPU_FFT_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

export DIRECT_FULL_FFT_ELEMENTS="${DIRECT_FULL_FFT_ELEMENTS:-4.5e8}"
export DENSITY_MAX_WORK_BYTES="${DENSITY_MAX_WORK_BYTES:-6442450944}"
export REF_GPU_WORK_BYTES="${REF_GPU_WORK_BYTES:-8589934592}"

python run_experiment.py \
  --manifest manifests/suitesparse_square_manifest.csv \
  --matrix-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/raw \
  --density-ratios 0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.0234375,0.015625,0.01171875,0.0078125 \
  --full-reference-cache full_fft_reference_cache \
  --sparse-methods spfft_grid,sparse_direct_grid \
  --sample-fractions 0.00015625,0.0003125,0.000625,0.00125,0.0025,0.005,0.01 \
  --sparse-batch-size 64 \
  --spfft-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --skip-existing \
  --normalization mass


# python run_experiment.py \
#   --manifest manifests/suitesparse_square_manifest.csv \
#   --matrix-index "0" \
#   --output-dir results/raw \
#   --density-ratios 0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.0234375,0.015625,0.01171875,0.0078125 \
#   --full-reference-cache full_fft_reference_cache \
#   --sparse-methods spfft_grid,sparse_direct_grid \
#   --sample-fractions 0.00015625,0.0003125,0.000625,0.00125,0.0025,0.005,0.01 \
#   --sparse-batch-size 64 \
#   --spfft-threads "${SLURM_CPUS_PER_TASK:-16}" \
#   --skip-existing \
#   --normalization mass