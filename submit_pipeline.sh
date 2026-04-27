#!/bin/bash

set -euo pipefail

download_job=$(sbatch --parsable slurm_download.sbatch)
echo "download job: ${download_job}"

eval_job=$(sbatch --parsable --dependency=afterok:${download_job} slurm_eval_array.sbatch)
echo "eval array job: ${eval_job}"

analyze_job=$(sbatch --parsable --dependency=afterok:${eval_job} slurm_analyze.sbatch)
echo "analyze job: ${analyze_job}"
