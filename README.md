# HPC Density FFT Benchmark

This folder is a batch-job version of the current sparse FFT project for SuiteSparse square matrices.

## What It Runs

The pipeline has three stages:

1. `download_square_matrices.py` downloads and cleans SuiteSparse square matrices.
2. `run_experiment.py` evaluates full FFT reference, density-map/compression FFT normalization variants, sparse FFT grid methods, and sparse error-vs-time curves across multiple sampling budgets.
3. `analyze_results.py` merges JSON outputs and writes CSV/Markdown summaries.

Datasets:

- `full_eval`: about 100 square matrices with size 10k-20k.
- `case_study`: about 20 square matrices with size 50k-100k.

Density map curve budgets:

- Default `--density-ratios` scans from `0.5` down to `0.0078125`.
- For an `n x n` matrix, each density candidate uses `out_size = round(n * density_ratio)`.
- Candidates that exceed `DENSITY_MAX_WORK_BYTES` or hit OOM are skipped and smaller ratios continue running.

Normalization methods:

- `none`: `D`
- `mass`: `D * sum(X) / sum(D)`
- `unit`: `D / sum(D)`

Compression baselines:

- `density_fft`: density/average pooled map with existing normalizations.
- `avg_pool_fft`, `max_pool_fft`, `nearest_downsample_fft`, `gaussian_compression_fft`: spatial compression baselines using `mass` normalization.

Sparse FFT comparison methods:

- `spfft_grid`: existing pruned SpFFT/JL-style rectangular frequency grid, GPU-assisted when possible, CPU fallback.
- `sparse_direct_grid`: existing direct sparse DFT on the same rectangular frequency grid, batched on GPU.
- `finufft_grid`: optional FINUFFT type-3 evaluation on the same sampled frequency grid.
- `cufinufft_grid`: optional cuFINUFFT evaluation when available, falling back to FINUFFT.
- `fps_sft` and `kapralov_sfft`: optional external baselines; currently skipped unless adapters/dependencies are configured.

Sparse curve sampling budgets:

- Default `--sample-fractions` is `0.00015625,0.0003125,0.000625,0.00125,0.0025,0.005,0.01`.
- Each value is an axis fraction. For an `n x n` matrix, the script samples about `(n * fraction)^2` shifted-frequency grid points.
- Every sparse candidate row records `curve_index`, `sample_fraction`, `sample_axis_rows`, `sample_axis_cols`, `sample_count`, errors, and timings.

Hybrid-selection features:

- Every matrix JSON includes top-level `hybrid_features` from the `1024 x 1024` density map.
- Recorded fields are `density_1024_entropy_norm`, `density_1024_std`, `density_1024_gini`, plus build/compute timings.

Metrics:

- interpolated `log_mae`
- interpolated and direct spectral entropy error
- interpolated and direct 16-bin radial ratio error

## GPU/Parallelism

The expensive paths are GPU-first:

- dense full FFT reference uses CuPy FFT on GPU, CPU scipy FFT fallback with `CPU_FFT_WORKERS`.
- density-map construction uses GPU `bincount`.
- density FFT, interpolation, entropy, radial bins, and log-MAE use CuPy arrays when available.
- sparse direct grid uses batched GPU phase summation through the existing project code.
- SpFFT grid uses the existing pruned GPU-assisted implementation and falls back to CPU pruned code.

Full FFT reference handling:

- Small matrices use direct CuPy full FFT when `rows * cols <= DIRECT_FULL_FFT_ELEMENTS`.
- Larger matrices automatically build a shifted full FFT magnitude cache under `full_fft_reference_cache/` using row FFT and column FFT blocks on GPU, stored as `float32` memmap on disk.
- Metrics against cached references are computed block-by-block, so log-MAE, interpolated entropy, and radial ratio do not require loading the full spectrum into RAM.
- This removes the previous large-matrix skip behavior; very large case studies are limited by wall time and disk capacity rather than by an artificial element cap.

## Submit On SLURM

From this directory:

```bash
mkdir -p logs data manifests results/raw results/analysis
sbatch slurm_download.sbatch
```

After download finishes, update the array range in `slurm_eval_array.sbatch` if fewer than 120 matrices were downloaded, then submit:

```bash
sbatch slurm_eval_array.sbatch
sbatch slurm_analyze.sbatch
```

Or submit the dependency chain:

```bash
bash submit_pipeline.sh
```

## Manual Test

Run one matrix:

```bash
python run_experiment.py \
  --manifest manifests/suitesparse_square_manifest.csv \
  --matrix-index 0 \
  --output-dir results/raw \
  --density-ratios 0.03125,0.015625 \
  --sample-fractions 0.00125,0.0025,0.005
```

Analyze:

```bash
python analyze_results.py --input-dir results/raw --output-dir results/analysis
```

Outputs:

- `results/raw/<matrix>.json`: one JSON per matrix.
- `results/analysis/all_results.csv`: flattened per-method rows.
- `results/analysis/summary_by_method.csv`: grouped means/medians/std, including sparse curve sampling columns.
- `results/analysis/best_by_metric.csv`: lowest mean error by split/metric.
- `results/analysis/report.md`: compact Markdown report.

## Notes

- The scripts import existing utilities from `../sparse_fft`; the original project files are not modified.
- If `libspfft.so` is not in the Python environment library path, set `SPFFT_LIBRARY_PATH` before submitting.
- Optional baselines require `finufft`/`cufinufft`; `fps_sft` requires an Octave/MATLAB adapter, and `kapralov_sfft` requires a compiled FFTW3-based external adapter.
- See `BASELINE_INSTALL.md` for baseline installation, current support status, and smoke-test commands.
- If an HPC partition uses different directives, edit `#SBATCH --gres=gpu:1`, memory, time, or array throttling in the `.sbatch` files.
