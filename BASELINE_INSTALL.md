# Baseline Library Installation

This benchmark can request the sparse baselines below through `--sparse-methods`:

- `spfft_grid`
- `sparse_direct_grid`
- `finufft_grid`
- `cufinufft_grid`
- `fps_sft`
- `kapralov_sfft`

Only `spfft_grid`, `sparse_direct_grid`, and `finufft_grid` are currently wired to runnable Python implementations in this repository. `cufinufft_grid` is wired but requires the optional `cufinufft` Python package. `fps_sft` and `kapralov_sfft` are placeholders until external adapters are added.

## Base Environment

On Perlmutter/NERSC, start from the same environment used by `slurm_eval_array.sbatch`:

```bash
module load conda
conda activate clasp
cd /path/to/hpc_density_fft
```

The core benchmark also needs the existing sibling project import path:

```bash
python - <<'PY'
import approx_fft_benchmark
from sparse_methods import sparse_sampling, spfft
print('core imports ok')
PY
```

If `spfft_grid` cannot find `libspfft.so`, set:

```bash
export SPFFT_LIBRARY_PATH=/path/to/libspfft.so
```

## FINUFFT CPU Baseline

`finufft_grid` uses the Python `finufft` package and runs on CPU.

Install:

```bash
python -m pip install finufft
```

Verify:

```bash
python - <<'PY'
import finufft
print('finufft ok', getattr(finufft, '__version__', 'unknown'))
PY
```

Current environment status checked here:

```text
finufft: import ok, version=2.5.1
```

## cuFINUFFT GPU Baseline

`cufinufft_grid` first tries `cufinufft`. If that import or execution fails, the current code falls back to CPU `finufft` and records backend `finufft_cpu`. Therefore, seeing `cufinufft_grid` rows does not prove cuFINUFFT ran unless the row backend is `cufinufft_gpu`.

Install a wheel compatible with the CUDA/CuPy stack in the active environment. If available for the platform:

```bash
python -m pip install cufinufft
```

Verify:

```bash
python - <<'PY'
import cupy as cp
import cufinufft
print('cupy cuda runtime', cp.cuda.runtime.runtimeGetVersion())
print('cufinufft ok', getattr(cufinufft, '__version__', 'unknown'))
PY
```

Current environment status checked here:

```text
cufinufft: import failed: ModuleNotFoundError: No module named 'cufinufft'
```

Until that import passes, `cufinufft_grid` should be treated as not installed.

## SpFFT/Grid Baselines

`spfft_grid` and `sparse_direct_grid` use existing project code:

- `spfft_grid`: tries GPU-assisted pruned SpFFT/JL-style implementation, then CPU fallback.
- `sparse_direct_grid`: direct sparse DFT on the sampled frequency grid.

No extra Python package is installed from this directory. Verify with a smoke run and check JSON statuses:

```bash
python run_experiment.py \
  --manifest manifests/suitesparse_square_manifest.csv \
  --matrix-index 0 \
  --output-dir results/smoke_baselines \
  --density-ratios 0.015625 \
  --sample-fractions 0.00125 \
  --sparse-methods spfft_grid,sparse_direct_grid \
  --sparse-batch-size 64
```

Then inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path('results/smoke_baselines').glob('*.json'):
    doc = json.load(path.open())
    for record in doc['records']:
        if record.get('method') in {'spfft_grid', 'sparse_direct_grid'}:
            print(record['method'], record['status'], record.get('backend'), record.get('note', ''))
PY
```

## FPS-SFT Baseline

`fps_sft` is not installed or adapted in the current codebase. `run_experiment.py` intentionally returns dependency-missing for it:

```python
if method in {'fps_sft', 'kapralov_sfft'}:
    raise ImportError(f'{method} external adapter is not configured in this environment')
```

To enable it, add an adapter that converts the benchmark sample points into the FPS-SFT implementation's expected input and returns coefficients compatible with `grid_log_from_coeffs`. The adapter must provide at least:

- input: sparse matrix COO rows/cols, sample points, grid metadata
- output: sampled complex coefficients in the same order as `points`
- timing: elapsed compute time, excluding benchmark interpolation/metric time

After adding the adapter, wire it in `sparse_grid_features()` for `method == 'fps_sft'`.

## Kapralov SFFT Baseline

`kapralov_sfft` is also not installed or adapted in the current codebase. It requires an external implementation and usually a compiled native dependency stack such as FFTW3.

To enable it:

1. Build the external implementation outside this benchmark.
2. Add a Python adapter that accepts the same sampled grid points used by this benchmark.
3. Return coefficients in point order and an elapsed compute time.
4. Wire the adapter in `sparse_grid_features()` for `method == 'kapralov_sfft'`.

Until those steps are done, keep `kapralov_sfft` out of production SLURM runs or expect `skipped:dependency_missing` rows.

## Smoke Test All Installed Baselines

After installing optional packages, run a small smoke test:

```bash
rm -rf results/smoke_baselines
python run_experiment.py \
  --manifest manifests/suitesparse_square_manifest.csv \
  --matrix-index 0 \
  --output-dir results/smoke_baselines \
  --density-ratios 0.015625 \
  --sample-fractions 0.00125 \
  --sparse-methods spfft_grid,sparse_direct_grid,finufft_grid,cufinufft_grid \
  --sparse-batch-size 64
```

Check methods, statuses, and backends:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path('results/smoke_baselines').glob('*.json'):
    print(path)
    doc = json.load(path.open())
    for record in doc['records']:
        method = record.get('method')
        if method and method.endswith('_grid'):
            print(method, record.get('status'), record.get('backend'), record.get('note', ''))
PY
```

Expected interpretation:

- `status == ok` means the method completed for that sample fraction.
- `backend == cufinufft_gpu` means cuFINUFFT actually ran.
- `backend == finufft_cpu` under `cufinufft_grid` means GPU cuFINUFFT did not run and CPU FINUFFT fallback was used.
- `skipped:dependency_missing` means the package or external adapter is not installed.
- `error:TypeError` or other errors mean the method imported but failed at runtime and should not be counted as installed.
