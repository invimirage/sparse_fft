import math
import time

import cupy as cp
import numpy as np
from scipy import fft as scipy_fft

import approx_fft_benchmark as sparse_bench


def _uniform_density_gpu(matrix, out_size: int) -> np.ndarray:
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    row_gpu = None
    col_gpu = None
    rb_gpu = None
    cb_gpu = None
    linear_gpu = None
    counts_gpu = None
    try:
        row_gpu = cp.asarray(matrix.row.astype(np.int64, copy=False))
        col_gpu = cp.asarray(matrix.col.astype(np.int64, copy=False))
        rb_gpu = cp.minimum((row_gpu * out_size) // rows, out_size - 1)
        cb_gpu = cp.minimum((col_gpu * out_size) // cols, out_size - 1)
        linear_gpu = (rb_gpu * out_size + cb_gpu).astype(cp.int64, copy=False)
        counts_gpu = cp.bincount(linear_gpu, minlength=out_size * out_size).reshape(out_size, out_size)
        counts = cp.asnumpy(counts_gpu).astype(np.int32, copy=False)
    finally:
        if row_gpu is not None:
            del row_gpu
        if col_gpu is not None:
            del col_gpu
        if rb_gpu is not None:
            del rb_gpu
        if cb_gpu is not None:
            del cb_gpu
        if linear_gpu is not None:
            del linear_gpu
        if counts_gpu is not None:
            del counts_gpu
        sparse_bench.clear_gpu_memory()

    _, row_sizes = uniform_bucket_assignments(rows, out_size)
    _, col_sizes = uniform_bucket_assignments(cols, out_size)
    area = np.outer(row_sizes, col_sizes)
    area = np.maximum(area, 1)
    return counts.astype(np.float32) / area.astype(np.float32)


def radial_entropy_from_mag_cpu(mag: np.ndarray, radial_bins: int) -> dict:
    mag = np.asarray(mag, dtype=np.float32)
    log_mag = np.log1p(mag.astype(np.float64, copy=False))
    rows, cols = mag.shape
    y = np.arange(rows, dtype=np.float32)[:, None]
    x = np.arange(cols, dtype=np.float32)[None, :]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = float(radius.max()) if radius.size else 0.0
    if r_max > 0:
        bins = np.clip(np.floor((radius / r_max) * radial_bins).astype(np.int32), 0, radial_bins - 1)
    else:
        bins = np.zeros_like(radius, dtype=np.int32)

    energy = (mag ** 2).ravel().astype(np.float64, copy=False)
    radial_energy = np.bincount(bins.ravel(), weights=energy, minlength=radial_bins).astype(np.float64, copy=False)
    radial = radial_energy / radial_energy.sum() if radial_energy.sum() > 0 else radial_energy

    prob = energy / energy.sum() if energy.sum() > 0 else energy
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log(prob))) if prob.size else 0.0
    return {
        "radial": radial.astype(np.float64),
        "entropy": entropy,
        "std": float(np.std(log_mag, dtype=np.float64)),
    }


def fft_features_from_dense_gpu(dense_gpu: cp.ndarray, radial_bins: int) -> dict:
    started = time.perf_counter()
    spectrum = cp.fft.fftshift(cp.fft.fft2(dense_gpu))
    mag = cp.abs(spectrum).astype(cp.float32)

    rows, cols = mag.shape
    y = cp.arange(rows, dtype=cp.float32)[:, None]
    x = cp.arange(cols, dtype=cp.float32)[None, :]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = cp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = float(cp.max(radius))
    bins = cp.clip(cp.floor((radius / r_max) * radial_bins).astype(cp.int32), 0, radial_bins - 1)

    energy = (mag ** 2).ravel()
    radial_energy = cp.bincount(bins.ravel(), weights=energy, minlength=radial_bins)
    radial = radial_energy / radial_energy.sum() if float(radial_energy.sum()) > 0 else radial_energy

    prob = energy / energy.sum() if float(energy.sum()) > 0 else energy
    prob = prob[prob > 0]
    entropy = -cp.sum(prob * cp.log(prob)) if prob.size else cp.asarray(0.0, dtype=cp.float32)
    log_mag = cp.log1p(mag)

    elapsed = time.perf_counter() - started
    result = {
        "radial": cp.asnumpy(radial).astype(np.float64),
        "entropy": float(entropy),
        "std": float(cp.std(log_mag)),
        "elapsed": elapsed,
    }
    del spectrum
    del mag
    del y
    del x
    del radius
    del bins
    del energy
    del radial_energy
    del log_mag
    sparse_bench.clear_gpu_memory()
    return result


def _resize_axis_weights(src_size: int, dst_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if src_size <= 0 or dst_size <= 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.float64)
    if src_size == 1 or dst_size == 1:
        return np.zeros(dst_size, dtype=np.intp), np.zeros(dst_size, dtype=np.intp), np.zeros(dst_size, dtype=np.float64)
    src_positions = np.linspace(0.0, float(src_size - 1), dst_size, dtype=np.float64)
    lo = np.floor(src_positions).astype(np.intp)
    hi = np.clip(lo + 1, 0, src_size - 1)
    weights = src_positions - lo.astype(np.float64)
    at_end = lo >= src_size - 1
    lo[at_end] = src_size - 1
    hi[at_end] = src_size - 1
    weights[at_end] = 0.0
    return lo, hi, weights


def resized_full_spectrum_std(sample_grid: np.ndarray, rows: int, cols: int, row_block: int = 64) -> float:
    sample_grid = np.asarray(sample_grid, dtype=np.float32)
    if sample_grid.ndim != 2 or sample_grid.shape[0] == 0 or sample_grid.shape[1] == 0:
        return math.nan

    row_lo, row_hi, row_w = _resize_axis_weights(sample_grid.shape[0], rows)
    col_lo, col_hi, col_w = _resize_axis_weights(sample_grid.shape[1], cols)
    total_count = 0
    sum_vals = 0.0
    sum_sq = 0.0
    for start in range(0, rows, row_block):
        stop = min(start + row_block, rows)
        rlo = row_lo[start:stop]
        rhi = row_hi[start:stop]
        rw = row_w[start:stop][:, None]
        c00 = sample_grid[np.ix_(rlo, col_lo)]
        c01 = sample_grid[np.ix_(rlo, col_hi)]
        c10 = sample_grid[np.ix_(rhi, col_lo)]
        c11 = sample_grid[np.ix_(rhi, col_hi)]
        top = c00 * (1.0 - col_w)[None, :] + c01 * col_w[None, :]
        bottom = c10 * (1.0 - col_w)[None, :] + c11 * col_w[None, :]
        block = top * (1.0 - rw) + bottom * rw
        block_log = np.log1p(block.astype(np.float64, copy=False))
        sum_vals += float(block_log.sum(dtype=np.float64))
        sum_sq += float(np.square(block_log, dtype=np.float64).sum(dtype=np.float64))
        total_count += block_log.size

    if total_count == 0:
        return math.nan
    mean = sum_vals / total_count
    var = max(sum_sq / total_count - mean * mean, 0.0)
    return float(math.sqrt(var))


def uniform_bucket_assignments(size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    bucket = np.floor(np.arange(size, dtype=np.float64) * out_size / size).astype(np.int64)
    bucket = np.clip(bucket, 0, out_size - 1)
    bucket_sizes = np.bincount(bucket, minlength=out_size)
    return bucket, bucket_sizes


def adaptive_bucket_assignments(size: int, coords: np.ndarray, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    if size <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(out_size, dtype=np.int64)
    if coords.size == 0:
        return uniform_bucket_assignments(size, out_size)

    hist = np.bincount(coords.astype(np.int64, copy=False), minlength=size).astype(np.float64, copy=False)
    if not np.any(hist):
        return uniform_bucket_assignments(size, out_size)

    total = float(hist.sum())
    targets = np.linspace(0.0, total, out_size + 1)
    cdf = np.cumsum(hist)
    edges = np.searchsorted(cdf, targets[1:-1], side="left") + 1
    edges = np.clip(edges.astype(np.int64, copy=False), 1, size - 1) if size > 1 else np.zeros(0, dtype=np.int64)

    if edges.size:
        edges = np.maximum.accumulate(edges)
        edges = np.minimum(edges, np.arange(1, edges.size + 1) + size - out_size)
        edges = np.maximum.accumulate(edges)

    full_edges = np.concatenate(([0], edges, [size]))
    bucket_sizes = np.diff(full_edges)
    bucket_sizes = np.maximum(bucket_sizes, 1)

    bucket = np.empty(size, dtype=np.int64)
    for idx in range(out_size):
        bucket[full_edges[idx]:full_edges[idx + 1]] = idx
    return bucket, bucket_sizes


def compress_matrix_to_fixed(matrix, out_size: int, mode: str = "uniform") -> np.ndarray:
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape

    if mode == "uniform":
        try:
            return _uniform_density_gpu(matrix, out_size)
        except cp.cuda.memory.OutOfMemoryError:
            sparse_bench.clear_gpu_memory()
        except Exception:
            sparse_bench.clear_gpu_memory()

    if mode == "adaptive":
        row_bucket, row_sizes = adaptive_bucket_assignments(rows, matrix.row, out_size)
        col_bucket, col_sizes = adaptive_bucket_assignments(cols, matrix.col, out_size)
    elif mode == "uniform":
        row_bucket, row_sizes = uniform_bucket_assignments(rows, out_size)
        col_bucket, col_sizes = uniform_bucket_assignments(cols, out_size)
    else:
        raise ValueError(f"Unknown density-map mode: {mode}")

    rb = row_bucket[matrix.row]
    cb = col_bucket[matrix.col]
    counts = np.zeros((out_size, out_size), dtype=np.int32)
    np.add.at(counts, (rb, cb), 1)

    area = np.outer(row_sizes, col_sizes)
    area = np.maximum(area, 1)
    return counts.astype(np.float32) / area.astype(np.float32)


def coarse_spectrum_magnitude(matrix, out_size: int, mode: str = "uniform") -> np.ndarray:
    density_map = compress_matrix_to_fixed(matrix, out_size, mode=mode)
    try:
        dense_gpu = cp.asarray(density_map, dtype=cp.float32)
        spectrum = cp.fft.fftshift(cp.fft.fft2(dense_gpu))
        mag = cp.asnumpy(cp.abs(spectrum).astype(cp.float32))
    except cp.cuda.memory.OutOfMemoryError:
        sparse_bench.clear_gpu_memory()
        spectrum = scipy_fft.fftshift(scipy_fft.fft2(density_map.astype(np.float32, copy=False)))
        mag = np.abs(spectrum).astype(np.float32, copy=False)
    finally:
        if "dense_gpu" in locals():
            del dense_gpu
        if "spectrum" in locals():
            del spectrum
        sparse_bench.clear_gpu_memory()
    return mag


def density_map_fft_features(matrix, out_size: int, radial_bins: int, mode: str = "uniform") -> dict:
    started = time.perf_counter()
    density_map = compress_matrix_to_fixed(matrix, out_size, mode=mode)
    compress_elapsed = time.perf_counter() - started
    sparse_bench.clear_gpu_memory()
    fft_started = time.perf_counter()
    dense_gpu = cp.asarray(density_map, dtype=cp.float32)
    fft_result = fft_features_from_dense_gpu(dense_gpu, radial_bins)
    transfer_plus_fft_elapsed = time.perf_counter() - fft_started
    del dense_gpu
    sparse_bench.clear_gpu_memory()
    fft_result["elapsed"] += compress_elapsed
    fft_result["compress_elapsed"] = compress_elapsed
    fft_result["transfer_fft_elapsed"] = transfer_plus_fft_elapsed
    fft_result["mode"] = mode
    return fft_result


def density_interpolated_full_spectrum_std(matrix, out_size: int, rows: int, cols: int, mode: str = "uniform") -> tuple[float, float]:
    started = time.perf_counter()
    density_map = compress_matrix_to_fixed(matrix, out_size, mode=mode)
    try:
        dense_gpu = cp.asarray(density_map, dtype=cp.float32)
        spectrum = cp.fft.fftshift(cp.fft.fft2(dense_gpu))
        mag = cp.asnumpy(cp.abs(spectrum).astype(cp.float32))
    except cp.cuda.memory.OutOfMemoryError:
        sparse_bench.clear_gpu_memory()
        spectrum = scipy_fft.fftshift(scipy_fft.fft2(density_map.astype(np.float32, copy=False)))
        mag = np.abs(spectrum).astype(np.float32, copy=False)
    finally:
        if "dense_gpu" in locals():
            del dense_gpu
        if "spectrum" in locals():
            del spectrum
        sparse_bench.clear_gpu_memory()
    std_value = resized_full_spectrum_std(mag, rows, cols)
    return std_value, time.perf_counter() - started
