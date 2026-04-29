#!/usr/bin/env python3

"""Run one matrix through the density-normalization FFT benchmark.

The expensive paths use CuPy when available:

* full FFT reference and log spectrum construction
* density-map binning, compressed FFT, interpolation, entropy, radial metrics
* sparse direct DFT batches for sampled frequency grids

The script writes one JSON file per matrix so it is safe for SLURM array jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import fft as scipy_fft

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only login nodes may not have CUDA.
    cp = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSE_ROOT = REPO_ROOT / "sparse_fft"


def add_sparse_fft_root(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


add_sparse_fft_root(DEFAULT_SPARSE_ROOT)
import approx_fft_benchmark as bench  # noqa: E402
from sparse_methods import sparse_sampling, spfft as spfft_method  # noqa: E402


RADIAL_BINS = 16
NORMALIZATIONS = ("none", "mass", "unit")
COMPRESSION_METHODS = ("density_fft", "avg_pool_fft", "max_pool_fft", "nearest_downsample_fft", "gaussian_compression_fft")
COMPRESSION_BASELINE_NORMALIZATIONS = ("mass",)
DENSITY_RATIOS_DEFAULT = "0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.0234375,0.015625,0.01171875,0.0078125"
DIRECT_FULL_FFT_ELEMENTS = int(float(os.environ.get("DIRECT_FULL_FFT_ELEMENTS", "4.5e8")))
DENSITY_MAX_WORK_BYTES = int(float(os.environ.get("DENSITY_MAX_WORK_BYTES", str(6 * 1024**3))))
REF_ROW_BATCH_CAP = int(os.environ.get("REF_ROW_BATCH_CAP", "4096"))
REF_COL_BATCH_CAP = int(os.environ.get("REF_COL_BATCH_CAP", "2048"))
REF_GPU_WORK_BYTES = int(float(os.environ.get("REF_GPU_WORK_BYTES", str(8 * 1024**3))))
METRIC_ROW_BLOCK = int(os.environ.get("METRIC_ROW_BLOCK", "512"))
METRIC_COL_BLOCK = int(os.environ.get("METRIC_COL_BLOCK", "2048"))


def gpu_available() -> bool:
    return cp is not None


def sync() -> None:
    if gpu_available():
        cp.cuda.Stream.null.synchronize()


def clear_gpu() -> None:
    if gpu_available():
        try:
            bench.clear_gpu_memory()
        except Exception:
            cp.get_default_memory_pool().free_all_blocks()


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _load_manifest_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("path") and row.get("status") in {"downloaded", "exists"}]


def load_manifest_row(path: Path, index: int) -> dict:
    rows = _load_manifest_rows(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"matrix index {index} outside manifest range 0..{len(rows) - 1}")
    return rows[index]


def load_manifest_all(path: Path) -> list[dict]:
    return _load_manifest_rows(path)


def density_map_gpu(matrix, out_size: int):
    if not gpu_available():
        return density_map_cpu(matrix, out_size)
    coo = matrix.tocoo(copy=False)
    rows, cols = coo.shape
    started = time.perf_counter()
    row_gpu = cp.asarray(coo.row.astype(np.int64, copy=False))
    col_gpu = cp.asarray(coo.col.astype(np.int64, copy=False))
    rb = cp.minimum((row_gpu * out_size) // rows, out_size - 1)
    cb = cp.minimum((col_gpu * out_size) // cols, out_size - 1)
    linear = (rb * out_size + cb).astype(cp.int32, copy=False)
    counts = cp.zeros(out_size * out_size, dtype=cp.float32)
    import cupyx
    cupyx.scatter_add(counts, linear, cp.float32(1.0))
    counts = counts.reshape(out_size, out_size)
    row_edges = cp.floor(cp.arange(out_size + 1, dtype=cp.float64) * rows / out_size).astype(cp.int64)
    col_edges = cp.floor(cp.arange(out_size + 1, dtype=cp.float64) * cols / out_size).astype(cp.int64)
    row_sizes = cp.maximum(row_edges[1:] - row_edges[:-1], 1).astype(cp.float32)
    col_sizes = cp.maximum(col_edges[1:] - col_edges[:-1], 1).astype(cp.float32)
    density = counts / (row_sizes[:, None] * col_sizes[None, :])
    sync()
    elapsed = time.perf_counter() - started
    del row_gpu, col_gpu, rb, cb, linear, counts, row_edges, col_edges, row_sizes, col_sizes
    return density, elapsed, "gpu"


def density_map_cpu(matrix, out_size: int):
    coo = matrix.tocoo(copy=False)
    rows, cols = coo.shape
    started = time.perf_counter()
    rb = np.minimum((coo.row.astype(np.int64) * out_size) // rows, out_size - 1)
    cb = np.minimum((coo.col.astype(np.int64) * out_size) // cols, out_size - 1)
    counts = np.bincount(rb * out_size + cb, minlength=out_size * out_size).reshape(out_size, out_size).astype(np.float32)
    row_edges = np.floor(np.arange(out_size + 1, dtype=np.float64) * rows / out_size).astype(np.int64)
    col_edges = np.floor(np.arange(out_size + 1, dtype=np.float64) * cols / out_size).astype(np.int64)
    row_sizes = np.maximum(row_edges[1:] - row_edges[:-1], 1).astype(np.float32)
    col_sizes = np.maximum(col_edges[1:] - col_edges[:-1], 1).astype(np.float32)
    density = counts / (row_sizes[:, None] * col_sizes[None, :])
    return density, time.perf_counter() - started, "cpu"


def avg_pool_compression(matrix, out_size: int):
    density, elapsed, backend = density_map_gpu(matrix, out_size)
    return density, elapsed, backend


def max_pool_compression(matrix, out_size: int):
    coo = matrix.tocoo(copy=False)
    rows, cols = coo.shape
    started = time.perf_counter()
    rb = np.minimum((coo.row.astype(np.int64, copy=False) * out_size) // rows, out_size - 1)
    cb = np.minimum((coo.col.astype(np.int64, copy=False) * out_size) // cols, out_size - 1)
    grid = np.zeros((out_size, out_size), dtype=np.float32)
    grid[rb, cb] = 1.0
    return grid, time.perf_counter() - started, "cpu"


def nearest_downsample_compression(matrix, out_size: int):
    csr = matrix.tocsr(copy=False)
    rows, cols = csr.shape
    started = time.perf_counter()
    row_pos = np.clip(np.rint((np.arange(out_size, dtype=np.float64) + 0.5) * rows / out_size - 0.5).astype(np.int64), 0, rows - 1)
    col_pos = np.clip(np.rint((np.arange(out_size, dtype=np.float64) + 0.5) * cols / out_size - 0.5).astype(np.int64), 0, cols - 1)
    grid = np.zeros((out_size, out_size), dtype=np.float32)
    for out_r, src_r in enumerate(row_pos.tolist()):
        left = csr.indptr[src_r]
        right = csr.indptr[src_r + 1]
        if right <= left:
            continue
        cols_in_row = csr.indices[left:right]
        hits = np.isin(col_pos, cols_in_row, assume_unique=False)
        grid[out_r, hits] = 1.0
    return grid, time.perf_counter() - started, "cpu"


def gaussian_compression(matrix, out_size: int, sigma: float = 0.75, radius: int = 2):
    coo = matrix.tocoo(copy=False)
    rows, cols = coo.shape
    started = time.perf_counter()
    r_scaled = (coo.row.astype(np.float64, copy=False) + 0.5) * out_size / float(rows) - 0.5
    c_scaled = (coo.col.astype(np.float64, copy=False) + 0.5) * out_size / float(cols) - 0.5
    r0 = np.floor(r_scaled).astype(np.int64)
    c0 = np.floor(c_scaled).astype(np.int64)
    grid = np.zeros((out_size, out_size), dtype=np.float32)
    for dr in range(-radius, radius + 1):
        rr = r0 + dr
        valid_r = (rr >= 0) & (rr < out_size)
        if not np.any(valid_r):
            continue
        wr = np.exp(-0.5 * ((rr.astype(np.float64, copy=False) - r_scaled) / sigma) ** 2)
        for dc in range(-radius, radius + 1):
            cc = c0 + dc
            valid = valid_r & (cc >= 0) & (cc < out_size)
            if not np.any(valid):
                continue
            wc = np.exp(-0.5 * ((cc.astype(np.float64, copy=False) - c_scaled) / sigma) ** 2)
            w = wr * wc
            np.add.at(grid, (rr[valid], cc[valid]), w[valid].astype(np.float32, copy=False))
    total = float(grid.sum(dtype=np.float64))
    if total > 0.0:
        grid *= float(coo.nnz) / total
    return grid, time.perf_counter() - started, "cpu"


def compression_map(matrix, out_size: int, method: str):
    if method in {"density_fft", "avg_pool_fft"}:
        return avg_pool_compression(matrix, out_size)
    if method == "max_pool_fft":
        return max_pool_compression(matrix, out_size)
    if method == "nearest_downsample_fft":
        return nearest_downsample_compression(matrix, out_size)
    if method == "gaussian_compression_fft":
        return gaussian_compression(matrix, out_size)
    raise ValueError(f"unknown compression method: {method}")


def normalize_density(density, nnz: int, mode: str):
    total = density.sum()
    if float(total) <= 0.0:
        return density
    if mode == "none":
        return density
    if mode == "mass":
        return density * (float(nnz) / total)
    if mode == "unit":
        return density / total
    raise ValueError(f"unknown normalization: {mode}")


def fft_log_spectrum_dense_gpu(matrix):
    if not gpu_available():
        raise RuntimeError("CuPy is unavailable")
    coo = matrix.tocoo(copy=False)
    started = time.perf_counter()
    dense = cp.zeros(coo.shape, dtype=cp.float32)
    dense[cp.asarray(coo.row), cp.asarray(coo.col)] = 1.0
    sync()
    prep_s = time.perf_counter() - started
    fft_started = time.perf_counter()
    spectrum = cp.fft.fftshift(cp.fft.fft2(dense))
    log_mag = cp.log1p(cp.abs(spectrum).astype(cp.float32))
    sync()
    fft_s = time.perf_counter() - fft_started
    del dense, spectrum
    return log_mag.astype(cp.float32, copy=False), {"backend": "gpu", "prep_s": prep_s, "fft_s": fft_s, "total_s": time.perf_counter() - started}


def fft_log_spectrum_dense_cpu(matrix):
    started = time.perf_counter()
    coo = matrix.tocoo(copy=False)
    dense = np.zeros(coo.shape, dtype=np.float32)
    dense[coo.row, coo.col] = 1.0
    prep_s = time.perf_counter() - started
    fft_started = time.perf_counter()
    workers = max(1, int(os.environ.get("CPU_FFT_WORKERS", str(os.cpu_count() or 1))))
    spectrum = scipy_fft.fftshift(scipy_fft.fft2(dense, workers=workers))
    log_mag = np.log1p(np.abs(spectrum).astype(np.float32, copy=False)).astype(np.float32, copy=False)
    fft_s = time.perf_counter() - fft_started
    return log_mag, {"backend": "cpu", "prep_s": prep_s, "fft_s": fft_s, "total_s": time.perf_counter() - started}


def estimate_ref_row_batch(cols: int) -> int:
    per_row_bytes = cols * (4 + 8)
    return max(1, min(REF_ROW_BATCH_CAP, int(REF_GPU_WORK_BYTES // max(per_row_bytes, 1))))


def estimate_ref_col_batch(rows: int) -> int:
    per_col_bytes = rows * (8 + 8 + 4)
    return max(1, min(REF_COL_BATCH_CAP, int(REF_GPU_WORK_BYTES // max(per_col_bytes, 1))))


def build_full_fft_reference_cache(matrix_path: Path, cache_dir: Path) -> tuple[Path, dict]:
    if not gpu_available():
        raise RuntimeError("chunked full FFT reference requires CuPy")
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = bench.load_binary_coo(matrix_path).tocsr(copy=False)
    rows, cols = matrix.shape
    mag_path = cache_dir / f"{matrix_path.stem}_fullfft_shifted_mag_float32.dat"
    meta_path = cache_dir / f"{matrix_path.stem}_fullfft_shifted_meta.json"
    if mag_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("rows") == rows and meta.get("cols") == cols and meta.get("shifted") is True:
            meta["status"] = "cache_hit"
            return mag_path, meta

    row_fft_path = cache_dir / f"{matrix_path.stem}_rowfft_complex64.dat"
    row_fft = np.memmap(row_fft_path, dtype=np.complex64, mode="w+", shape=(rows, cols))
    row_batch = estimate_ref_row_batch(cols)
    row_started = time.perf_counter()
    for start in range(0, rows, row_batch):
        stop = min(start + row_batch, rows)
        dense_block = np.zeros((stop - start, cols), dtype=np.float32)
        for local_row, global_row in enumerate(range(start, stop)):
            left = matrix.indptr[global_row]
            right = matrix.indptr[global_row + 1]
            if right > left:
                dense_block[local_row, matrix.indices[left:right]] = 1.0
        dense_gpu = cp.asarray(dense_block, dtype=cp.float32)
        fft_block = cp.fft.fft(dense_gpu, axis=1).astype(cp.complex64, copy=False)
        row_fft[start:stop, :] = cp.asnumpy(fft_block)
        row_fft.flush()
        del dense_gpu, fft_block, dense_block
        clear_gpu()
        print(f"reference row FFT {matrix_path.stem}: {stop}/{rows}", flush=True)

    mag = np.memmap(mag_path, dtype=np.float32, mode="w+", shape=(rows, cols))
    row_target = (np.arange(rows, dtype=np.int64) - (rows // 2)) % rows
    col_batch = estimate_ref_col_batch(rows)
    col_started = time.perf_counter()
    for start in range(0, cols, col_batch):
        stop = min(start + col_batch, cols)
        block = np.array(row_fft[:, start:stop], dtype=np.complex64, copy=True)
        block_gpu = cp.asarray(block, dtype=cp.complex64)
        fft_block = cp.fft.fft(block_gpu, axis=0).astype(cp.complex64, copy=False)
        abs_block = cp.asnumpy(cp.abs(fft_block).astype(cp.float32))
        col_source = np.arange(start, stop, dtype=np.int64)
        col_target = (col_source - (cols // 2)) % cols
        mag[np.ix_(row_target, col_target)] = abs_block
        mag.flush()
        del block, block_gpu, fft_block, abs_block
        clear_gpu()
        print(f"reference col FFT {matrix_path.stem}: {stop}/{cols}", flush=True)

    del row_fft, mag
    row_fft_path.unlink(missing_ok=True)
    meta = {
        "rows": rows,
        "cols": cols,
        "shifted": True,
        "row_batch": row_batch,
        "col_batch": col_batch,
        "row_fft_s": time.perf_counter() - row_started,
        "col_fft_s": time.perf_counter() - col_started,
        "status": "built",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mag_path, meta


def reference_log_spectrum(matrix, matrix_path: Path, cache_dir: Path, force_cache: bool):
    rows, cols = matrix.shape
    if not force_cache and rows * cols <= DIRECT_FULL_FFT_ELEMENTS:
        try:
            if gpu_available():
                log_mag, timing = fft_log_spectrum_dense_gpu(matrix)
                timing["status"] = "ok"
                return {"kind": "array", "log": log_mag, "rows": rows, "cols": cols}, timing
        except Exception as exc:
            clear_gpu()
            gpu_error = f"gpu_error:{type(exc).__name__}"
        else:
            gpu_error = "gpu_unavailable"
        try:
            log_mag, timing = fft_log_spectrum_dense_cpu(matrix)
            timing["status"] = gpu_error + "|cpu_ok"
            return {"kind": "array", "log": log_mag, "rows": rows, "cols": cols}, timing
        except Exception as exc:
            print(f"direct full FFT failed, falling back to cache: {type(exc).__name__}: {exc}", flush=True)
            clear_gpu()
    try:
        started = time.perf_counter()
        mag_path, meta = build_full_fft_reference_cache(matrix_path, cache_dir)
        timing = dict(meta)
        timing.update({"backend": "chunked_gpu_memmap", "status": meta.get("status", "ok"), "total_s": time.perf_counter() - started, "mag_path": str(mag_path)})
        return {"kind": "memmap", "mag_path": mag_path, "rows": rows, "cols": cols}, timing
    except Exception as exc:
        return None, {"status": f"reference_error:{type(exc).__name__}", "note": str(exc)}


def to_gpu(array):
    return cp.asarray(array, dtype=cp.float32) if gpu_available() else np.asarray(array, dtype=np.float32)


def xp_of(array):
    return cp.get_array_module(array) if gpu_available() else np


def interp2(sample, rows: int, cols: int):
    xp = xp_of(sample)
    h, w = sample.shape
    if h == rows and w == cols:
        return sample
    rpos = xp.linspace(0.0, float(h - 1), rows, dtype=xp.float32) if rows > 1 else xp.zeros(1, dtype=xp.float32)
    cpos = xp.linspace(0.0, float(w - 1), cols, dtype=xp.float32) if cols > 1 else xp.zeros(1, dtype=xp.float32)
    r0 = xp.floor(rpos).astype(xp.int64)
    c0 = xp.floor(cpos).astype(xp.int64)
    r1 = xp.clip(r0 + 1, 0, h - 1)
    c1 = xp.clip(c0 + 1, 0, w - 1)
    rw = (rpos - r0.astype(xp.float32))[:, None]
    cw = (cpos - c0.astype(xp.float32))[None, :]
    top = sample[r0[:, None], c0[None, :]] * (1.0 - cw) + sample[r0[:, None], c1[None, :]] * cw
    bottom = sample[r1[:, None], c0[None, :]] * (1.0 - cw) + sample[r1[:, None], c1[None, :]] * cw
    return (top * (1.0 - rw) + bottom * rw).astype(xp.float32, copy=False)


def downsample_like(reference, out_size: int):
    return interp2(reference, out_size, out_size)


def entropy_from_log_spectrum(log_spec) -> float:
    xp = xp_of(log_spec)
    total = log_spec.sum()
    if float(total) <= 0.0:
        return 0.0
    p = log_spec / total
    p = p[p > 0]
    return float(-(p * xp.log(p)).sum()) if int(p.size) else 0.0


def radial_ratio_from_log_spectrum(log_spec, bins_count: int = RADIAL_BINS):
    xp = xp_of(log_spec)
    rows, cols = log_spec.shape
    y = xp.arange(rows, dtype=xp.float32)[:, None]
    x = xp.arange(cols, dtype=xp.float32)[None, :]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = xp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = float(radius.max()) if rows * cols else 0.0
    if r_max > 0:
        bins = xp.clip(xp.floor((radius / r_max) * bins_count).astype(xp.int32), 0, bins_count - 1)
    else:
        bins = xp.zeros((rows, cols), dtype=xp.int32)
    energy = xp.bincount(bins.ravel(), weights=log_spec.ravel(), minlength=bins_count).astype(xp.float64)
    total = energy.sum()
    ratio = energy / total if float(total) > 0.0 else energy
    return ratio


def radial_error(a, b) -> float:
    xp = xp_of(a)
    return float(xp.mean(xp.abs(a - b)))


def log_mae(reference, candidate) -> float:
    xp = xp_of(reference)
    return float(xp.mean(xp.abs(reference - candidate)))


def density_fft_log_spectrum(density):
    xp = xp_of(density)
    started = time.perf_counter()
    spectrum = xp.fft.fftshift(xp.fft.fft2(density))
    log_spec = xp.log1p(xp.abs(spectrum).astype(xp.float32))
    if gpu_available() and xp is cp:
        sync()
    return log_spec.astype(xp.float32, copy=False), time.perf_counter() - started


def normalized_density_entropy(density) -> float:
    xp = xp_of(density)
    flat = density.ravel().astype(xp.float64, copy=False)
    total = flat.sum()
    if float(total) <= 0.0 or int(flat.size) == 0:
        return 0.0
    prob = flat / total
    prob = prob[prob > 0.0]
    if int(prob.size) == 0:
        return 0.0
    entropy = float(-(prob * xp.log(prob)).sum())
    return entropy / math.log(float(flat.size)) if int(flat.size) > 1 else 0.0


def density_gini(density) -> float:
    xp = xp_of(density)
    flat = xp.clip(density.ravel().astype(xp.float64, copy=False), 0.0, None)
    if int(flat.size) == 0:
        return 0.0
    total = flat.sum()
    if float(total) <= 0.0:
        return 0.0
    sorted_flat = xp.sort(flat)
    n = int(sorted_flat.size)
    index = xp.arange(1, n + 1, dtype=xp.float64)
    return float((2.0 * xp.sum(index * sorted_flat) / (n * total)) - ((n + 1.0) / n))


def density1024_features(matrix) -> dict:
    started = time.perf_counter()
    density, build_s, backend = density_map_gpu(matrix, 1024)
    density = to_gpu(density)
    compute_started = time.perf_counter()
    features = {
        "density_1024_entropy_norm": normalized_density_entropy(density),
        "density_1024_std": float(xp_of(density).std(density.astype(xp_of(density).float64, copy=False))),
        "density_1024_gini": density_gini(density),
        "density_1024_sum": float(density.sum()),
        "density_1024_backend": backend,
        "density_1024_build_s": build_s,
        "density_1024_compute_s": time.perf_counter() - compute_started,
        "density_1024_total_s": time.perf_counter() - started,
    }
    del density
    clear_gpu()
    return features


def compute_reference_metrics(reference, resolutions: Iterable[int]) -> dict:
    if reference["kind"] == "memmap":
        return compute_reference_metrics_memmap(reference, resolutions)
    reference = reference["log"]
    out = {
        "full_entropy": entropy_from_log_spectrum(reference),
        "full_radial": radial_ratio_from_log_spectrum(reference).tolist() if not gpu_available() else cp.asnumpy(radial_ratio_from_log_spectrum(reference)).tolist(),
        "downsample": {},
    }
    for res in resolutions:
        small = downsample_like(reference, res)
        radial = radial_ratio_from_log_spectrum(small)
        out["downsample"][str(res)] = {
            "entropy": entropy_from_log_spectrum(small),
            "radial": radial.tolist() if not gpu_available() else cp.asnumpy(radial).tolist(),
        }
        del small, radial
    return out


def radial_bins_numpy(rows: int, cols: int, row_start: int, row_stop: int, col_start: int, col_stop: int, bins_count: int = RADIAL_BINS) -> np.ndarray:
    y = np.arange(row_start, row_stop, dtype=np.float32)[:, None]
    x = np.arange(col_start, col_stop, dtype=np.float32)[None, :]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = math.sqrt(max(cy * cy + cx * cx, 1.0))
    return np.clip(np.floor((radius / r_max) * bins_count).astype(np.int32), 0, bins_count - 1)


def entropy_from_sums(total: float, xlogx: float) -> float:
    if total <= 0.0:
        return 0.0
    return float(math.log(total) - (xlogx / total))


def compute_reference_metrics_memmap(reference: dict, resolutions: Iterable[int]) -> dict:
    rows, cols = reference["rows"], reference["cols"]
    mag = np.memmap(reference["mag_path"], dtype=np.float32, mode="r", shape=(rows, cols))
    total = 0.0
    xlogx = 0.0
    radial_energy = np.zeros(RADIAL_BINS, dtype=np.float64)
    for row_start in range(0, rows, METRIC_ROW_BLOCK):
        row_stop = min(row_start + METRIC_ROW_BLOCK, rows)
        for col_start in range(0, cols, METRIC_COL_BLOCK):
            col_stop = min(col_start + METRIC_COL_BLOCK, cols)
            block = np.log1p(np.asarray(mag[row_start:row_stop, col_start:col_stop], dtype=np.float32))
            total += float(block.sum(dtype=np.float64))
            positive = block[block > 0]
            if positive.size:
                xlogx += float((positive.astype(np.float64) * np.log(positive.astype(np.float64))).sum(dtype=np.float64))
            bins = radial_bins_numpy(rows, cols, row_start, row_stop, col_start, col_stop)
            radial_energy += np.bincount(bins.ravel(), weights=block.ravel(), minlength=RADIAL_BINS)
    full_radial = radial_energy / radial_energy.sum() if radial_energy.sum() > 0 else radial_energy
    out = {"full_entropy": entropy_from_sums(total, xlogx), "full_radial": full_radial.tolist(), "downsample": {}}
    for res in resolutions:
        small = downsample_reference_memmap(reference, res)
        out["downsample"][str(res)] = {"entropy": entropy_from_log_spectrum(small), "radial": radial_ratio_from_log_spectrum(small).tolist()}
    del mag
    return out


def downsample_reference_memmap(reference: dict, out_size: int) -> np.ndarray:
    rows, cols = reference["rows"], reference["cols"]
    mag = np.memmap(reference["mag_path"], dtype=np.float32, mode="r", shape=(rows, cols))
    row_pos = np.linspace(0.0, float(rows - 1), out_size, dtype=np.float64) if out_size > 1 else np.zeros(1, dtype=np.float64)
    col_pos = np.linspace(0.0, float(cols - 1), out_size, dtype=np.float64) if out_size > 1 else np.zeros(1, dtype=np.float64)
    r0 = np.floor(row_pos).astype(np.int64)
    c0 = np.floor(col_pos).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    rw = (row_pos - r0.astype(np.float64))[:, None]
    cw = (col_pos - c0.astype(np.float64))[None, :]
    c00 = np.log1p(np.asarray(mag[np.ix_(r0, c0)], dtype=np.float32))
    c01 = np.log1p(np.asarray(mag[np.ix_(r0, c1)], dtype=np.float32))
    c10 = np.log1p(np.asarray(mag[np.ix_(r1, c0)], dtype=np.float32))
    c11 = np.log1p(np.asarray(mag[np.ix_(r1, c1)], dtype=np.float32))
    top = c00 * (1.0 - cw) + c01 * cw
    bottom = c10 * (1.0 - cw) + c11 * cw
    del mag
    return (top * (1.0 - rw) + bottom * rw).astype(np.float32, copy=False)


def interp_block_numpy(sample: np.ndarray, rows: int, cols: int, row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
    h, w = sample.shape
    rpos = np.linspace(0.0, float(h - 1), rows, dtype=np.float64)[row_start:row_stop] if rows > 1 else np.zeros(row_stop - row_start)
    cpos = np.linspace(0.0, float(w - 1), cols, dtype=np.float64)[col_start:col_stop] if cols > 1 else np.zeros(col_stop - col_start)
    r0 = np.floor(rpos).astype(np.int64)
    c0 = np.floor(cpos).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    rw = (rpos - r0.astype(np.float64))[:, None]
    cw = (cpos - c0.astype(np.float64))[None, :]
    top = sample[np.ix_(r0, c0)] * (1.0 - cw) + sample[np.ix_(r0, c1)] * cw
    bottom = sample[np.ix_(r1, c0)] * (1.0 - cw) + sample[np.ix_(r1, c1)] * cw
    return (top * (1.0 - rw) + bottom * rw).astype(np.float32, copy=False)


def interpolated_metrics(reference: dict, ref_metrics: dict, small_log) -> dict:
    if reference["kind"] == "array":
        ref_log = reference["log"]
        rows, cols = ref_log.shape
        full_log = interp2(small_log, rows, cols)
        full_radial = radial_ratio_from_log_spectrum(full_log)
        ref_radial = to_gpu(np.asarray(ref_metrics["full_radial"], dtype=np.float64))
        out = {
            "log_mae_interp": log_mae(ref_log, full_log),
            "entropy_error_interp": abs(ref_metrics["full_entropy"] - entropy_from_log_spectrum(full_log)),
            "radial_error_interp": radial_error(ref_radial, full_radial),
        }
        del full_log, full_radial, ref_radial
        return out

    rows, cols = reference["rows"], reference["cols"]
    small_cpu = cp.asnumpy(small_log) if gpu_available() and cp.get_array_module(small_log) is cp else np.asarray(small_log, dtype=np.float32)
    mag = np.memmap(reference["mag_path"], dtype=np.float32, mode="r", shape=(rows, cols))
    mae_total = 0.0
    count = 0
    cand_total = 0.0
    cand_xlogx = 0.0
    cand_radial = np.zeros(RADIAL_BINS, dtype=np.float64)
    for row_start in range(0, rows, METRIC_ROW_BLOCK):
        row_stop = min(row_start + METRIC_ROW_BLOCK, rows)
        for col_start in range(0, cols, METRIC_COL_BLOCK):
            col_stop = min(col_start + METRIC_COL_BLOCK, cols)
            cand = interp_block_numpy(small_cpu, rows, cols, row_start, row_stop, col_start, col_stop)
            ref = np.log1p(np.asarray(mag[row_start:row_stop, col_start:col_stop], dtype=np.float32))
            mae_total += float(np.abs(cand - ref).sum(dtype=np.float64))
            count += cand.size
            cand_total += float(cand.sum(dtype=np.float64))
            positive = cand[cand > 0]
            if positive.size:
                cand_xlogx += float((positive.astype(np.float64) * np.log(positive.astype(np.float64))).sum(dtype=np.float64))
            bins = radial_bins_numpy(rows, cols, row_start, row_stop, col_start, col_stop)
            cand_radial += np.bincount(bins.ravel(), weights=cand.ravel(), minlength=RADIAL_BINS)
    cand_radial = cand_radial / cand_radial.sum() if cand_radial.sum() > 0 else cand_radial
    del mag
    return {
        "log_mae_interp": float(mae_total / count) if count else math.nan,
        "entropy_error_interp": abs(ref_metrics["full_entropy"] - entropy_from_sums(cand_total, cand_xlogx)),
        "radial_error_interp": float(np.mean(np.abs(np.asarray(ref_metrics["full_radial"], dtype=np.float64) - cand_radial))),
    }


def direct_metrics(reference: dict, ref_metrics: dict, small_log) -> dict:
    res = int(small_log.shape[0])
    if reference["kind"] == "array":
        ref_small = downsample_like(reference["log"], res)
    else:
        ref_small = downsample_reference_memmap(reference, res)
    ref_small = to_gpu(ref_small)
    small_radial = radial_ratio_from_log_spectrum(small_log)
    ref_small_radial = radial_ratio_from_log_spectrum(ref_small)
    out = {
        "entropy_error_direct": abs(entropy_from_log_spectrum(ref_small) - entropy_from_log_spectrum(small_log)),
        "radial_error_direct": radial_error(ref_small_radial, small_radial),
    }
    del ref_small, small_radial, ref_small_radial
    return out


def sampled_radial_error(ref_metrics: dict, features: dict) -> float:
    ref_radial = np.asarray(ref_metrics.get("full_radial", []), dtype=np.float64)
    sampled_radial = np.asarray(features.get("radial", []), dtype=np.float64)
    if ref_radial.size == 0 or sampled_radial.size == 0 or ref_radial.shape != sampled_radial.shape:
        return math.nan
    return float(np.mean(np.abs(ref_radial - sampled_radial)))


def sampled_entropy_error(ref_metrics: dict, features: dict) -> float:
    sample_count = int(features.get("sample_count") or 0)
    spectral_points = int(features.get("spectral_points") or 0)
    if sample_count <= 1 or spectral_points <= 1:
        return math.nan
    ref_entropy = float(ref_metrics.get("full_entropy", math.nan))
    sampled_entropy = float(features.get("entropy", math.nan))
    if not math.isfinite(ref_entropy) or not math.isfinite(sampled_entropy):
        return math.nan
    ref_entropy_norm = ref_entropy / math.log(float(spectral_points))
    sampled_entropy_norm = sampled_entropy / math.log(float(sample_count))
    return abs(ref_entropy_norm - sampled_entropy_norm)


def density_work_bytes(out_size: int) -> int:
    cells = int(out_size) * int(out_size)
    return cells * (4 + 8 + 8 + 4 + 4)


def evaluate_compression(method: str, matrix, reference, ref_metrics: dict, out_size: int, density_ratio: float, curve_index: int, normalizations: tuple[str, ...]) -> list[dict]:
    if density_work_bytes(out_size) > DENSITY_MAX_WORK_BYTES:
        return [{"method": method, "normalization": "all", "resolution": out_size, "density_ratio": density_ratio, "curve_index": curve_index, "status": "skipped:workset_limit", "estimated_work_bytes": density_work_bytes(out_size)}]
    try:
        density, compress_s, compress_backend = compression_map(matrix, out_size, method)
    except Exception as exc:
        clear_gpu()
        return [{"method": method, "normalization": "all", "resolution": out_size, "density_ratio": density_ratio, "curve_index": curve_index, "status": f"skipped:{type(exc).__name__}", "note": str(exc)}]
    density = to_gpu(density)
    records: list[dict] = []
    normalizations = normalizations if method == "density_fft" else COMPRESSION_BASELINE_NORMALIZATIONS
    for norm in normalizations:
        started = time.perf_counter()
        normalized = normalize_density(density, int(matrix.nnz), norm)
        small_log, fft_s = density_fft_log_spectrum(normalized)
        interp_started = time.perf_counter()
        metric_updates = {}
        if reference is not None:
            metric_updates.update(interpolated_metrics(reference, ref_metrics, small_log))
            metric_updates.update(direct_metrics(reference, ref_metrics, small_log))
        interp_s = time.perf_counter() - interp_started

        row = {
            "method": method,
            "normalization": norm,
            "resolution": out_size,
            "density_ratio": density_ratio,
            "curve_index": curve_index,
            "status": "ok",
            "compress_backend": compress_backend,
            "compress_s": compress_s,
            "fft_s": fft_s,
            "interp_s": interp_s,
            "total_s": time.perf_counter() - started + compress_s,
        }
        row.update(metric_updates)
        records.append(row)
        del normalized, small_log
        clear_gpu()
    del density
    return records


def evaluate_density(matrix, reference, ref_metrics: dict, out_size: int, density_ratio: float, curve_index: int, normalizations: tuple[str, ...]) -> list[dict]:
    records: list[dict] = []
    for method in COMPRESSION_METHODS:
        records.extend(evaluate_compression(method, matrix, reference, ref_metrics, out_size, density_ratio, curve_index, normalizations))
    return records


def grid_log_from_coeffs(points, coeffs: np.ndarray) -> np.ndarray:
    row_coords = sorted({point.u_shift for point in points})
    col_coords = sorted({point.v_shift for point in points})
    row_index = {coord: idx for idx, coord in enumerate(row_coords)}
    col_index = {coord: idx for idx, coord in enumerate(col_coords)}
    grid = np.zeros((len(row_coords), len(col_coords)), dtype=np.float32)
    mag = np.log1p(np.abs(coeffs).astype(np.float32, copy=False))
    for point, value in zip(points, mag, strict=True):
        grid[row_index[point.u_shift], col_index[point.v_shift]] = float(value)
    return grid


def finufft_grid_features(matrix, points, counts: dict, meta: dict, radial_bins: int, prefer_gpu: bool = False):
    coo = matrix.tocoo(copy=False)
    rows, cols = coo.shape
    started = time.perf_counter()
    backend = "finufft_cpu"
    if prefer_gpu:
        try:
            import cufinufft  # type: ignore
            if hasattr(cufinufft, "nufft2d3"):
                x_gpu = cp.asarray((bench.TWO_PI * coo.row.astype(np.float64, copy=False)) / float(rows))
                y_gpu = cp.asarray((bench.TWO_PI * coo.col.astype(np.float64, copy=False)) / float(cols))
                c_gpu = cp.ones(coo.nnz, dtype=cp.complex64)
                s_gpu = cp.asarray(np.array([point.u_freq for point in points], dtype=np.float64))
                t_gpu = cp.asarray(np.array([point.v_freq for point in points], dtype=np.float64))
                coeffs_gpu = cufinufft.nufft2d3(x_gpu, y_gpu, c_gpu, s_gpu, t_gpu, isign=-1, eps=1e-5)
                sync()
                coeffs = cp.asnumpy(coeffs_gpu).astype(np.complex64, copy=False)
                elapsed = time.perf_counter() - started
                result = sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)
                result["backend"] = "cufinufft_gpu"
                return result
        except ImportError:
            pass
        except Exception:
            clear_gpu()

    try:
        import finufft  # type: ignore
    except ImportError as exc:
        raise ImportError("install finufft or cufinufft to enable finufft_grid/cufinufft_grid") from exc

    x = (bench.TWO_PI * coo.row.astype(np.float64, copy=False)) / float(rows)
    y = (bench.TWO_PI * coo.col.astype(np.float64, copy=False)) / float(cols)
    c = np.ones(coo.nnz, dtype=np.complex128)
    s = np.array([point.u_freq for point in points], dtype=np.float64)
    t = np.array([point.v_freq for point in points], dtype=np.float64)
    coeffs = finufft.nufft2d3(x, y, c, s, t, isign=-1, eps=1e-9).astype(np.complex64, copy=False)
    elapsed = time.perf_counter() - started
    result = sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)
    result["backend"] = backend
    return result


def sparse_grid_features(method: str, matrix, points, counts, meta, batch_size: int, threads: int, spfft_library: str | None):
    if method == "spfft_grid":
        try:
            return spfft_method.spfft_jl_features(matrix, points, counts, meta, RADIAL_BINS), "gpu_pruned"
        except Exception:
            clear_gpu()
            return spfft_method.spfft_jl_features_cpu(matrix, points, counts, meta, RADIAL_BINS), "cpu_pruned"
    if method == "sparse_direct_grid":
        return sparse_sampling.sampled_sparse_fft_features_from_points(matrix, points, counts, meta, RADIAL_BINS, batch_size), "gpu_direct"
    if method == "finufft_grid":
        features = finufft_grid_features(matrix, points, counts, meta, RADIAL_BINS, prefer_gpu=False)
        return features, features.get("backend", "finufft_cpu")
    if method == "cufinufft_grid":
        features = finufft_grid_features(matrix, points, counts, meta, RADIAL_BINS, prefer_gpu=True)
        return features, features.get("backend", "finufft_cpu")
    if method in {"fps_sft", "kapralov_sfft"}:
        raise ImportError(f"{method} external adapter is not configured in this environment")
    raise ValueError(method)


def evaluate_sparse_grid(method: str, matrix, reference, ref_metrics: dict, sample_fraction: float, batch_size: int, threads: int, spfft_library: str | None, curve_index: int) -> dict:
    rows, cols = matrix.shape
    original_fraction = os.environ.get("SPARSE_GRID_FRACTION")
    os.environ["SPARSE_GRID_FRACTION"] = str(sample_fraction)
    started = time.perf_counter()
    row_count = max(2, int(math.floor(rows * sample_fraction)))
    col_count = max(2, int(math.floor(cols * sample_fraction)))
    row_coords = sparse_sampling.evenly_spaced_shifted_coords(rows, row_count)
    col_coords = sparse_sampling.evenly_spaced_shifted_coords(cols, col_count)
    points = []
    counts = {"total": rows * cols, "core": 0, "axis": 0, "diag": 0, "rest": 0}
    weight = (rows * cols) / float(len(row_coords) * len(col_coords))
    for u_shift in row_coords:
        for v_shift in col_coords:
            region = bench.classify_sample_region(u_shift, v_shift, rows, cols, 0.12, 3, 3)
            counts[region] += 1
            points.append(bench.SamplePoint(u_shift, v_shift, bench.shifted_to_fft_index(u_shift, rows), bench.shifted_to_fft_index(v_shift, cols), 1, region, weight))
    meta = {"u_bounds": bench.shifted_coord_bounds(rows), "v_bounds": bench.shifted_coord_bounds(cols), "sampled_rows": len(row_coords), "sampled_cols": len(col_coords)}
    sample_s = time.perf_counter() - started
    try:
        features, backend = sparse_grid_features(method, matrix, points, counts, meta, batch_size, threads, spfft_library)
        grid = to_gpu(grid_log_from_coeffs(features["points"], features["coeffs"]))
        interp_started = time.perf_counter()
        metric_updates = {}
        if reference is not None:
            metric_updates.update(interpolated_metrics(reference, ref_metrics, grid))
            metric_updates.update(direct_metrics(reference, ref_metrics, grid))
            metric_updates["entropy_error_direct"] = sampled_entropy_error(ref_metrics, features)
            metric_updates["radial_error_direct"] = sampled_radial_error(ref_metrics, features)
        interp_s = time.perf_counter() - interp_started
        row = {
            "method": method,
            "normalization": "na",
            "resolution": int(grid.shape[0]),
            "status": "ok",
            "curve_index": curve_index,
            "sample_fraction": sample_fraction,
            "sample_axis_rows": len(row_coords),
            "sample_axis_cols": len(col_coords),
            "sample_count": len(points),
            "backend": backend,
            "sample_s": sample_s,
            "compute_s": float(features["elapsed"]),
            "interp_s": interp_s,
            "total_s": sample_s + float(features["elapsed"]) + interp_s,
        }
        row.update(metric_updates)
        del grid
        return row
    except ImportError as exc:
        clear_gpu()
        return {"method": method, "normalization": "na", "resolution": row_count, "status": "skipped:dependency_missing", "note": str(exc), "curve_index": curve_index, "sample_fraction": sample_fraction, "sample_axis_rows": len(row_coords), "sample_axis_cols": len(col_coords), "sample_count": len(points), "sample_s": sample_s}
    except Exception as exc:
        clear_gpu()
        return {"method": method, "normalization": "na", "resolution": row_count, "status": f"error:{type(exc).__name__}", "note": str(exc), "curve_index": curve_index, "sample_fraction": sample_fraction, "sample_axis_rows": len(row_coords), "sample_axis_cols": len(col_coords), "sample_count": len(points), "sample_s": sample_s}
    finally:
        if original_fraction is None:
            os.environ.pop("SPARSE_GRID_FRACTION", None)
        else:
            os.environ["SPARSE_GRID_FRACTION"] = original_fraction


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one HPC density FFT normalization benchmark")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--matrix-index", type=str, help="Integer index into manifest, or 'all' to iterate every valid row")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--skip-existing", action="store_true", help="When --matrix-index=all, skip matrices whose output JSON already exists")
    parser.add_argument("--split", default="manual")
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--resolutions", default="", help="Legacy fixed density sizes. If set, these are run in addition to --density-ratios")
    parser.add_argument("--density-ratios", default=DENSITY_RATIOS_DEFAULT, help="Comma-separated density map ratios scanned from large to small; default max is 0.5")
    parser.add_argument("--normalization", choices=("all", *NORMALIZATIONS), default="all", help="Density FFT normalization to run; default runs all normalizations")
    parser.add_argument("--sparse-methods", default="spfft_grid,sparse_direct_grid")
    parser.add_argument("--sample-fraction", type=float, default=0.01, help="Legacy single sparse grid axis fraction, used only if --sample-fractions is empty")
    parser.add_argument("--sample-fractions", default="0.00015625,0.0003125,0.000625,0.00125,0.0025,0.005,0.01", help="Comma-separated sparse grid axis fractions for error-vs-time curves")
    parser.add_argument("--sparse-batch-size", type=int, default=64)
    parser.add_argument("--spfft-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--spfft-library", default=os.environ.get("SPFFT_LIBRARY_PATH"))
    parser.add_argument("--full-reference-cache", type=Path, default=Path("full_fft_reference_cache"))
    parser.add_argument("--force-reference-cache", action="store_true", help="Force chunked memmap full FFT reference even for small matrices")
    parser.add_argument("--sparse-fft-root", type=Path, default=DEFAULT_SPARSE_ROOT)
    return parser.parse_args()


def run_one_matrix(matrix_path: Path, split: str, args: argparse.Namespace) -> Path:
    matrix = bench.load_binary_coo(matrix_path)
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"expected square matrix, got {rows}x{cols}: {matrix_path}")

    legacy_resolutions = parse_int_list(args.resolutions) if args.resolutions.strip() else []
    density_ratios = parse_float_list(args.density_ratios)
    density_candidates = []
    seen_sizes = set()
    for idx, ratio in enumerate(density_ratios):
        out_size = max(1, int(round(min(rows, cols) * ratio)))
        if out_size in seen_sizes:
            continue
        seen_sizes.add(out_size)
        density_candidates.append((idx, ratio, out_size))
    for res in legacy_resolutions:
        if res not in seen_sizes:
            density_candidates.append((len(density_candidates), float(res) / float(min(rows, cols)), res))
    sample_fractions = parse_float_list(args.sample_fractions) if args.sample_fractions.strip() else [args.sample_fraction]
    sample_fractions.sort(reverse=True)  # 从大到小跑，如果大的 OOM 报错，后面小的还可以继续尝试
    output = {
        "matrix": matrix_path.stem,
        "path": str(matrix_path),
        "split": split,
        "rows": int(rows),
        "cols": int(cols),
        "nnz": int(matrix.nnz),
        "density_ratios": density_ratios,
        "density_candidates": [{"curve_index": idx, "density_ratio": ratio, "resolution": out_size} for idx, ratio, out_size in density_candidates],
        "sample_fractions": sample_fractions,
        "records": [],
    }

    try:
        output["hybrid_features"] = density1024_features(matrix)
    except Exception as exc:
        output["hybrid_features"] = {"status": f"error:{type(exc).__name__}", "note": str(exc)}
        clear_gpu()

    reference, ref_timing = reference_log_spectrum(matrix, matrix_path, args.full_reference_cache, args.force_reference_cache)
    output["reference"] = ref_timing
    if reference is not None:
        ref_metrics = compute_reference_metrics(reference, [])
    else:
        ref_metrics = {"full_entropy": math.nan, "full_radial": [math.nan] * RADIAL_BINS, "downsample": {}}

    for curve_index, density_ratio, out_size in density_candidates:
        try:
            norms_to_run = NORMALIZATIONS if args.normalization == "all" else (args.normalization,)
            output["records"].extend(evaluate_density(matrix, reference, ref_metrics, out_size, density_ratio, curve_index, norms_to_run))
        except Exception as exc:
            output["records"].append({"method": "density_fft", "normalization": "all", "resolution": out_size, "density_ratio": density_ratio, "curve_index": curve_index, "status": f"error:{type(exc).__name__}", "note": str(exc)})
            clear_gpu()

    for method in [item.strip() for item in args.sparse_methods.split(",") if item.strip()]:
        for curve_index, sample_fraction in enumerate(sample_fractions):
            output["records"].append(evaluate_sparse_grid(method, matrix, reference, ref_metrics, sample_fraction, args.sparse_batch_size, args.spfft_threads, args.spfft_library, curve_index))

    if reference is not None:
        del reference
    clear_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{matrix_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, default=json_ready)
    print(f"Wrote {out_path}", flush=True)
    return out_path


def main() -> int:
    args = parse_args()
    add_sparse_fft_root(args.sparse_fft_root)

    if args.matrix is not None:
        run_one_matrix(args.matrix, args.split, args)
        return 0

    if args.manifest is None or args.matrix_index is None:
        raise SystemExit("Provide --matrix or both --manifest and --matrix-index")

    if str(args.matrix_index).lower() == "all":
        rows = load_manifest_all(args.manifest)
        total = len(rows)
        if total == 0:
            raise SystemExit(f"manifest {args.manifest} has no usable rows")
        print(f"[all] running {total} matrices from {args.manifest}", flush=True)
        failures: list[tuple[int, str, str]] = []
        for idx, row in enumerate(rows):
            matrix_path = Path(row["path"])
            split = row.get("split") or args.split
            out_path = args.output_dir / f"{matrix_path.stem}.json"
            if args.skip_existing and out_path.exists():
                print(f"[{idx + 1}/{total}] skip {matrix_path.stem} (output exists)", flush=True)
                continue
            print(f"[{idx + 1}/{total}] {matrix_path.stem} ({matrix_path})", flush=True)
            try:
                run_one_matrix(matrix_path, split, args)
            except Exception as exc:
                clear_gpu()
                failures.append((idx, matrix_path.stem, f"{type(exc).__name__}: {exc}"))
                print(f"[{idx + 1}/{total}] FAILED {matrix_path.stem}: {type(exc).__name__}: {exc}", flush=True)
        if failures:
            print(f"\n[all] done with {len(failures)} failures:", flush=True)
            for idx, name, msg in failures:
                print(f"  index={idx} matrix={name} error={msg}", flush=True)
            return 1
        print(f"\n[all] done — all {total} matrices succeeded", flush=True)
        return 0

    try:
        index = int(args.matrix_index)
    except ValueError:
        raise SystemExit(f"--matrix-index must be an integer or 'all', got {args.matrix_index!r}")
    manifest_row = load_manifest_row(args.manifest, index)
    matrix_path = Path(manifest_row["path"])
    split = manifest_row.get("split") or args.split
    run_one_matrix(matrix_path, split, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
