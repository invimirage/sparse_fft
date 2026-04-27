import ctypes
import math
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from scipy import fft as scipy_fft

import approx_fft_benchmark as sparse_bench
from sparse_methods import density as density_method
from sparse_methods import sparse_sampling


SPFFT_PU_HOST = 1
SPFFT_TRANS_C2C = 0
SPFFT_INDEX_TRIPLETS = 0
SPFFT_NO_SCALING = 0
SPFFT_INPUT_MAX_BYTES = int(os.environ.get("SPFFT_INPUT_MAX_BYTES", str(6 * 1024 ** 3)))
PRUNED_GPU_MAX_BYTES = int(os.environ.get("SPFFT_PRUNED_GPU_MAX_BYTES", str(1024 ** 3)))


def binary_dense_cpu(matrix, max_bytes: int | None = None) -> np.ndarray:
    matrix = matrix.tocoo(copy=False)
    required_bytes = matrix.shape[0] * matrix.shape[1] * np.dtype(np.float32).itemsize
    if max_bytes is not None and required_bytes > max_bytes:
        raise MemoryError(f"dense float32 input requires {required_bytes} bytes")
    dense = np.zeros(matrix.shape, dtype=np.float32)
    dense[matrix.row, matrix.col] = 1.0
    return dense


class SpFFTWrapper:
    def __init__(self, library_path: str | None = None):
        self.library_path = library_path or os.environ.get("SPFFT_LIBRARY_PATH") or str(Path(sys.prefix) / "lib" / "libspfft.so")
        self.lib = ctypes.CDLL(self.library_path)

        self.lib.spfft_float_transform_create_independent.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.spfft_float_transform_create_independent.restype = ctypes.c_int
        self.lib.spfft_float_transform_forward_ptr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.spfft_float_transform_forward_ptr.restype = ctypes.c_int
        self.lib.spfft_float_transform_destroy.argtypes = [ctypes.c_void_p]
        self.lib.spfft_float_transform_destroy.restype = ctypes.c_int

    def _check(self, err: int, where: str) -> None:
        if err != 0:
            raise RuntimeError(f"SpFFT {where} failed with error code {err}")

    def forward_sparse_output(self, dense_real: np.ndarray, shifted_points: list[tuple[int, int]], threads: int) -> np.ndarray:
        rows, cols = dense_real.shape
        indices = np.zeros((len(shifted_points), 3), dtype=np.int32)
        if shifted_points:
            indices[:, 0] = np.asarray([u for (u, _) in shifted_points], dtype=np.int32)
            indices[:, 1] = np.asarray([v for (_, v) in shifted_points], dtype=np.int32)

        transform = ctypes.c_void_p()
        self._check(
            self.lib.spfft_float_transform_create_independent(
                ctypes.byref(transform),
                int(threads),
                SPFFT_PU_HOST,
                SPFFT_TRANS_C2C,
                int(rows),
                int(cols),
                1,
                int(len(shifted_points)),
                SPFFT_INDEX_TRIPLETS,
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ),
            "transform_create",
        )

        try:
            dense_complex = np.zeros(rows * cols * 2, dtype=np.float32)
            dense_complex[0::2] = dense_real.ravel(order="C")
            output = np.zeros(len(shifted_points) * 2, dtype=np.float32)
            self._check(
                self.lib.spfft_float_transform_forward_ptr(
                    transform,
                    dense_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    SPFFT_NO_SCALING,
                ),
                "forward",
            )
        finally:
            self._check(self.lib.spfft_float_transform_destroy(transform), "transform_destroy")

        return output[0::2].astype(np.float32) + 1j * output[1::2].astype(np.float32)


def spfft_blkdiv(n: int, k: int) -> tuple[int, int]:
    l = max(1, min(int(k), int(n)))
    while n % l != 0:
        l -= 1
    return l, n // l


def _fft_blocks_gpu(blocks: np.ndarray) -> np.ndarray:
    required_bytes = blocks.nbytes * 3
    if required_bytes > PRUNED_GPU_MAX_BYTES:
        raise MemoryError(f"pruned gpu block workset requires {required_bytes} bytes")
    blocks_gpu = None
    fft_gpu = None
    try:
        blocks_gpu = cp.asarray(blocks, dtype=cp.complex64)
        fft_gpu = cp.fft.fft(blocks_gpu, axis=1).astype(cp.complex64, copy=False)
        return cp.asnumpy(fft_gpu)
    except cp.cuda.memory.OutOfMemoryError as exc:
        raise MemoryError(str(exc)) from exc
    finally:
        if blocks_gpu is not None:
            del blocks_gpu
        if fft_gpu is not None:
            del fft_gpu
        sparse_bench.clear_gpu_memory()


def _fft_blocks_cpu(blocks: np.ndarray) -> np.ndarray:
    return scipy_fft.fft(blocks, axis=1).astype(np.complex64, copy=False)


def _phase_sum_gpu(bucket_values: np.ndarray, bucket_ids: np.ndarray, row_terms: np.ndarray, col_terms: np.ndarray, m: int, n: int) -> np.ndarray:
    required_bytes = bucket_values.nbytes + bucket_ids.nbytes + row_terms.nbytes + col_terms.nbytes + (bucket_values.shape[0] * bucket_values.shape[1] * 8)
    if required_bytes > PRUNED_GPU_MAX_BYTES:
        raise MemoryError(f"pruned gpu phase workset requires {required_bytes} bytes")
    bucket_values_gpu = None
    bucket_ids_gpu = None
    row_terms_gpu = None
    col_terms_gpu = None
    phases_gpu = None
    out_gpu = None
    try:
        bucket_values_gpu = cp.asarray(bucket_values, dtype=cp.complex64)
        bucket_ids_gpu = cp.asarray(bucket_ids.astype(np.float32, copy=False))[:, None]
        row_terms_gpu = cp.asarray(row_terms.astype(np.float32, copy=False))[None, :]
        col_terms_gpu = cp.asarray(col_terms.astype(np.float32, copy=False))[None, :]
        phases_gpu = cp.exp((-1j * sparse_bench.TWO_PI) * (bucket_ids_gpu * ((row_terms_gpu / float(m)) + (col_terms_gpu / float(n)))))
        out_gpu = cp.sum(bucket_values_gpu * phases_gpu.astype(cp.complex64), axis=0)
        return cp.asnumpy(out_gpu).astype(np.complex64, copy=False)
    except cp.cuda.memory.OutOfMemoryError as exc:
        raise MemoryError(str(exc)) from exc
    finally:
        del bucket_values_gpu
        del bucket_ids_gpu
        del row_terms_gpu
        del col_terms_gpu
        del phases_gpu
        del out_gpu
        sparse_bench.clear_gpu_memory()


def _phase_sum_cpu(bucket_values: np.ndarray, bucket_ids: np.ndarray, row_terms: np.ndarray, col_terms: np.ndarray, m: int, n: int) -> np.ndarray:
    phases = np.exp(
        (-1j * sparse_bench.TWO_PI)
        * (bucket_ids.astype(np.float32, copy=False)[:, None] * ((row_terms.astype(np.float32, copy=False)[None, :] / float(m)) + (col_terms.astype(np.float32, copy=False)[None, :] / float(n))))
    ).astype(np.complex64, copy=False)
    return np.sum(bucket_values * phases, axis=0, dtype=np.complex64)


def _pruned_sparse_fft_1d(positions: np.ndarray, values: np.ndarray, n: int, target_freqs: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    if positions.size == 0:
        return np.zeros(len(target_freqs), dtype=np.complex64)
    k = len(target_freqs)
    l, m = spfft_blkdiv(n, k)
    bucket_ids = (positions // l).astype(np.int32, copy=False)
    local_pos = (positions % l).astype(np.int32, copy=False)
    unique_buckets, inverse = np.unique(bucket_ids, return_inverse=True)
    blocks = np.zeros((len(unique_buckets), l), dtype=np.complex64)
    np.add.at(blocks, (inverse, local_pos), values.astype(np.complex64, copy=False))

    try:
        blocks_fft = _fft_blocks_gpu(blocks) if use_gpu else _fft_blocks_cpu(blocks)
    except MemoryError:
        blocks_fft = _fft_blocks_cpu(blocks)

    row_terms = (target_freqs // l).astype(np.int32, copy=False)
    col_terms = (target_freqs % l).astype(np.int32, copy=False)
    bucket_values = blocks_fft[:, col_terms]
    try:
        return _phase_sum_gpu(bucket_values, unique_buckets, row_terms, col_terms, m, n) if use_gpu else _phase_sum_cpu(bucket_values, unique_buckets, row_terms, col_terms, m, n)
    except MemoryError:
        return _phase_sum_cpu(bucket_values, unique_buckets, row_terms, col_terms, m, n)


def _pruned_sparse_fft_2d_from_sparse(matrix, u_freqs: np.ndarray, v_freqs: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    matrix_csc = matrix.tocsc(copy=False)
    rows, cols = matrix_csc.shape
    first_stage = np.zeros((len(u_freqs), cols), dtype=np.complex64)
    nonempty_cols = np.flatnonzero(np.diff(matrix_csc.indptr) > 0)
    for col in nonempty_cols.tolist():
        start = matrix_csc.indptr[col]
        stop = matrix_csc.indptr[col + 1]
        row_pos = matrix_csc.indices[start:stop].astype(np.int32, copy=False)
        values = matrix_csc.data[start:stop].astype(np.complex64, copy=False)
        first_stage[:, col] = _pruned_sparse_fft_1d(row_pos, values, rows, u_freqs, use_gpu=use_gpu)

    second_stage = np.zeros((len(u_freqs), len(v_freqs)), dtype=np.complex64)
    active_cols = nonempty_cols.astype(np.int32, copy=False)
    for row_idx in range(len(u_freqs)):
        second_stage[row_idx, :] = _pruned_sparse_fft_1d(active_cols, first_stage[row_idx, active_cols], cols, v_freqs, use_gpu=use_gpu)
    return second_stage


def spfft_jl_features(matrix, points, counts: dict, meta: dict, radial_bins: int) -> dict:
    rows, cols = matrix.shape
    u_freqs_sorted = sorted({p.u_freq for p in points})
    v_freqs_sorted = sorted({p.v_freq for p in points})
    u_arr = np.array(u_freqs_sorted, dtype=np.intp)
    v_arr = np.array(v_freqs_sorted, dtype=np.intp)
    u_idx_map = {f: i for i, f in enumerate(u_freqs_sorted)}
    v_idx_map = {f: i for i, f in enumerate(v_freqs_sorted)}

    started = time.perf_counter()
    try:
        result_grid = _pruned_sparse_fft_2d_from_sparse(matrix, u_arr, v_arr, use_gpu=True)
        coeffs = np.array(
            [result_grid[u_idx_map[p.u_freq], v_idx_map[p.v_freq]] for p in points],
            dtype=np.complex64,
        )
        elapsed = time.perf_counter() - started
        return sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)
    except MemoryError:
        raise
    finally:
        sparse_bench.clear_gpu_memory()


def spfft_jl_features_cpu(matrix, points, counts: dict, meta: dict, radial_bins: int) -> dict:
    u_freqs_sorted = sorted({p.u_freq for p in points})
    v_freqs_sorted = sorted({p.v_freq for p in points})
    u_idx_map = {f: i for i, f in enumerate(u_freqs_sorted)}
    v_idx_map = {f: i for i, f in enumerate(v_freqs_sorted)}

    started = time.perf_counter()
    result_grid = _pruned_sparse_fft_2d_from_sparse(matrix, np.asarray(u_freqs_sorted, dtype=np.intp), np.asarray(v_freqs_sorted, dtype=np.intp), use_gpu=False)
    coeffs = np.array(
        [result_grid[u_idx_map[p.u_freq], v_idx_map[p.v_freq]] for p in points],
        dtype=np.complex64,
    )
    elapsed = time.perf_counter() - started
    return sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)


def spfft_sampled_fft_features(matrix, points, counts: dict, meta: dict, radial_bins: int, spfft_threads: int, spfft_library: str | None):
    matrix = matrix.tocoo(copy=False)
    started = time.perf_counter()
    dense = binary_dense_cpu(matrix, max_bytes=SPFFT_INPUT_MAX_BYTES)
    coeffs = SpFFTWrapper(spfft_library).forward_sparse_output(dense, [(point.u_shift, point.v_shift) for point in points], spfft_threads)
    elapsed = time.perf_counter() - started
    return sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)


def full_spectrum_shifted(matrix) -> np.ndarray:
    matrix = matrix.tocoo(copy=False)
    try:
        dense_gpu = cp.zeros(matrix.shape, dtype=cp.float32)
        dense_gpu[cp.asarray(matrix.row), cp.asarray(matrix.col)] = 1.0
        spectrum = cp.fft.fftshift(cp.fft.fft2(dense_gpu))
        out = cp.asnumpy(spectrum)
    except cp.cuda.memory.OutOfMemoryError:
        sparse_bench.clear_gpu_memory()
        dense = binary_dense_cpu(matrix)
        out = scipy_fft.fftshift(scipy_fft.fft2(dense)).astype(np.complex64, copy=False)
    finally:
        if "dense_gpu" in locals():
            del dense_gpu
        if "spectrum" in locals():
            del spectrum
        sparse_bench.clear_gpu_memory()
    return out.astype(np.complex64, copy=False)


def _scaled_shift_coords(src_size: int, dst_size: int) -> np.ndarray:
    src_lo, _ = sparse_bench.shifted_coord_bounds(src_size)
    dst_lo, dst_hi = sparse_bench.shifted_coord_bounds(dst_size)
    src_coords = src_lo + np.arange(src_size, dtype=np.int32)
    scaled = np.rint(src_coords.astype(np.float64) * (float(dst_size) / float(src_size))).astype(np.int64)
    return np.clip(scaled, dst_lo, dst_hi).astype(np.int32)


def deterministic_topk_sample_points(matrix, total_samples: int, coarse_size: int = 512, mode: str = "uniform"):
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    total_samples = max(1, int(total_samples))
    coarse_dim = max(32, min(int(coarse_size), rows, cols))
    mag = density_method.coarse_spectrum_magnitude(matrix, coarse_dim, mode=mode)
    flat = mag.ravel()
    candidate_count = min(flat.size, max(total_samples * 16, 4096))
    row_map = _scaled_shift_coords(coarse_dim, rows)
    col_map = _scaled_shift_coords(coarse_dim, cols)

    selected: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    while len(selected) < total_samples:
        if candidate_count >= flat.size:
            candidate_idx = np.argsort(flat)[::-1]
        else:
            candidate_idx = np.argpartition(flat, -candidate_count)[-candidate_count:]
            candidate_idx = candidate_idx[np.argsort(flat[candidate_idx])[::-1]]
        for idx in candidate_idx.tolist():
            r_small, c_small = divmod(idx, coarse_dim)
            coord = (int(row_map[r_small]), int(col_map[c_small]))
            if coord in used:
                continue
            used.add(coord)
            selected.append(coord)
            if len(selected) == total_samples:
                break
        if len(selected) == total_samples or candidate_count >= flat.size:
            break
        candidate_count = min(flat.size, candidate_count * 2)

    if len(selected) < total_samples:
        side = max(1, int(math.ceil(math.sqrt(total_samples))))
        row_coords = sparse_sampling.evenly_spaced_shifted_coords(rows, side)
        col_coords = sparse_sampling.evenly_spaced_shifted_coords(cols, side)
        for u_shift in row_coords:
            for v_shift in col_coords:
                coord = (u_shift, v_shift)
                if coord in used:
                    continue
                used.add(coord)
                selected.append(coord)
                if len(selected) == total_samples:
                    break
            if len(selected) == total_samples:
                break

    counts = {"total": rows * cols, "core": 0, "axis": 0, "diag": 0, "rest": 0}
    total_weight = (rows * cols) / float(len(selected)) if selected else 0.0
    points = []
    for u_shift, v_shift in selected:
        region = sparse_bench.classify_sample_region(u_shift, v_shift, rows, cols, 0.12, 3, 3)
        counts[region] += 1
        points.append(
            sparse_bench.SamplePoint(
                u_shift=u_shift,
                v_shift=v_shift,
                u_freq=sparse_bench.shifted_to_fft_index(u_shift, rows),
                v_freq=sparse_bench.shifted_to_fft_index(v_shift, cols),
                multiplicity=1,
                stratum=region,
                weight=total_weight,
            )
        )
    meta = {
        "u_bounds": sparse_bench.shifted_coord_bounds(rows),
        "v_bounds": sparse_bench.shifted_coord_bounds(cols),
        "sampled_rows": len({p.u_shift for p in points}),
        "sampled_cols": len({p.v_shift for p in points}),
        "selection": "deterministic_topk",
        "coarse_size": coarse_dim,
    }
    return points, counts, meta


def deterministic_topk_fft_features(matrix, total_samples: int, radial_bins: int, batch_size: int, spfft_threads: int, spfft_library: str | None, coarse_size: int = 512):
    points, counts, meta = deterministic_topk_sample_points(matrix, total_samples, coarse_size=coarse_size)
    try:
        result = spfft_sampled_fft_features(matrix, points, counts, meta, radial_bins, spfft_threads, spfft_library)
        result["selection"] = "deterministic_topk"
        return result
    except Exception:
        result = sparse_sampling.sampled_sparse_fft_features_from_points(matrix, points, counts, meta, radial_bins, batch_size)
        result["selection"] = "deterministic_topk"
        return result


def oracle_topk_fft_features(matrix, total_samples: int, radial_bins: int) -> dict:
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    started = time.perf_counter()
    spectrum = full_spectrum_shifted(matrix)
    magnitudes = np.abs(spectrum).astype(np.float32, copy=False)
    flat = magnitudes.ravel()
    total_samples = max(1, min(int(total_samples), flat.size))
    if total_samples >= flat.size:
        top_idx = np.argsort(flat)[::-1]
    else:
        top_idx = np.argpartition(flat, -total_samples)[-total_samples:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    row_lo, _ = sparse_bench.shifted_coord_bounds(rows)
    col_lo, _ = sparse_bench.shifted_coord_bounds(cols)
    counts = {"total": rows * cols, "core": 0, "axis": 0, "diag": 0, "rest": 0}
    weight = (rows * cols) / float(len(top_idx)) if len(top_idx) else 0.0
    points = []
    coeffs = np.empty(len(top_idx), dtype=np.complex64)
    for out_idx, flat_idx in enumerate(top_idx.tolist()):
        r, c = divmod(flat_idx, cols)
        u_shift = row_lo + r
        v_shift = col_lo + c
        region = sparse_bench.classify_sample_region(u_shift, v_shift, rows, cols, 0.12, 3, 3)
        counts[region] += 1
        points.append(
            sparse_bench.SamplePoint(
                u_shift=u_shift,
                v_shift=v_shift,
                u_freq=sparse_bench.shifted_to_fft_index(u_shift, rows),
                v_freq=sparse_bench.shifted_to_fft_index(v_shift, cols),
                multiplicity=1,
                stratum=region,
                weight=weight,
            )
        )
        coeffs[out_idx] = spectrum[r, c]
    elapsed = time.perf_counter() - started
    meta = {
        "u_bounds": sparse_bench.shifted_coord_bounds(rows),
        "v_bounds": sparse_bench.shifted_coord_bounds(cols),
        "sampled_rows": len({p.u_shift for p in points}),
        "sampled_cols": len({p.v_shift for p in points}),
        "selection": "oracle_topk",
    }
    result = sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)
    result["selection"] = "oracle_topk"
    return result
