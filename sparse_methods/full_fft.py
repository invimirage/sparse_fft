import os
import time

import cupy as cp
import numpy as np
from scipy import fft as scipy_fft

import approx_fft_benchmark as sparse_bench


def dense_binary_cpu(matrix) -> np.ndarray:
    matrix = matrix.tocoo(copy=False)
    dense = np.zeros(matrix.shape, dtype=np.float32)
    dense[matrix.row, matrix.col] = 1.0
    return dense


def direct_gpu_fft_magnitude(matrix) -> tuple[np.ndarray, dict[str, float]]:
    matrix = matrix.tocoo(copy=False)
    sparse_bench.clear_gpu_memory()
    started = time.perf_counter()
    prep_started = time.perf_counter()
    dense_gpu = cp.zeros(matrix.shape, dtype=cp.float32)
    dense_gpu[cp.asarray(matrix.row), cp.asarray(matrix.col)] = 1.0
    cp.cuda.Stream.null.synchronize()
    prep_s = time.perf_counter() - prep_started
    fft_started = time.perf_counter()
    spectrum = cp.fft.fftshift(cp.fft.fft2(dense_gpu))
    mag_gpu = cp.abs(spectrum).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()
    fft_s = time.perf_counter() - fft_started
    transfer_started = time.perf_counter()
    mag = cp.asnumpy(mag_gpu)
    transfer_s = time.perf_counter() - transfer_started
    total_s = time.perf_counter() - started
    del dense_gpu
    del spectrum
    del mag_gpu
    sparse_bench.clear_gpu_memory()
    return mag, {
        "prep_s": prep_s,
        "fft_s": fft_s,
        "transfer_s": transfer_s,
        "total_s": total_s,
    }


def separable_gpu_fft_magnitude(matrix) -> tuple[np.ndarray, dict[str, float]]:
    matrix = matrix.tocoo(copy=False)
    sparse_bench.clear_gpu_memory()
    started = time.perf_counter()
    prep_started = time.perf_counter()
    dense_gpu = cp.zeros(matrix.shape, dtype=cp.complex64)
    dense_gpu[cp.asarray(matrix.row), cp.asarray(matrix.col)] = 1.0 + 0.0j
    cp.cuda.Stream.null.synchronize()
    prep_s = time.perf_counter() - prep_started

    row_started = time.perf_counter()
    row_fft = cp.fft.fft(dense_gpu, axis=1)
    cp.cuda.Stream.null.synchronize()
    row_fft_s = time.perf_counter() - row_started
    del dense_gpu

    transpose1_started = time.perf_counter()
    transposed = cp.ascontiguousarray(row_fft.T)
    cp.cuda.Stream.null.synchronize()
    transpose1_s = time.perf_counter() - transpose1_started
    del row_fft

    col_started = time.perf_counter()
    col_fft_t = cp.fft.fft(transposed, axis=1)
    cp.cuda.Stream.null.synchronize()
    col_fft_s = time.perf_counter() - col_started
    del transposed

    transpose2_started = time.perf_counter()
    spectrum = cp.ascontiguousarray(col_fft_t.T)
    cp.cuda.Stream.null.synchronize()
    transpose2_s = time.perf_counter() - transpose2_started
    del col_fft_t

    shift_started = time.perf_counter()
    mag_gpu = cp.abs(cp.fft.fftshift(spectrum)).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()
    shift_s = time.perf_counter() - shift_started
    del spectrum

    transfer_started = time.perf_counter()
    mag = cp.asnumpy(mag_gpu)
    transfer_s = time.perf_counter() - transfer_started
    total_s = time.perf_counter() - started
    del mag_gpu
    sparse_bench.clear_gpu_memory()
    return mag, {
        "prep_s": prep_s,
        "row_fft_s": row_fft_s,
        "transpose1_s": transpose1_s,
        "col_fft_s": col_fft_s,
        "transpose2_s": transpose2_s,
        "shift_s": shift_s,
        "transfer_s": transfer_s,
        "total_s": total_s,
    }


def cpu_parallel_fft_magnitude(matrix) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    prep_started = time.perf_counter()
    dense = dense_binary_cpu(matrix)
    prep_s = time.perf_counter() - prep_started
    fft_started = time.perf_counter()
    workers = max(1, int(os.environ.get("CPU_FFT_WORKERS", str(os.cpu_count() or 1))))
    spectrum = scipy_fft.fftshift(scipy_fft.fft2(dense, workers=workers))
    mag = np.abs(spectrum).astype(np.float32, copy=False)
    fft_s = time.perf_counter() - fft_started
    total_s = time.perf_counter() - started
    return mag, {
        "prep_s": prep_s,
        "fft_s": fft_s,
        "transfer_s": 0.0,
        "total_s": total_s,
    }
