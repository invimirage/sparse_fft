import time

import finufft
import numpy as np

from sparse_methods import sparse_sampling


def finufft_sampled_fft_features(matrix, points, counts: dict, meta: dict, radial_bins: int, eps: float = 1e-6) -> dict:
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    started = time.perf_counter()
    x = (2.0 * np.pi * matrix.row.astype(np.float64, copy=False)) / float(rows)
    y = (2.0 * np.pi * matrix.col.astype(np.float64, copy=False)) / float(cols)
    strengths = np.ones(matrix.nnz, dtype=np.complex128)
    s = np.asarray([point.u_shift for point in points], dtype=np.float64)
    t = np.asarray([point.v_shift for point in points], dtype=np.float64)
    coeffs = finufft.nufft2d3(x, y, strengths, s, t, isign=-1, eps=eps).astype(np.complex64, copy=False)
    elapsed = time.perf_counter() - started
    return sparse_sampling.sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)
