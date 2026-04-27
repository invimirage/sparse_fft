import math
import os
import time

import cupy as cp
import numpy as np

import approx_fft_benchmark as sparse_bench


SPARSE_DIRECT_MAX_OPS = int(float(os.environ.get("SPARSE_DIRECT_MAX_OPS", "5e10")))
SPARSE_DIRECT_MAX_POINTS = int(float(os.environ.get("SPARSE_DIRECT_MAX_POINTS", "1000000")))


def sampled_features_from_coeffs(points, counts: dict, meta: dict, coeffs: np.ndarray, radial_bins: int, elapsed: float) -> dict:
    r_max = math.sqrt(
        max(abs(meta["u_bounds"][0]), abs(meta["u_bounds"][1])) ** 2
        + max(abs(meta["v_bounds"][0]), abs(meta["v_bounds"][1])) ** 2
    )
    radial_energy = np.zeros(radial_bins, dtype=np.float64)
    sample_energy = np.zeros(len(points), dtype=np.float64)
    for idx, point in enumerate(points):
        radius = math.sqrt(point.u_shift ** 2 + point.v_shift ** 2)
        bin_id = min(int((radius / r_max) * radial_bins), radial_bins - 1) if r_max > 0 else 0
        energy = float(abs(coeffs[idx]) ** 2)
        total_weight = point.weight * point.multiplicity
        weighted_energy = total_weight * energy
        sample_energy[idx] = weighted_energy
        radial_energy[bin_id] += weighted_energy

    radial = radial_energy / radial_energy.sum() if radial_energy.sum() > 0 else radial_energy
    sample_prob = sample_energy / sample_energy.sum() if sample_energy.sum() > 0 else sample_energy
    sample_prob = sample_prob[sample_prob > 0]
    entropy = float(-np.sum(sample_prob * np.log(sample_prob))) if sample_prob.size else 0.0
    return {
        "radial": radial.astype(np.float64),
        "entropy": entropy,
        "elapsed": elapsed,
        "points": points,
        "counts": counts,
        "meta": meta,
        "coeffs": coeffs.astype(np.complex64, copy=False),
        "sample_count": len(points),
        "spectral_points": counts["total"],
        "strata_core": counts["core"],
        "strata_axis": counts["axis"],
        "strata_diag": counts["diag"],
        "strata_rest": counts["rest"],
    }


def percent_axis_sample_count(size: int, fraction: float = 0.01) -> int:
    return max(1, int(math.floor(size * fraction)))


def evenly_spaced_shifted_coords(size: int, count: int) -> list[int]:
    lo, hi = sparse_bench.shifted_coord_bounds(size)
    if count >= size:
        return list(range(lo, hi + 1))
    positions = np.linspace(0, size - 1, count)
    idx = sorted(set(int(round(pos)) for pos in positions))
    if len(idx) < count:
        used = set(idx)
        for candidate in range(size):
            if candidate in used:
                continue
            idx.append(candidate)
            used.add(candidate)
            if len(idx) == count:
                break
        idx.sort()
    return [lo + i for i in idx[:count]]


def cartesian_grid_sample_points(rows: int, cols: int):
    row_coords = evenly_spaced_shifted_coords(rows, percent_axis_sample_count(rows))
    col_coords = evenly_spaced_shifted_coords(cols, percent_axis_sample_count(cols))
    points = []
    total_samples = len(row_coords) * len(col_coords)
    total_weight = (rows * cols) / float(total_samples) if total_samples > 0 else 0.0
    counts = {"total": rows * cols, "core": 0, "axis": 0, "diag": 0, "rest": 0}
    for u_shift in row_coords:
        for v_shift in col_coords:
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
        "sampled_rows": len(row_coords),
        "sampled_cols": len(col_coords),
    }
    return points, counts, meta


def sampled_sparse_fft_features_from_points(matrix, points, counts: dict, meta: dict, radial_bins: int, batch_size: int):
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    if len(points) > SPARSE_DIRECT_MAX_POINTS:
        raise MemoryError(f"sample grid has {len(points)} points > limit {SPARSE_DIRECT_MAX_POINTS}")
    estimated_ops = int(len(points)) * int(matrix.nnz)
    if estimated_ops > SPARSE_DIRECT_MAX_OPS:
        raise MemoryError(f"sparse direct workload {estimated_ops} exceeds limit {SPARSE_DIRECT_MAX_OPS}")

    r_idx = cp.asarray(matrix.row.astype(np.float32))
    c_idx = cp.asarray(matrix.col.astype(np.float32))
    freq_u = cp.asarray(np.array([point.u_freq for point in points], dtype=np.float32))
    freq_v = cp.asarray(np.array([point.v_freq for point in points], dtype=np.float32))

    coeffs = np.empty(len(points), dtype=np.complex64)
    started = time.perf_counter()
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        u_batch = freq_u[start:stop]
        v_batch = freq_v[start:stop]
        phase = (
            (u_batch[:, None] * r_idx[None, :]) / float(rows)
            + (v_batch[:, None] * c_idx[None, :]) / float(cols)
        )
        coeff_batch = cp.exp((-1j * sparse_bench.TWO_PI) * phase).astype(cp.complex64).sum(axis=1)
        coeffs[start:stop] = cp.asnumpy(coeff_batch)
        del phase
        del coeff_batch
        sparse_bench.clear_gpu_memory()
    elapsed = time.perf_counter() - started
    return sampled_features_from_coeffs(points, counts, meta, coeffs, radial_bins, elapsed)


def sampled_sparse_fft_features(matrix, radial_bins: int, batch_size: int):
    rows, cols = matrix.shape
    points, counts, meta = cartesian_grid_sample_points(rows, cols)
    return sampled_sparse_fft_features_from_points(matrix, points, counts, meta, radial_bins, batch_size)
