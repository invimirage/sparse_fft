#!/usr/bin/env python3

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse


TWO_PI = 2.0 * math.pi


def clear_gpu_memory() -> None:
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@dataclass(frozen=True)
class SamplePoint:
    u_shift: int
    v_shift: int
    u_freq: int
    v_freq: int
    multiplicity: int
    stratum: str
    weight: float


def shifted_coord_bounds(size: int) -> tuple[int, int]:
    return -(size // 2), (size - 1) // 2


def shifted_to_fft_index(coord: int, size: int) -> int:
    return coord if coord >= 0 else coord + size


def row_is_self_conjugate(u_shift: int, rows: int) -> bool:
    return u_shift == 0 or (rows % 2 == 0 and u_shift == -(rows // 2))


def point_multiplicity(u_shift: int, v_shift: int, rows: int, cols: int) -> int:
    row_self = row_is_self_conjugate(u_shift, rows)
    col_self = v_shift == 0 or (cols % 2 == 0 and v_shift == -(cols // 2))
    return 1 if row_self and col_self else 2


def allowed_v_intervals(u_shift: int, rows: int, cols: int) -> list[tuple[int, int]]:
    v_min, v_max = shifted_coord_bounds(cols)
    if u_shift > 0:
        return [(v_min, v_max)]
    if row_is_self_conjugate(u_shift, rows):
        intervals = []
        if cols % 2 == 0:
            intervals.append((v_min, v_min))
        intervals.append((0, v_max))
        return intervals
    return []


def interval_len(interval: tuple[int, int]) -> int:
    lo, hi = interval
    return max(0, hi - lo + 1)


def intersect_interval(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return (lo, hi)


def subtract_intervals(base: list[tuple[int, int]], remove: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = base[:]
    for r_lo, r_hi in remove:
        next_result = []
        for b_lo, b_hi in result:
            if r_hi < b_lo or r_lo > b_hi:
                next_result.append((b_lo, b_hi))
                continue
            if b_lo < r_lo:
                next_result.append((b_lo, r_lo - 1))
            if r_hi < b_hi:
                next_result.append((r_hi + 1, b_hi))
        result = next_result
    return result


def normalize_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted((lo, hi) for lo, hi in intervals if lo <= hi)
    if not intervals:
        return []
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        cur_lo, cur_hi = merged[-1]
        if lo <= cur_hi + 1:
            merged[-1] = (cur_lo, max(cur_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    return sum(interval_len(interval) for interval in intervals)


def sample_from_intervals(rng: np.random.Generator, intervals: list[tuple[int, int]]) -> int:
    total = interval_total(intervals)
    pick = int(rng.integers(total))
    for lo, hi in intervals:
        width = hi - lo + 1
        if pick < width:
            return lo + pick
        pick -= width
    raise RuntimeError("Failed to sample interval")


def estimate_gpu_memory_gb(rows: int, cols: int) -> float:
    elems = rows * cols
    return elems * (2 + 8 + 4) / (1024 ** 3)


def read_matrix_header(path: Path) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            rows, cols, nnz = map(int, line.split()[:3])
            return rows, cols, nnz
    raise ValueError(f"Missing Matrix Market shape line in {path}")


def load_binary_coo(path: Path) -> csr_matrix:
    matrix = mmread(path)
    if not issparse(matrix):
        matrix = csr_matrix(matrix)
    else:
        matrix = csr_matrix(matrix)
    matrix = matrix.sign().astype(np.float32)
    matrix.eliminate_zeros()
    return matrix


def build_row_specs(rows: int, cols: int, core_frac: float, axis_band: int, diag_band: int):
    u_min, u_max = shifted_coord_bounds(rows)
    v_min, v_max = shifted_coord_bounds(cols)
    core_u = max(1, int(round(rows * core_frac / 2.0)))
    core_v = max(1, int(round(cols * core_frac / 2.0)))
    specs = {"core": [], "axis": [], "diag": [], "rest": []}
    counts = {key: 0 for key in specs}

    for u_shift in range(u_min, u_max + 1):
        allowed = allowed_v_intervals(u_shift, rows, cols)
        if not allowed:
            continue

        core = []
        for interval in allowed:
            clipped = intersect_interval(interval, (-core_v, core_v))
            if clipped is not None and abs(u_shift) <= core_u:
                core.append(clipped)
        core = normalize_intervals(core)

        axis = []
        if abs(u_shift) <= axis_band:
            axis = subtract_intervals(allowed, core)
        else:
            for interval in allowed:
                clipped = intersect_interval(interval, (-axis_band, axis_band))
                if clipped is not None:
                    axis.append(clipped)
            axis = subtract_intervals(normalize_intervals(axis), core)
        axis = normalize_intervals(axis)

        diag_candidates = []
        abs_u = abs(u_shift)
        diag_ranges = [
            (abs_u - diag_band, abs_u + diag_band),
            (-abs_u - diag_band, -abs_u + diag_band),
        ]
        for interval in allowed:
            for diag_range in diag_ranges:
                clipped = intersect_interval(interval, diag_range)
                if clipped is not None:
                    diag_candidates.append(clipped)
        diag = subtract_intervals(normalize_intervals(diag_candidates), core + axis)
        diag = normalize_intervals(diag)

        rest = subtract_intervals(allowed, core + axis + diag)
        rest = normalize_intervals(rest)

        for name, intervals in (("core", core), ("axis", axis), ("diag", diag), ("rest", rest)):
            count = interval_total(intervals)
            if count <= 0:
                continue
            specs[name].append((u_shift, intervals, count))
            counts[name] += count

    counts["total"] = sum(counts[name] for name in ("core", "axis", "diag", "rest"))
    meta = {
        "core_u": core_u,
        "core_v": core_v,
        "u_bounds": (u_min, u_max),
        "v_bounds": (v_min, v_max),
    }
    return specs, counts, meta


def gaussian_score(x: float, sigma: float) -> float:
    sigma = max(float(sigma), 1.0)
    return math.exp(-0.5 * (x / sigma) ** 2) / sigma


def total_half_spectrum_points(rows: int, cols: int) -> int:
    u_min, u_max = shifted_coord_bounds(rows)
    total = 0
    for u_shift in range(u_min, u_max + 1):
        total += interval_total(allowed_v_intervals(u_shift, rows, cols))
    return total


def classify_sample_region(u_shift: int, v_shift: int, rows: int, cols: int, core_frac: float, axis_band: int, diag_band: int) -> str:
    core_u = max(1, int(round(rows * core_frac / 2.0)))
    core_v = max(1, int(round(cols * core_frac / 2.0)))
    if abs(u_shift) <= core_u and abs(v_shift) <= core_v:
        return "core"
    if abs(u_shift) <= axis_band or abs(v_shift) <= axis_band:
        return "axis"
    if abs(v_shift - u_shift) <= diag_band or abs(v_shift + u_shift) <= diag_band:
        return "diag"
    return "rest"


def sample_points(rows: int, cols: int, total_samples: int, seed: int, core_frac: float, axis_band: int, diag_band: int):
    total_unique = total_half_spectrum_points(rows, cols)
    total_samples = min(total_samples, total_unique)
    rng = np.random.default_rng(seed)
    u_min, u_max = shifted_coord_bounds(rows)
    row_specs = []
    row_counts = []
    for u_shift in range(u_min, u_max + 1):
        intervals = allowed_v_intervals(u_shift, rows, cols)
        count = interval_total(intervals)
        if count <= 0:
            continue
        row_specs.append((u_shift, intervals))
        row_counts.append(count)
    row_prob = np.array(row_counts, dtype=np.float64)
    row_prob /= row_prob.sum()

    chosen: set[tuple[int, int]] = set()

    while len(chosen) < total_samples:
        row_idx = int(rng.choice(len(row_specs), p=row_prob))
        u_shift, intervals = row_specs[row_idx]
        v_shift = sample_from_intervals(rng, intervals)
        chosen.add((u_shift, v_shift))

    samples = []
    counts = {"total": total_unique, "core": 0, "axis": 0, "diag": 0, "rest": 0}
    uniform_weight = total_unique / float(total_samples) if total_samples > 0 else 0.0
    for (u_shift, v_shift) in sorted(chosen):
        region = classify_sample_region(u_shift, v_shift, rows, cols, core_frac, axis_band, diag_band)
        counts[region] += 1
        samples.append(
            SamplePoint(
                u_shift=u_shift,
                v_shift=v_shift,
                u_freq=shifted_to_fft_index(u_shift, rows),
                v_freq=shifted_to_fft_index(v_shift, cols),
                multiplicity=point_multiplicity(u_shift, v_shift, rows, cols),
                stratum=region,
                weight=uniform_weight,
            )
        )
    meta = {
        "u_bounds": shifted_coord_bounds(rows),
        "v_bounds": shifted_coord_bounds(cols),
    }
    return samples, counts, meta


def approximate_fft_features(matrix: csr_matrix, sample_count: int, radial_bins: int, seed: int, batch_size: int, core_frac: float, axis_band: int, diag_band: int):
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape
    points, counts, meta = sample_points(rows, cols, sample_count, seed, core_frac, axis_band, diag_band)

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
        coeff_batch = cp.exp((-1j * TWO_PI) * phase).astype(cp.complex64).sum(axis=1)
        coeffs[start:stop] = cp.asnumpy(coeff_batch)
        del phase
        del coeff_batch
        clear_gpu_memory()
    elapsed = time.perf_counter() - started

    r_max = math.sqrt(max(abs(meta["u_bounds"][0]), abs(meta["u_bounds"][1])) ** 2 + max(abs(meta["v_bounds"][0]), abs(meta["v_bounds"][1])) ** 2)
    radial_energy = np.zeros(radial_bins, dtype=np.float64)
    sample_energy = np.zeros(len(points), dtype=np.float64)
    for idx, point in enumerate(points):
        radius = math.sqrt(point.u_shift ** 2 + point.v_shift ** 2)
        bin_id = min(int((radius / r_max) * radial_bins), radial_bins - 1)
        energy = float(abs(coeffs[idx]) ** 2)
        total_weight = point.weight * point.multiplicity
        weighted_energy = total_weight * energy
        sample_energy[idx] = weighted_energy
        radial_energy[bin_id] += weighted_energy

    radial = radial_energy / radial_energy.sum() if radial_energy.sum() > 0 else radial_energy
    sample_prob = sample_energy / sample_energy.sum() if sample_energy.sum() > 0 else sample_energy
    sample_prob = sample_prob[sample_prob > 0]
    entropy_pointwise = float(-np.sum(sample_prob * np.log(sample_prob))) if sample_prob.size else 0.0
    radial_prob = radial[radial > 0]
    entropy_radial = float(-np.sum(radial_prob * np.log(radial_prob))) if radial_prob.size else 0.0

    return {
        "radial": radial.astype(np.float64),
        "entropy": entropy_pointwise,
        "entropy_radial": entropy_radial,
        "elapsed": elapsed,
        "sample_count": len(points),
        "spectral_points": counts["total"],
        "strata_core": counts["core"],
        "strata_axis": counts["axis"],
        "strata_diag": counts["diag"],
        "strata_rest": counts["rest"],
    }


def dense_fft_features(matrix: csr_matrix, radial_bins: int):
    matrix = matrix.tocoo(copy=False)
    rows, cols = matrix.shape

    started = time.perf_counter()
    clear_gpu_memory()
    dense = cp.zeros((rows, cols), dtype=cp.float16)
    dense[cp.asarray(matrix.row), cp.asarray(matrix.col)] = 1.0
    spectrum = cp.fft.fftshift(cp.fft.fft2(dense))
    mag = cp.abs(spectrum).astype(cp.float32)

    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    y = cp.arange(rows, dtype=cp.float32)[:, None]
    x = cp.arange(cols, dtype=cp.float32)[None, :]
    radius = cp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = float(cp.max(radius))
    bins = cp.clip(cp.floor((radius / r_max) * radial_bins).astype(cp.int32), 0, radial_bins - 1)

    energy = (mag ** 2).ravel()
    radial_energy = cp.bincount(bins.ravel(), weights=energy, minlength=radial_bins)
    radial = radial_energy / radial_energy.sum() if float(radial_energy.sum()) > 0 else radial_energy

    prob = energy / energy.sum() if float(energy.sum()) > 0 else energy
    prob = prob[prob > 0]
    entropy = -cp.sum(prob * cp.log(prob)) if prob.size else cp.asarray(0.0, dtype=cp.float32)

    radial_prob = radial[radial > 0]
    entropy_radial = -cp.sum(radial_prob * cp.log(radial_prob)) if radial_prob.size else cp.asarray(0.0, dtype=cp.float32)
    elapsed = time.perf_counter() - started

    result = {
        "radial": cp.asnumpy(radial).astype(np.float64),
        "entropy": float(entropy),
        "entropy_radial": float(entropy_radial),
        "elapsed": elapsed,
    }

    del dense
    del spectrum
    del mag
    del y
    del x
    del radius
    del bins
    del energy
    del radial_energy
    clear_gpu_memory()
    return result


def select_candidates(dataset_dir: Path, min_rows: int, max_rows: int, max_dense_elements: int | None, max_matrices: int | None, matrix_names: set[str] | None):
    candidates = []
    for path in sorted(dataset_dir.glob("*.mtx")):
        if matrix_names is not None and path.stem not in matrix_names:
            continue
        rows, cols, nnz = read_matrix_header(path)
        if rows <= min_rows or rows >= max_rows:
            continue
        elems = rows * cols
        if max_dense_elements is not None and elems > max_dense_elements:
            continue
        candidates.append({
            "path": path,
            "matrix": path.stem,
            "rows": rows,
            "cols": cols,
            "nnz": nnz,
            "dense_elements": elems,
            "gpu_memory_gb": estimate_gpu_memory_gb(rows, cols),
        })
    if max_matrices is not None:
        candidates = candidates[:max_matrices]
    return candidates


def compare_one(path: Path, sample_count: int, radial_bins: int, seed: int, batch_size: int, core_frac: float, axis_band: int, diag_band: int):
    clear_gpu_memory()
    matrix = load_binary_coo(path)
    approx = approximate_fft_features(matrix, sample_count, radial_bins, seed, batch_size, core_frac, axis_band, diag_band)
    clear_gpu_memory()
    dense = dense_fft_features(matrix, radial_bins)
    clear_gpu_memory()

    radial_abs_err = np.abs(approx["radial"] - dense["radial"])
    radial_rel_l1 = float(radial_abs_err.sum() / max(dense["radial"].sum(), 1e-12))
    entropy_pointwise_abs_err = abs(approx["entropy"] - dense["entropy"])
    entropy_abs_err = abs(approx["entropy_radial"] - dense["entropy_radial"])
    speedup = dense["elapsed"] / approx["elapsed"] if approx["elapsed"] > 0 else float("inf")

    row = {
        "matrix": path.stem,
        "rows": matrix.shape[0],
        "cols": matrix.shape[1],
        "nnz": int(matrix.nnz),
        "sample_count": approx["sample_count"],
        "dense_time_s": dense["elapsed"],
        "approx_time_s": approx["elapsed"],
        "speedup": speedup,
        "entropy_dense": dense["entropy_radial"],
        "entropy_approx": approx["entropy_radial"],
        "entropy_abs_err": entropy_abs_err,
        "entropy_pointwise_dense": dense["entropy"],
        "entropy_pointwise_approx": approx["entropy"],
        "entropy_pointwise_abs_err": entropy_pointwise_abs_err,
        "radial_l1_err": float(radial_abs_err.sum()),
        "radial_mae": float(radial_abs_err.mean()),
        "radial_max_err": float(radial_abs_err.max()),
        "radial_rel_l1": radial_rel_l1,
        "spectral_points_unique_half": approx["spectral_points"],
        "strata_core": approx["strata_core"],
        "strata_axis": approx["strata_axis"],
        "strata_diag": approx["strata_diag"],
        "strata_rest": approx["strata_rest"],
    }
    for idx, value in enumerate(dense["radial"]):
        row[f"radial_dense_{idx}"] = float(value)
    for idx, value in enumerate(approx["radial"]):
        row[f"radial_approx_{idx}"] = float(value)
    for idx, value in enumerate(radial_abs_err):
        row[f"radial_abs_err_{idx}"] = float(value)
    return row


def write_report(df: pd.DataFrame, report_path: Path):
    lines = []
    lines.append("# Approximate FFT Benchmark")
    lines.append("")
    if df.empty:
        lines.append("No matrices satisfied the selection constraints.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(f"Matrices benchmarked: {len(df)}")
    lines.append(f"Mean speedup: {df['speedup'].mean():.2f}x")
    lines.append(f"Median speedup: {df['speedup'].median():.2f}x")
    lines.append(f"Mean radial MAE: {df['radial_mae'].mean():.6f}")
    lines.append(f"Mean radial-bin entropy abs err: {df['entropy_abs_err'].mean():.6f}")
    lines.append("")
    lines.append("| Matrix | Shape | NNZ | Dense s | Approx s | Speedup | Radial MAE | Entropy err |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.matrix} | {row.rows}x{row.cols} | {row.nnz} | {row.dense_time_s:.3f} | {row.approx_time_s:.3f} | {row.speedup:.2f}x | {row.radial_mae:.6f} | {row.entropy_abs_err:.6f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sparse-sampled FFT features against dense FFT")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/rzhang38/VenomTileSkipping/FFT-Graph-Sparse/dataset_combined"))
    parser.add_argument("--output-csv", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--output-report", type=Path, default=Path("benchmark_report.md"))
    parser.add_argument("--min-rows", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--radial-bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--core-frac", type=float, default=0.12)
    parser.add_argument("--axis-band", type=int, default=3)
    parser.add_argument("--diag-band", type=int, default=3)
    parser.add_argument("--max-dense-elements", type=int, default=120000000)
    parser.add_argument("--max-matrices", type=int, default=8)
    parser.add_argument("--matrix-names", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    matrix_names = None
    if args.matrix_names:
        matrix_names = {item.strip() for item in args.matrix_names.split(",") if item.strip()}
    candidates = select_candidates(
        args.dataset_dir,
        args.min_rows,
        args.max_rows,
        args.max_dense_elements,
        args.max_matrices,
        matrix_names,
    )

    rows = []
    for idx, item in enumerate(candidates, start=1):
        print(
            f"[{idx}/{len(candidates)}] {item['matrix']} {item['rows']}x{item['cols']} nnz={item['nnz']} gpu_est={item['gpu_memory_gb']:.2f}GB"
        )
        row = compare_one(
            item["path"],
            sample_count=args.sample_count,
            radial_bins=args.radial_bins,
            seed=args.seed + idx,
            batch_size=args.batch_size,
            core_frac=args.core_frac,
            axis_band=args.axis_band,
            diag_band=args.diag_band,
        )
        rows.append(row)
        print(
            f"  dense={row['dense_time_s']:.3f}s approx={row['approx_time_s']:.3f}s speedup={row['speedup']:.2f}x radial_mae={row['radial_mae']:.6f} entropy_err={row['entropy_abs_err']:.6f}"
        )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    write_report(df, args.output_report)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_report}")


if __name__ == "__main__":
    main()
