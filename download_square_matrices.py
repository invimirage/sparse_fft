#!/usr/bin/env python3

"""Download square SuiteSparse matrices for the HPC FFT experiments.

This script follows the selection style in Ginkgo/download_diverse_large_matrices.py
but requires square matrices and creates two manifests:

* full_eval: about 100 matrices with 10k-20k rows/cols
* case_study: about 20 matrices with 50k-100k rows/cols
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import ssgetpy


REPO_ROOT = Path(__file__).resolve().parents[1]
GINKGO_ROOT = REPO_ROOT / "Ginkgo"
if str(GINKGO_ROOT) not in sys.path:
    sys.path.insert(0, str(GINKGO_ROOT))

from download_dataset import clean_matrix, select_mtx_file  # noqa: E402


def existing_names(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.mtx")} if directory.exists() else set()


def search_square_candidates(min_size: int, max_size: int, exclude: set[str]) -> list[dict]:
    index = ssgetpy.search(rowbounds=(min_size, max_size), limit=100000)
    items: list[dict] = []
    for mat in index:
        name = getattr(mat, "name", None)
        group = getattr(mat, "group", None)
        rows = int(getattr(mat, "rows", 0) or getattr(mat, "nrows", 0) or 0)
        cols = int(getattr(mat, "cols", 0) or getattr(mat, "ncols", 0) or 0)
        nnz = int(getattr(mat, "nnz", 0) or 0)
        kind = getattr(mat, "kind", "") or ""
        if not name or name in exclude:
            continue
        if rows != cols:
            continue
        if not (min_size <= rows <= max_size):
            continue
        if kind == "sequence":
            continue
        items.append({"group": group or "unknown", "name": name, "rows": rows, "cols": cols, "nnz": nnz, "kind": kind})
    return items


def select_diverse(items: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_group[item["group"]].append(item)
    group_names = list(by_group)
    rng.shuffle(group_names)
    for group in group_names:
        rng.shuffle(by_group[group])

    selected: list[dict] = []
    round_idx = 0
    while len(selected) < limit:
        added = False
        for group in group_names:
            bucket = by_group[group]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_idx += 1
    return selected


import concurrent.futures

def _download_single(args_tuple):
    idx, total, item, output_dir = args_tuple
    group = item["group"]
    name = item["name"]
    output_path = output_dir / f"{name}.mtx"
    print(f"[{idx}/{total}] {group}/{name}", flush=True)
    if output_path.exists():
        print(f"[{idx}/{total}] exists: {output_path}", flush=True)
        return item | {"status": "exists", "path": str(output_path)}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ssgetpy.fetch(f"{group}/{name}", location=tmpdir)
            mtx_file = select_mtx_file(tmpdir, name)
            if not mtx_file:
                print(f"[{idx}/{total}] no .mtx found", flush=True)
                return item | {"status": "missing_mtx", "path": ""}
            if clean_matrix(mtx_file, str(output_path)):
                print(f"[{idx}/{total}] saved: {output_path}", flush=True)
                return item | {"status": "downloaded", "path": str(output_path)}
            else:
                print(f"[{idx}/{total}] clean failed", flush=True)
                return item | {"status": "clean_failed", "path": ""}
    except Exception as exc:
        print(f"[{idx}/{total}] error: {exc}", flush=True)
        return item | {"status": f"error:{type(exc).__name__}", "path": ""}

def download_selected(output_dir: Path, selected: list[dict], workers: int = 1) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, len(selected), item, output_dir) for i, item in enumerate(selected, start=1)]
    if workers <= 1:
        return [_download_single(t) for t in tasks]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_download_single, tasks))

def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "group", "name", "rows", "cols", "nnz", "kind", "status", "path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def run_split(split: str, output_dir: Path, min_size: int, max_size: int, limit: int, seed: int, exclude: set[str], workers: int) -> list[dict]:
    print(f"Searching {split}: square {min_size}-{max_size}, limit={limit}", flush=True)
    candidates = search_square_candidates(min_size, max_size, exclude)
    print(f"Candidates: {len(candidates)}", flush=True)
    selected = select_diverse(candidates, limit, seed)
    print(f"Selected: {len(selected)}", flush=True)
    downloaded = download_selected(output_dir, selected, workers)
    return [row | {"split": split} for row in downloaded]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download square SuiteSparse matrices for density FFT HPC experiments")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/suitesparse_square_manifest.csv"))
    parser.add_argument("--full-limit", type=int, default=100)
    parser.add_argument("--case-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-dir", type=Path, action="append", default=[])
    import os
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser.add_argument("--workers", type=int, default=default_workers)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    exclude: set[str] = set()
    for directory in args.exclude_dir:
        exclude.update(existing_names(directory))

    rows: list[dict] = []
    full_rows = run_split("full_eval", args.output_root / "full_eval_10k_20k", 10000, 20000, args.full_limit, args.seed, exclude, args.workers)
    exclude.update(row["name"] for row in full_rows)
    case_rows = run_split("case_study", args.output_root / "case_study_50k_100k", 50000, 100000, args.case_limit, args.seed + 1, exclude, args.workers)
    rows.extend(full_rows)
    rows.extend(case_rows)
    write_manifest(args.manifest, rows)
    print(f"Wrote manifest: {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
