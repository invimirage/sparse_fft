#!/usr/bin/env python3

"""Merge per-matrix JSON outputs and summarize normalization performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "log_mae_interp",
    "entropy_error_interp",
    "entropy_error_direct",
    "radial_error_interp",
    "radial_error_direct",
]


def load_records(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        base = {
            "matrix": doc.get("matrix"),
            "split": doc.get("split"),
            "rows": doc.get("rows"),
            "cols": doc.get("cols"),
            "nnz": doc.get("nnz"),
            "reference_status": (doc.get("reference") or {}).get("status"),
            "reference_backend": (doc.get("reference") or {}).get("backend"),
            "reference_total_s": (doc.get("reference") or {}).get("total_s"),
        }
        hybrid_features = doc.get("hybrid_features") or {}
        for key, value in hybrid_features.items():
            base[key] = value
        for record in doc.get("records", []):
            rows.append(base | record)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    group_cols = ["split", "method", "normalization", "resolution"]
    for optional in ["curve_index", "density_ratio", "sample_fraction", "sample_axis_rows", "sample_axis_cols", "sample_count"]:
        if optional in ok.columns:
            group_cols.append(optional)
    agg = {}
    for metric in METRICS:
        if metric in ok.columns:
            agg[metric] = ["count", "mean", "median", "std"]
    for timing in ["compress_s", "fft_s", "interp_s", "compute_s", "total_s"]:
        if timing in ok.columns:
            agg[timing] = ["mean", "median"]
    for feature in ["density_1024_entropy_norm", "density_1024_std", "density_1024_gini"]:
        if feature in ok.columns:
            agg[feature] = ["mean", "median"]
    if not agg:
        return pd.DataFrame()
    out = ok.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join(col).rstrip("_") for col in out.columns]
    return out.reset_index()


def best_by_metric(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        column = f"{metric}_mean"
        if column not in summary.columns:
            continue
        usable = summary.dropna(subset=[column])
        for split, split_df in usable.groupby("split", dropna=False):
            idx = split_df[column].idxmin()
            row = split_df.loc[idx].to_dict()
            rows.append({"split": split, "metric": metric, "best_method": row.get("method"), "best_normalization": row.get("normalization"), "best_resolution": row.get("resolution"), "mean_error": row.get(column)})
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, best: pd.DataFrame, path: Path) -> None:
    lines = ["# Density FFT HPC Results", ""]
    if best.empty:
        lines.append("No successful metric rows were available.")
    else:
        lines.extend(["## Best Mean Error By Metric", "", best.to_markdown(index=False), ""])
    if not summary.empty:
        keep = [col for col in ["split", "method", "normalization", "resolution", "curve_index", "density_ratio", "sample_fraction", "sample_count", "log_mae_interp_mean", "entropy_error_interp_mean", "entropy_error_direct_mean", "radial_error_interp_mean", "radial_error_direct_mean", "total_s_median", "density_1024_entropy_norm_mean", "density_1024_std_mean", "density_1024_gini_mean"] if col in summary.columns]
        lines.extend(["## Summary", "", summary[keep].to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HPC density FFT JSON results")
    parser.add_argument("--input-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_records(args.input_dir)
    all_csv = args.output_dir / "all_results.csv"
    summary_csv = args.output_dir / "summary_by_method.csv"
    best_csv = args.output_dir / "best_by_metric.csv"
    report_md = args.output_dir / "report.md"
    df.to_csv(all_csv, index=False)
    summary = summarize(df)
    summary.to_csv(summary_csv, index=False)
    best = best_by_metric(summary)
    best.to_csv(best_csv, index=False)
    write_markdown(summary, best, report_md)
    print(f"Wrote {all_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {best_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
