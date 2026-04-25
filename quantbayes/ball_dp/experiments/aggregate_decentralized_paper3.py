#!/usr/bin/env python3
"""Aggregate official Paper 3 decentralized experiment outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.serialization import save_dataframe

KIND_TO_DIR_AND_FILES = {
    "observer": ("decentralized_observer", "observer_rows.csv", "observer_summary.csv"),
    "map_attack": (
        "decentralized_map_attack",
        "map_attack_rows.csv",
        "map_attack_summary.csv",
    ),
    "utility": (
        "decentralized_prototype_utility",
        "prototype_utility_rows.csv",
        "prototype_utility_summary.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper3")
    parser.add_argument(
        "--kind", choices=["observer", "map_attack", "utility", "all"], default="all"
    )
    return parser.parse_args()


def aggregate_one(root: Path, kind: str) -> tuple[Path, Path]:
    subdir, rows_name, summary_name = KIND_TO_DIR_AND_FILES[kind]
    search_root = root / "paper3" / subdir
    row_paths = sorted(search_root.glob(f"**/{rows_name}"))
    summary_paths = sorted(search_root.glob(f"**/{summary_name}"))
    if not row_paths and not summary_paths:
        print(
            f"No {rows_name} or {summary_name} files found under {search_root}; skipping {kind}."
        )
        return search_root / rows_name, search_root / summary_name

    aggregate_dir = root / "paper3" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    if row_paths:
        frames = []
        for path in row_paths:
            df = pd.read_csv(path)
            df["source_path"] = str(path)
            frames.append(df)
        rows = pd.concat(frames, ignore_index=True)
    else:
        rows = pd.DataFrame()
    if summary_paths:
        frames = []
        for path in summary_paths:
            df = pd.read_csv(path)
            df["source_path"] = str(path)
            frames.append(df)
        summaries = pd.concat(frames, ignore_index=True)
    else:
        summaries = pd.DataFrame()

    rows_out = aggregate_dir / f"all_{kind}_rows.csv"
    summary_out = aggregate_dir / f"all_{kind}_summary.csv"
    save_dataframe(rows, rows_out, save_parquet_if_possible=False)
    save_dataframe(summaries, summary_out, save_parquet_if_possible=False)
    print(f"Wrote {rows_out} ({len(rows)} rows)")
    print(f"Wrote {summary_out} ({len(summaries)} rows)")
    return rows_out, summary_out


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    kinds = list(KIND_TO_DIR_AND_FILES) if args.kind == "all" else [args.kind]
    for kind in kinds:
        aggregate_one(root, kind)


if __name__ == "__main__":
    main()
