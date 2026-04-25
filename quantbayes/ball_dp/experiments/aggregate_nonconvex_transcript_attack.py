#!/usr/bin/env python3
"""Aggregate Paper 2 nonconvex transcript attack summaries."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper2")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument(
        "--out-name",
        type=str,
        default="aggregate_nonconvex_transcript_attack_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root) / "paper2" / "nonconvex_transcript_attack"
    if args.model_tag:
        search_root = root / args.model_tag
    else:
        search_root = root
    paths = sorted(search_root.glob("**/attack_summary.csv"))
    if not paths:
        raise SystemExit(f"No attack_summary.csv files found under {search_root}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["summary_path"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if {"dataset_tag", "mechanism", "attack_mode", "epsilon", "m"}.issubset(
        out.columns
    ):
        out = out.sort_values(
            ["dataset_tag", "mechanism", "attack_mode", "epsilon", "m"]
        ).reset_index(drop=True)
    out_path = root / args.out_name
    save_dataframe(out, out_path, save_parquet_if_possible=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
