#!/usr/bin/env python3
"""Aggregate official Paper 2 nonconvex transcript utility/bound runs.

This script intentionally keeps dependencies light: it only scans CSV files already
written by run_nonconvex_transcript_experiment.py and produces cross-dataset CSV
summaries plus a few sanity-check figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_FILENAME = "nonconvex_transcript_seed_rows.csv"
SUMMARY_FILENAME = "nonconvex_transcript_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Paper 2 nonconvex transcript utility/bound results."
    )
    parser.add_argument("--results-root", type=str, default="results_paper2_nonconvex")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset tags to include. Defaults to every dataset directory under results-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to <results-root>/aggregate.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def savefig_stem(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def dataset_dirs(results_root: Path, datasets: Iterable[str] | None) -> list[Path]:
    if datasets:
        return [results_root / str(ds) for ds in datasets]
    return [
        p
        for p in sorted(results_root.iterdir())
        if p.is_dir() and p.name != "aggregate"
    ]


def load_rows(results_root: Path, datasets: Iterable[str] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in dataset_dirs(results_root, datasets):
        path = d / SEED_FILENAME
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - diagnostics path
            print(f"Skipping unreadable {path}: {exc}")
            continue
        if "dataset_tag" not in df.columns:
            df["dataset_tag"] = d.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    if "row_key" in raw.columns:
        raw = raw.drop_duplicates(subset=["row_key"], keep="last")
    return raw


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    group_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "mechanism",
            "epsilon",
            "m",
            "radius_tag",
            "task",
            "hidden_dim",
            "num_steps",
            "batch_size",
            "clip_norm",
        ]
        if c in raw.columns
    ]
    metric_cols = [
        c
        for c in [
            "accuracy",
            "public_eval_loss",
            "primary_bound_hayes",
            "primary_bound_rdp",
            "bound_hayes_ball",
            "bound_hayes_standard_same_noise",
            "bound_rdp_ball",
            "bound_rdp_standard_same_noise",
            "mean_delta_ball",
            "mean_delta_standard",
            "mean_ball_to_standard_ratio",
            "max_direct_c_ball",
            "max_direct_c_standard",
            "noise_multiplier",
            "actual_epsilon_ball",
            "actual_epsilon_standard",
            "primary_epsilon",
        ]
        if c in raw.columns
    ]
    if not group_cols or not metric_cols:
        return pd.DataFrame()

    rows = []
    for key, g in raw.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key, strict=True)}
        row["n_rows"] = int(len(g))
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            finite = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{col}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def plot_metric(
    summary: pd.DataFrame, *, metric: str, ylabel: str, title: str, out_stem: Path
) -> None:
    if (
        summary.empty
        or metric not in summary.columns
        or "epsilon" not in summary.columns
    ):
        return
    label_cols = [c for c in ["dataset_tag", "mechanism"] if c in summary.columns]
    if not label_cols:
        return
    fig, ax = plt.subplots()
    for key, g in summary.groupby(label_cols, dropna=False):
        g = g.sort_values("epsilon")
        x = pd.to_numeric(g["epsilon"], errors="coerce")
        y = pd.to_numeric(g[metric], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
            ax.plot(x[mask], y[mask], marker="o", label=label)
    ax.set_xlabel("epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_stem(fig, out_stem)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plot_dir = output_dir / "figures"
    plot_metric(
        summary,
        metric="accuracy_mean",
        ylabel="public test accuracy",
        title="Paper 2 utility across datasets",
        out_stem=plot_dir / "all_accuracy_vs_epsilon",
    )
    plot_metric(
        summary,
        metric="primary_bound_hayes_mean",
        ylabel=r"$B_{1/m}^{test}$",
        title="Paper 2 finite-prior Hayes bound across datasets",
        out_stem=plot_dir / "all_hayes_bound_vs_epsilon",
    )
    plot_metric(
        summary,
        metric="mean_ball_to_standard_ratio_mean",
        ylabel=r"mean $\Delta_t(r)/(2C_t)$",
        title="Ball-to-standard sensitivity ratio across datasets",
        out_stem=plot_dir / "all_sensitivity_ratio_vs_epsilon",
    )

    if "m" in summary.columns and "primary_bound_hayes_mean" in summary.columns:
        tmp = summary.copy()
        labels = [
            c for c in ["dataset_tag", "mechanism", "epsilon"] if c in tmp.columns
        ]
        fig, ax = plt.subplots()
        for key, g in tmp.groupby(labels, dropna=False):
            g = g.sort_values("m")
            x = pd.to_numeric(g["m"], errors="coerce")
            y = pd.to_numeric(g["primary_bound_hayes_mean"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
                ax.plot(x[mask], y[mask], marker="o", label=label)
        ax.set_xlabel("finite prior size m")
        ax.set_ylabel(r"$B_{1/m}^{test}$")
        ax.set_title("Paper 2 Hayes bound vs finite-prior size")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        savefig_stem(fig, plot_dir / "all_hayes_bound_vs_m")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    results_root = Path(args.results_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir else results_root / "aggregate"
    )
    raw = load_rows(results_root, args.datasets)
    if raw.empty:
        print(f"No {SEED_FILENAME} files found under {results_root}")
        return
    summary = summarize(raw)
    save_csv(raw, output_dir / "all_nonconvex_transcript_seed_rows.csv")
    save_csv(summary, output_dir / "all_nonconvex_transcript_summary.csv")
    if not bool(args.no_plots):
        write_plots(summary, output_dir)
    print(
        f"Wrote {len(raw)} raw rows and {len(summary)} summary rows to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
