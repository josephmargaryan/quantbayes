#!/usr/bin/env python3
"""Aggregate official Paper 2 nonconvex transcript attack runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
%%bash
set -euo pipefail

RESULTS=/content/quantbayes/results_paper2_nonconvex_attack_official
SUPPORT_SOURCE=public_only

DATASETS=(
  "AG News-embeddings"
  "BANKING77-embeddings"
  "CIFAR-10-embeddings"
  "DBpedia-14-embeddings"
  "Emotion-embeddings"
  "IMDb-embeddings"
  "MNIST-embeddings"
  "TREC-6-embeddings"
  "Yelp Review Full-embeddings"
)

for DS in "${DATASETS[@]}"; do
  echo "=== Paper 2 transcript attacks: $DS ==="

  python quantbayes/ball_dp/experiments/run_nonconvex_transcript_attack_experiment.py \
    --results-root "$RESULTS" \
    --dataset "$DS" \
    --radius q80 \
    --hidden-dim 128 \
    --A 4.0 \
    --Lambda 4.0 \
    --epsilon-grid 8 \
    --fixed-m 8 \
    --release-seeds 0 1 \
    --num-steps 400 \
    --batch-size 128 \
    --clip-norm 1.0 \
    --learning-rate 1e-3 \
    --mechanisms both \
    --support-source "$SUPPORT_SOURCE" \
    --support-selection farthest \
    --anchor-selection rare_class \
    --num-supports 4 \
    --support-draws 2 \
    --targets-per-support 1 \
    --max-feasible-search 5000 \
    --strict-feasible-supports \
    --attack-modes known_inclusion unknown_inclusion \
    --known-step-mode all \
    --trace-capture-every 1
done

python quantbayes/ball_dp/experiments/aggregate_nonconvex_attack.py \
  --results-root "$RESULTS"
"""


ATTACK_FILENAME = "nonconvex_transcript_attack_rows.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Paper 2 nonconvex transcript finite-prior attack results."
    )
    parser.add_argument(
        "--results-root", type=str, default="results_paper2_nonconvex_attack"
    )
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
        path = d / ATTACK_FILENAME
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
            "attack_mode",
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
            "exact_identification_success",
            "prior_exact_hit",
            "prior_rank",
            "prior_hit@1",
            "prior_hit@5",
            "prior_hit@10",
            "mse",
            "feature_l2",
            "primary_bound_hayes",
            "bound_hayes_ball",
            "bound_hayes_standard_same_noise",
            "accuracy",
            "selected_step_count",
            "trace_steps_recorded",
            "noise_multiplier",
            "actual_epsilon_ball",
            "actual_epsilon_standard",
        ]
        if c in raw.columns
    ]
    rows = []
    for key, g in raw.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key, strict=True)}
        row["n_trials"] = int(len(g))
        ok = g.get("attack_status", pd.Series(["ok"] * len(g))).astype(str).eq("ok")
        row["n_ok"] = int(ok.sum())
        row["ok_fraction"] = float(ok.mean()) if len(ok) else float("nan")
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


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty or "epsilon" not in summary.columns:
        return
    plot_dir = output_dir / "figures"

    label_cols = [
        c for c in ["dataset_tag", "mechanism", "attack_mode"] if c in summary.columns
    ]
    fig, ax = plt.subplots()
    for key, g in summary.groupby(label_cols, dropna=False):
        g = g.sort_values("epsilon")
        x = pd.to_numeric(g["epsilon"], errors="coerce")
        y = pd.to_numeric(g.get("exact_identification_success_mean"), errors="coerce")
        b = pd.to_numeric(g.get("primary_bound_hayes_mean"), errors="coerce")
        mask_y = np.isfinite(x) & np.isfinite(y)
        mask_b = np.isfinite(x) & np.isfinite(b)
        label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
        if mask_y.any():
            ax.plot(x[mask_y], y[mask_y], marker="o", label=label)
        if mask_b.any():
            ax.plot(
                x[mask_b], b[mask_b], linestyle="--", marker="x", label=f"bound {label}"
            )
    ax.set_xlabel("epsilon")
    ax.set_ylabel("probability")
    ax.set_title("Paper 2 finite-prior transcript attacks vs Hayes bound")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    savefig_stem(fig, plot_dir / "all_attack_success_vs_bound")

    if "prior_rank_mean" in summary.columns:
        fig, ax = plt.subplots()
        for key, g in summary.groupby(label_cols, dropna=False):
            g = g.sort_values("epsilon")
            x = pd.to_numeric(g["epsilon"], errors="coerce")
            y = pd.to_numeric(g["prior_rank_mean"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
                ax.plot(x[mask], y[mask], marker="o", label=label)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("mean prior rank")
        ax.set_title("Paper 2 finite-prior attack rank")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        savefig_stem(fig, plot_dir / "all_attack_prior_rank_vs_epsilon")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    results_root = Path(args.results_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir else results_root / "aggregate"
    )
    raw = load_rows(results_root, args.datasets)
    if raw.empty:
        print(f"No {ATTACK_FILENAME} files found under {results_root}")
        return
    summary = summarize(raw)
    save_csv(raw, output_dir / "all_nonconvex_transcript_attack_rows.csv")
    save_csv(summary, output_dir / "all_nonconvex_transcript_attack_summary.csv")
    if not bool(args.no_plots):
        write_plots(summary, output_dir)
    print(
        f"Wrote {len(raw)} attack rows and {len(summary)} summary rows to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
