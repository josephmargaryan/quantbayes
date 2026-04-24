#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.serialization import save_dataframe

BALL_COLOR = "#1f77b4"
STANDARD_COLOR = "#d62728"
MODEL_ALIASES = {
    "ridge_prototype": "ridge_prototype",
    "ridge-prototype": "ridge_prototype",
    "softmax_logistic": "softmax_logistic",
    "softmax-logistic": "softmax_logistic",
    "binary_logistic": "binary_logistic",
    "binary-logistic": "binary_logistic",
    "sqr_hinge_svm": "squared_hinge",
    "squared_hinge": "squared_hinge",
    "squared-hinge": "squared_hinge",
    "svm": "squared_hinge",
}


def canonicalize_model(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIASES:
        supported = ", ".join(sorted(set(MODEL_ALIASES.values())))
        raise ValueError(
            f"Unsupported model {name!r}. Supported model directories: {supported}"
        )
    return MODEL_ALIASES[key]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.2, 5.4),
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig_stem(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def resolve_dataset_dir(model_dir: Path, dataset_query: str) -> Path:
    query = str(dataset_query).strip().lower().replace(" ", "_")
    candidates: list[tuple[Path, set[str]]] = []
    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        summary_path = dataset_dir / "erm_summary.csv"
        keys = {
            dataset_dir.name.lower().replace(" ", "_"),
            dataset_dir.name.lower().replace("-", "_").replace(" ", "_"),
        }
        if summary_path.exists():
            summary = pd.read_csv(summary_path, nrows=1)
            if not summary.empty:
                dataset_tag = (
                    str(summary.loc[0, "dataset_tag"]).lower().replace(" ", "_")
                )
                dataset_name = (
                    str(summary.loc[0, "dataset_name"]).lower().replace(" ", "_")
                )
                keys.add(dataset_tag)
                keys.add(dataset_name)
                keys.add(dataset_name.replace("-", "_"))
        candidates.append((dataset_dir, keys))

    matches = [path for path, keys in candidates if query in keys]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        supported = ", ".join(sorted(path.name for path, _ in candidates))
        raise FileNotFoundError(
            f"Could not find dataset {dataset_query!r} under {model_dir}. Available dataset directories: {supported}"
        )
    names = ", ".join(str(p.name) for p in matches)
    raise RuntimeError(f"Ambiguous dataset query {dataset_query!r}. Matches: {names}")


def symmetric_error(mean: pd.Series, low: pd.Series, high: pd.Series) -> np.ndarray:
    return np.vstack(
        [
            mean.to_numpy(dtype=float) - low.to_numpy(dtype=float),
            high.to_numpy(dtype=float) - mean.to_numpy(dtype=float),
        ]
    )


def build_plot_dataframe(
    summary: pd.DataFrame, *, radius_tag: str, m: int
) -> pd.DataFrame:
    mask = (summary["radius_tag"] == str(radius_tag)) & np.isclose(
        summary["m"].astype(float), float(m)
    )
    plot_df = summary.loc[mask].copy()
    if plot_df.empty:
        raise RuntimeError(
            f"No rows matched radius_tag={radius_tag!r} and m={m}. Available radius tags: "
            f"{sorted(summary['radius_tag'].astype(str).unique())}; available m values: {sorted(summary['m'].astype(int).unique())}"
        )
    plot_df = plot_df.sort_values("epsilon").reset_index(drop=True)
    return plot_df


def make_frontier_points(plot_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epsilon": plot_df["epsilon"],
            "radius_tag": plot_df["radius_tag"],
            "m": plot_df["m"],
            "gamma_ball_matched": plot_df["bound_direct_ball_mean"],
            "gamma_ball_matched_ci_low": plot_df["bound_direct_ball_ci_low"],
            "gamma_ball_matched_ci_high": plot_df["bound_direct_ball_ci_high"],
            "gamma_standard_matched": plot_df["bound_direct_standard_mean"],
            "gamma_standard_matched_ci_low": plot_df["bound_direct_standard_ci_low"],
            "gamma_standard_matched_ci_high": plot_df["bound_direct_standard_ci_high"],
            "accuracy_ball": plot_df["accuracy_ball_mean"],
            "accuracy_ball_ci_low": plot_df["accuracy_ball_ci_low"],
            "accuracy_ball_ci_high": plot_df["accuracy_ball_ci_high"],
            "accuracy_standard": plot_df["accuracy_standard_mean"],
            "accuracy_standard_ci_low": plot_df["accuracy_standard_ci_low"],
            "accuracy_standard_ci_high": plot_df["accuracy_standard_ci_high"],
            "sigma_ball": plot_df["sigma_ball_mean"],
            "sigma_standard": plot_df["sigma_standard_mean"],
        }
    )


def plot_accuracy_vs_gamma(
    frontier_df: pd.DataFrame,
    *,
    dataset_name: str,
    model_family: str,
    radius_tag: str,
    m: int,
    out_stem: Path,
) -> None:
    fig, ax = plt.subplots()

    x_ball = frontier_df["gamma_ball_matched"]
    y_ball = frontier_df["accuracy_ball"]
    x_std = frontier_df["gamma_standard_matched"]
    y_std = frontier_df["accuracy_standard"]

    ax.errorbar(
        x_ball,
        y_ball,
        xerr=symmetric_error(
            frontier_df["gamma_ball_matched"],
            frontier_df["gamma_ball_matched_ci_low"],
            frontier_df["gamma_ball_matched_ci_high"],
        ),
        yerr=symmetric_error(
            frontier_df["accuracy_ball"],
            frontier_df["accuracy_ball_ci_low"],
            frontier_df["accuracy_ball_ci_high"],
        ),
        fmt="o-",
        color=BALL_COLOR,
        capsize=3,
        label="Ball-DP (matched privacy)",
    )
    ax.errorbar(
        x_std,
        y_std,
        xerr=symmetric_error(
            frontier_df["gamma_standard_matched"],
            frontier_df["gamma_standard_matched_ci_low"],
            frontier_df["gamma_standard_matched_ci_high"],
        ),
        yerr=symmetric_error(
            frontier_df["accuracy_standard"],
            frontier_df["accuracy_standard_ci_low"],
            frontier_df["accuracy_standard_ci_high"],
        ),
        fmt="s-",
        color=STANDARD_COLOR,
        capsize=3,
        label="Standard DP (matched privacy)",
    )

    for row in frontier_df.itertuples(index=False):
        ax.annotate(
            f"ε={float(row.epsilon):g}",
            (float(row.gamma_ball_matched), float(row.accuracy_ball)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            color=BALL_COLOR,
        )

    ax.set_xlabel(r"Direct exact-ID upper bound $\gamma$")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Accuracy vs direct exact-ID upper bound\n{dataset_name} · {model_family}, radius={radius_tag}, m={m}"
    )
    ax.legend(loc="best")
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5,
        -0.04,
        "Each point uses the same mechanism on both axes: matched-privacy direct Gaussian γ on x, matched-privacy accuracy on y. Same-noise comparators are intentionally excluded.",
        ha="center",
        va="top",
        fontsize=9,
    )
    savefig_stem(fig, out_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a matched-privacy accuracy-vs-gamma frontier for one dataset/model using the per-dataset ERM summary."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--radius", type=str, default="q80")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    model_family = canonicalize_model(args.model)
    results_root = Path(args.results_root)
    model_dir = results_root / "erm" / model_family
    if not model_dir.exists():
        raise FileNotFoundError(f"Model results directory not found: {model_dir}")

    dataset_dir = resolve_dataset_dir(model_dir, args.dataset)
    summary_path = dataset_dir / "erm_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"ERM summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    plot_df = build_plot_dataframe(summary, radius_tag=str(args.radius), m=int(args.m))
    frontier_df = make_frontier_points(plot_df)

    dataset_name = str(plot_df.iloc[0]["dataset_name"])
    output_dir = (
        ensure_dir(args.output_dir)
        if args.output_dir is not None
        else ensure_dir(dataset_dir / "figures")
    )
    stem = (
        output_dir
        / f"fig_accuracy_vs_gamma_{str(plot_df.iloc[0]['dataset_tag'])}_{model_family}"
    )

    save_dataframe(
        frontier_df,
        output_dir
        / f"accuracy_vs_gamma_points_{str(plot_df.iloc[0]['dataset_tag'])}_{model_family}.csv",
        save_parquet_if_possible=False,
    )
    metadata: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_name,
        "dataset_tag": str(plot_df.iloc[0]["dataset_tag"]),
        "model_family": model_family,
        "radius_tag": str(args.radius),
        "m": int(args.m),
        "x_axis": "matched-privacy direct Gaussian exact-identification upper bound",
        "y_axis": "matched-privacy predictive accuracy",
        "excluded_comparator": "same-noise standard comparator",
        "source_summary": str(summary_path),
        "output_figure_stem": str(stem),
    }
    (
        output_dir
        / f"accuracy_vs_gamma_metadata_{str(plot_df.iloc[0]['dataset_tag'])}_{model_family}.json"
    ).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    plot_accuracy_vs_gamma(
        frontier_df,
        dataset_name=dataset_name,
        model_family=model_family,
        radius_tag=str(args.radius),
        m=int(args.m),
        out_stem=stem,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
