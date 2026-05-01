#!/usr/bin/env python3
"""Plot accuracy-vs-gamma frontiers across datasets and policy radii.

Intended use from the repository root, for example:

    python quantbayes/ball_dp/experiments/plot_accuracy_vs_gamma_radius_grid.py \
        --results-root quantbayes/results \
        --model ridge_prototype \
        --m 10

The script consumes per-dataset ERM summaries at

    <results-root>/erm/<model>/<dataset>/erm_summary.csv

and writes a two-column publication-style aggregate figure to

    <results-root>/erm/<model>/_aggregate/fig_agg_accuracy_gamma_radius_grid.{pdf,png}

plus a CSV containing all plotted points and a JSON metadata file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

RADIUS_ORDER: tuple[str, ...] = ("q50", "q80", "q95")
RADIUS_LABELS: dict[str, str] = {
    "q50": r"$r=q_{50}$",
    "q80": r"$r=q_{80}$",
    "q95": r"$r=q_{95}$",
}

# Okabe-Ito / colorblind-friendly palette.
RADIUS_COLORS: dict[str, str] = {
    "q50": "#009E73",  # green
    "q80": "#0072B2",  # blue
    "q95": "#CC79A7",  # purple
}
RADIUS_MARKERS: dict[str, str] = {
    "q50": "o",
    "q80": "s",
    "q95": "^",
}
STANDARD_COLOR = "#4D4D4D"
STANDARD_LABEL = "Standard global"

DEFAULT_DATASET_ORDER: tuple[str, ...] = (
    "ag_news",
    "banking77",
    "cifar10",
    "emotion",
    "imdb",
    "mnist",
    "trec6",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset_tag",
    "dataset_name",
    "radius_tag",
    "epsilon",
    "m",
    "accuracy_ball_mean",
    "accuracy_ball_ci_low",
    "accuracy_ball_ci_high",
    "accuracy_standard_mean",
    "accuracy_standard_ci_low",
    "accuracy_standard_ci_high",
    "bound_direct_ball_mean",
    "bound_direct_ball_ci_low",
    "bound_direct_ball_ci_high",
    "bound_direct_standard_mean",
    "bound_direct_standard_ci_low",
    "bound_direct_standard_ci_high",
)


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    dataset_dir: Path
    tag: str
    name: str
    summary: pd.DataFrame


def configure_matplotlib() -> None:
    """Use compact settings suitable for an IEEE two-column figure*."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def canonicalize_model_name(model: str) -> str:
    """Small standalone fallback matching the usual directory naming convention."""
    return str(model).strip().lower().replace("-", "_").replace(" ", "_")


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


def normalize_key(value: str) -> str:
    return (
        str(value).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    )


def clean_dataset_name(name: str) -> str:
    out = str(name)
    for suffix in ("-embeddings", "_embeddings", " embeddings"):
        if out.lower().endswith(suffix):
            out = out[: -len(suffix)]
            break
    aliases = {
        "ag_news": "AG News",
        "banking77": "BANKING77",
        "cifar10": "CIFAR-10",
        "trec6": "TREC-6",
        "imdb": "IMDb",
        "mnist": "MNIST",
        "emotion": "Emotion",
    }
    return aliases.get(normalize_key(out), out)


def validate_summary(summary: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in summary.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def collapse_duplicate_eps_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric duplicate epsilon rows; keep the first non-null metadata value."""
    if df.empty:
        return df.copy()
    agg: dict[str, str | Any] = {}
    for col in df.columns:
        if col == "epsilon":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "mean"
        else:
            agg[col] = first_non_null
    return df.groupby("epsilon", as_index=False, sort=True).agg(agg)


def read_dataset_summary(summary_path: Path) -> DatasetSummary:
    summary = pd.read_csv(summary_path)
    validate_summary(summary, summary_path)
    if summary.empty:
        raise ValueError(f"Empty summary file: {summary_path}")
    tag = str(first_non_null(summary["dataset_tag"]))
    name = str(first_non_null(summary["dataset_name"]))
    return DatasetSummary(
        path=summary_path,
        dataset_dir=summary_path.parent,
        tag=tag,
        name=name,
        summary=summary,
    )


def discover_summaries(
    model_dir: Path,
    dataset_queries: Sequence[str] | None,
) -> list[DatasetSummary]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model results directory not found: {model_dir}")

    all_paths = sorted(
        p / "erm_summary.csv"
        for p in model_dir.iterdir()
        if p.is_dir() and p.name != "_aggregate" and (p / "erm_summary.csv").exists()
    )
    if not all_paths:
        raise FileNotFoundError(
            f"No per-dataset erm_summary.csv files found under {model_dir}"
        )

    summaries = [read_dataset_summary(path) for path in all_paths]

    if dataset_queries:
        query_keys = {normalize_key(q) for q in dataset_queries}
        selected: list[DatasetSummary] = []
        for item in summaries:
            keys = {
                normalize_key(item.dataset_dir.name),
                normalize_key(item.tag),
                normalize_key(item.name),
                normalize_key(clean_dataset_name(item.name)),
            }
            if keys & query_keys:
                selected.append(item)
        found_keys = {
            key
            for item in selected
            for key in (
                normalize_key(item.dataset_dir.name),
                normalize_key(item.tag),
                normalize_key(item.name),
                normalize_key(clean_dataset_name(item.name)),
            )
        }
        missing_queries = sorted(q for q in query_keys if q not in found_keys)
        if missing_queries:
            available = ", ".join(sorted(item.dataset_dir.name for item in summaries))
            raise FileNotFoundError(
                f"Could not match dataset queries {missing_queries}. Available: {available}"
            )
        summaries = selected

    order = {key: i for i, key in enumerate(DEFAULT_DATASET_ORDER)}

    def sort_key(item: DatasetSummary) -> tuple[int, str]:
        keys = [
            normalize_key(item.tag),
            normalize_key(item.dataset_dir.name),
            normalize_key(item.name),
        ]
        rank = min((order[k] for k in keys if k in order), default=len(order) + 1)
        return rank, clean_dataset_name(item.name).lower()

    return sorted(summaries, key=sort_key)


def filter_radius_curve(summary: pd.DataFrame, *, radius: str, m: int) -> pd.DataFrame:
    mask = (summary["radius_tag"].astype(str) == str(radius)) & np.isclose(
        summary["m"].astype(float), float(m)
    )
    df = summary.loc[mask].copy()
    if df.empty:
        return df
    df = collapse_duplicate_eps_rows(df)
    return df.sort_values("epsilon").reset_index(drop=True)


def filter_standard_curve(
    summary: pd.DataFrame,
    *,
    m: int,
    preferred_radius: str,
    radii: Sequence[str],
) -> pd.DataFrame:
    """Return one standard-comparator row per epsilon.

    The standard release is independent of the local radius in the ERM driver, but
    rows are repeated because the table is indexed by radius. Prefer q80 (or the
    requested preferred radius) when duplicate rows are present.
    """
    mask = np.isclose(summary["m"].astype(float), float(m))
    df = summary.loc[mask].copy()
    if df.empty:
        return df

    priority: dict[str, int] = {str(preferred_radius): 0}
    for i, radius in enumerate(radii, start=1):
        priority.setdefault(str(radius), i)
    priority.setdefault("q80", len(priority) + 1)
    priority.setdefault("q50", len(priority) + 2)
    priority.setdefault("q95", len(priority) + 3)

    df["_radius_priority"] = (
        df["radius_tag"].astype(str).map(lambda r: priority.get(r, len(priority) + 10))
    )
    df = df.sort_values(["epsilon", "_radius_priority"]).drop_duplicates(
        subset=["epsilon"], keep="first"
    )
    df = df.drop(columns=["_radius_priority"])
    return collapse_duplicate_eps_rows(df).sort_values("epsilon").reset_index(drop=True)


def restrict_to_common_eps(curves: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    nonempty_eps_sets = [
        set(map(float, df["epsilon"])) for df in curves.values() if not df.empty
    ]
    if not nonempty_eps_sets:
        return curves
    common = set.intersection(*nonempty_eps_sets)
    return {
        radius: df[df["epsilon"].astype(float).isin(common)].reset_index(drop=True)
        for radius, df in curves.items()
    }


def check_complete_radius_grid(
    datasets: Sequence[DatasetSummary],
    *,
    radii: Sequence[str],
    m: int,
    preferred_standard_radius: str,
    expected_epsilon_grid: Sequence[float] | None,
) -> list[str]:
    """Return human-readable completeness failures for the requested grid."""
    failures: list[str] = []

    def _close_contains(values: Sequence[float], target: float) -> bool:
        return any(
            np.isclose(float(v), float(target), rtol=1e-10, atol=1e-12) for v in values
        )

    for item in datasets:
        standard = filter_standard_curve(
            item.summary, m=m, preferred_radius=preferred_standard_radius, radii=radii
        )
        if expected_epsilon_grid is None:
            expected = sorted(float(v) for v in standard["epsilon"].tolist())
            if not expected:
                union: set[float] = set()
                for radius in radii:
                    curve = filter_radius_curve(item.summary, radius=radius, m=m)
                    union.update(float(v) for v in curve["epsilon"].tolist())
                expected = sorted(union)
        else:
            expected = sorted(float(v) for v in expected_epsilon_grid)

        if not expected:
            failures.append(f"{item.tag}: no epsilon grid available at m={m}")
            continue

        std_eps = [float(v) for v in standard["epsilon"].tolist()]
        std_missing = [eps for eps in expected if not _close_contains(std_eps, eps)]
        if std_missing:
            failures.append(
                f"{item.tag}: standard comparator missing eps={std_missing} at m={m}"
            )

        for radius in radii:
            curve = filter_radius_curve(item.summary, radius=radius, m=m)
            got = [float(v) for v in curve["epsilon"].tolist()]
            missing = [eps for eps in expected if not _close_contains(got, eps)]
            if missing:
                failures.append(
                    f"{item.tag}:{radius} missing eps={missing}; present={got}"
                )
    return failures


def y_limits_from_values(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if math.isclose(lo, hi):
        pad = 0.025
    else:
        pad = max(0.015, 0.08 * (hi - lo))
    return max(0.0, lo - pad), min(1.0, hi + pad)


def x_limits_from_values(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if math.isclose(lo, hi):
        pad = max(0.01, 0.05 * abs(lo))
    else:
        pad = max(0.01, 0.04 * (hi - lo))
    return max(0.0, lo - pad), min(1.0, hi + pad)


def plot_accuracy_gamma_radius_grid(
    datasets: Sequence[DatasetSummary],
    *,
    m: int,
    radii: Sequence[str],
    preferred_standard_radius: str,
    out_stem: Path,
    ncols: int,
    include_ci: bool,
    connect_same_epsilon_radii: bool,
    common_epsilon_only: bool,
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("No datasets to plot.")

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(datasets) / ncols))
    fig_width = 7.25
    fig_height = max(3.4, 1.85 * nrows + 0.65)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), squeeze=False
    )
    flat_axes = list(axes.ravel())

    all_x: list[float] = []
    plotted_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    # First pass: collect x-limits and plotted-point CSV rows.
    dataset_payloads: list[dict[str, Any]] = []
    for item in datasets:
        curves = {
            radius: filter_radius_curve(item.summary, radius=radius, m=m)
            for radius in radii
        }
        if common_epsilon_only:
            curves = restrict_to_common_eps(curves)
        standard = filter_standard_curve(
            item.summary, m=m, preferred_radius=preferred_standard_radius, radii=radii
        )
        if common_epsilon_only:
            eps_sets = [
                set(map(float, df["epsilon"])) for df in curves.values() if not df.empty
            ]
            if eps_sets:
                common = set.intersection(*eps_sets)
                standard = standard[
                    standard["epsilon"].astype(float).isin(common)
                ].reset_index(drop=True)

        for radius, df in curves.items():
            if not df.empty:
                all_x.extend(df["bound_direct_ball_mean"].astype(float).tolist())
                for row in df.itertuples(index=False):
                    plotted_rows.append(
                        {
                            "dataset_tag": item.tag,
                            "dataset_name": item.name,
                            "series": f"Ball-DP {radius}",
                            "radius_tag": radius,
                            "epsilon": float(row.epsilon),
                            "m": int(round(float(row.m))),
                            "gamma_mean": float(row.bound_direct_ball_mean),
                            "gamma_ci_low": float(row.bound_direct_ball_ci_low),
                            "gamma_ci_high": float(row.bound_direct_ball_ci_high),
                            "accuracy_mean": float(row.accuracy_ball_mean),
                            "accuracy_ci_low": float(row.accuracy_ball_ci_low),
                            "accuracy_ci_high": float(row.accuracy_ball_ci_high),
                        }
                    )
        if not standard.empty:
            all_x.extend(standard["bound_direct_standard_mean"].astype(float).tolist())
            for row in standard.itertuples(index=False):
                plotted_rows.append(
                    {
                        "dataset_tag": item.tag,
                        "dataset_name": item.name,
                        "series": STANDARD_LABEL,
                        "radius_tag": "global",
                        "epsilon": float(row.epsilon),
                        "m": int(round(float(row.m))),
                        "gamma_mean": float(row.bound_direct_standard_mean),
                        "gamma_ci_low": float(row.bound_direct_standard_ci_low),
                        "gamma_ci_high": float(row.bound_direct_standard_ci_high),
                        "accuracy_mean": float(row.accuracy_standard_mean),
                        "accuracy_ci_low": float(row.accuracy_standard_ci_low),
                        "accuracy_ci_high": float(row.accuracy_standard_ci_high),
                    }
                )

        radius_eps = {
            radius: [float(x) for x in curves[radius]["epsilon"].tolist()]
            for radius in radii
        }
        coverage[item.tag] = {
            "dataset_name": item.name,
            "radius_epsilon_grid": radius_eps,
            "standard_epsilon_grid": [float(x) for x in standard["epsilon"].tolist()],
            "missing_radii": [radius for radius, eps in radius_eps.items() if not eps],
            "single_point_radii": [
                radius for radius, eps in radius_eps.items() if len(eps) == 1
            ],
        }
        dataset_payloads.append({"item": item, "curves": curves, "standard": standard})

    xlim = x_limits_from_values(all_x)

    for ax, payload in zip(flat_axes, dataset_payloads, strict=False):
        item = payload["item"]
        curves: dict[str, pd.DataFrame] = payload["curves"]
        standard: pd.DataFrame = payload["standard"]

        panel_y_values: list[float] = []

        if connect_same_epsilon_radii:
            eps_union = sorted(
                {
                    float(eps)
                    for df in curves.values()
                    if not df.empty
                    for eps in df["epsilon"].tolist()
                }
            )
            for eps in eps_union:
                points: list[tuple[float, float]] = []
                for radius in radii:
                    df = curves[radius]
                    if df.empty:
                        continue
                    rows = df[np.isclose(df["epsilon"].astype(float), float(eps))]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    points.append(
                        (
                            float(row["bound_direct_ball_mean"]),
                            float(row["accuracy_ball_mean"]),
                        )
                    )
                if len(points) >= 2:
                    x = float(np.mean([p[0] for p in points]))
                    ys = [p[1] for p in points]
                    ax.vlines(
                        x,
                        min(ys),
                        max(ys),
                        color="0.72",
                        linewidth=0.8,
                        alpha=0.75,
                        zorder=0,
                    )

        for radius in radii:
            df = curves[radius]
            if df.empty:
                continue
            x = df["bound_direct_ball_mean"].astype(float).to_numpy()
            y = df["accuracy_ball_mean"].astype(float).to_numpy()
            ylo = df["accuracy_ball_ci_low"].astype(float).to_numpy()
            yhi = df["accuracy_ball_ci_high"].astype(float).to_numpy()
            panel_y_values.extend(y.tolist())
            if include_ci:
                panel_y_values.extend(ylo.tolist())
                panel_y_values.extend(yhi.tolist())
                ax.fill_between(
                    x,
                    ylo,
                    yhi,
                    color=RADIUS_COLORS.get(radius, "#000000"),
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x,
                y,
                color=RADIUS_COLORS.get(radius, "#000000"),
                marker=RADIUS_MARKERS.get(radius, "o"),
                markersize=3.8,
                linewidth=1.45,
                label=RADIUS_LABELS.get(radius, radius),
                zorder=3,
            )

        if not standard.empty:
            x = standard["bound_direct_standard_mean"].astype(float).to_numpy()
            y = standard["accuracy_standard_mean"].astype(float).to_numpy()
            ylo = standard["accuracy_standard_ci_low"].astype(float).to_numpy()
            yhi = standard["accuracy_standard_ci_high"].astype(float).to_numpy()
            panel_y_values.extend(y.tolist())
            if include_ci:
                panel_y_values.extend(ylo.tolist())
                panel_y_values.extend(yhi.tolist())
                ax.fill_between(
                    x,
                    ylo,
                    yhi,
                    color=STANDARD_COLOR,
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                x,
                y,
                color=STANDARD_COLOR,
                linestyle="--",
                marker="D",
                markersize=3.3,
                linewidth=1.25,
                label=STANDARD_LABEL,
                zorder=2,
            )

        ax.set_title(clean_dataset_name(item.name), pad=2.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*y_limits_from_values(panel_y_values))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", which="major", pad=1.5)

    for ax in flat_axes[len(dataset_payloads) :]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            color=RADIUS_COLORS.get(radius, "#000000"),
            marker=RADIUS_MARKERS.get(radius, "o"),
            linewidth=1.6,
            markersize=4.2,
            label=RADIUS_LABELS.get(radius, radius),
        )
        for radius in radii
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=STANDARD_COLOR,
            linestyle="--",
            marker="D",
            linewidth=1.35,
            markersize=3.8,
            label=STANDARD_LABEL,
        )
    )
    if connect_same_epsilon_radii:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.72",
                linewidth=0.9,
                label=r"same $\varepsilon$ radius span",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(handles), 5),
        handlelength=2.1,
        columnspacing=1.1,
    )
    fig.text(
        0.5,
        0.035,
        r"Direct exact-identification upper bound $\gamma$ (matched privacy)",
        ha="center",
        va="center",
        fontsize=8.8,
    )
    fig.text(
        0.012,
        0.51,
        "Test accuracy",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.8,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.09,
        top=0.895,
        wspace=0.28,
        hspace=0.38,
    )

    savefig_stem(fig, out_stem)

    points_path = out_stem.parent / "accuracy_gamma_radius_grid_points.csv"
    pd.DataFrame(plotted_rows).to_csv(points_path, index=False)

    metadata: dict[str, Any] = {
        "output_figure_stem": str(out_stem),
        "points_csv": str(points_path),
        "m": int(m),
        "radii": list(radii),
        "preferred_standard_radius": preferred_standard_radius,
        "common_epsilon_only": bool(common_epsilon_only),
        "include_ci": bool(include_ci),
        "connect_same_epsilon_radii": bool(connect_same_epsilon_radii),
        "coverage": coverage,
        "interpretation_note": (
            "Under matched Gaussian calibration, direct gamma often has the same numerical "
            "x-coordinate across local radii at fixed epsilon and m because sigma scales with "
            "the corresponding sensitivity. The radius labels therefore encode the adjacency "
            "scope; vertical spans at a fixed gamma/epsilon show the utility cost of enlarging "
            "the protected neighborhood."
        ),
    }
    metadata_path = out_stem.parent / "accuracy_gamma_radius_grid_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    metadata["metadata_json"] = str(metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-style grid of accuracy-vs-gamma frontiers across datasets "
            "and local policy radii from existing ERM summary CSVs."
        )
    )
    parser.add_argument("--results-root", type=str, default="quantbayes/results")
    parser.add_argument("--model", type=str, default="ridge_prototype")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--radii", nargs="+", type=str, default=list(RADIUS_ORDER))
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=None,
        help="Optional dataset subset, e.g. ag_news banking77 imdb. Default: all summaries found.",
    )
    parser.add_argument(
        "--preferred-standard-radius",
        type=str,
        default="q80",
        help="Which duplicated radius row to use for the standard comparator curve.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable accuracy confidence ribbons. Useful if the figure is too visually dense.",
    )
    parser.add_argument(
        "--no-radius-spans",
        action="store_true",
        help="Disable thin vertical segments connecting radii at the same epsilon.",
    )
    parser.add_argument(
        "--common-epsilon-only",
        action="store_true",
        help=(
            "Restrict every radius curve to epsilons available for all requested radii within a dataset. "
            "Leave off when you want the script to show all available points and report coverage."
        ),
    )
    parser.add_argument(
        "--expected-epsilon-grid",
        nargs="+",
        type=float,
        default=None,
        help="Expected epsilon grid for completeness checks. Default: use the standard comparator grid per dataset.",
    )
    parser.add_argument(
        "--require-complete-radius-grid",
        action="store_true",
        help="Fail unless every requested radius has every expected epsilon at the chosen m.",
    )
    parser.add_argument(
        "--require-at-least-two-points-per-radius",
        action="store_true",
        help="Fail if any requested dataset/radius has fewer than two epsilon points at the chosen m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    model_family = canonicalize_model_name(args.model)
    results_root = Path(args.results_root)
    model_dir = results_root / "erm" / model_family
    output_dir = (
        ensure_dir(args.output_dir)
        if args.output_dir is not None
        else ensure_dir(model_dir / "_aggregate")
    )
    out_stem = output_dir / "fig_agg_accuracy_gamma_radius_grid"

    radii = [str(r) for r in args.radii]
    unknown_radii = [r for r in radii if r not in RADIUS_ORDER]
    if unknown_radii:
        print(
            f"Warning: unknown radius tags {unknown_radii}; they will be plotted with fallback styling.",
            file=sys.stderr,
        )

    datasets = discover_summaries(model_dir, args.datasets)

    if args.require_complete_radius_grid:
        failures = check_complete_radius_grid(
            datasets,
            radii=radii,
            m=int(args.m),
            preferred_standard_radius=str(args.preferred_standard_radius),
            expected_epsilon_grid=args.expected_epsilon_grid,
        )
        if failures:
            raise RuntimeError(
                "Requested a complete radius/epsilon grid, but some entries are missing: "
                + "; ".join(failures)
            )

    if args.require_at_least_two_points_per_radius:
        failures: list[str] = []
        for item in datasets:
            for radius in radii:
                curve = filter_radius_curve(item.summary, radius=radius, m=int(args.m))
                if len(curve) < 2:
                    failures.append(f"{item.tag}:{radius} has {len(curve)} point(s)")
        if failures:
            raise RuntimeError(
                "Requested full radius curves, but some dataset/radius pairs are incomplete: "
                + "; ".join(failures)
            )

    metadata = plot_accuracy_gamma_radius_grid(
        datasets,
        m=int(args.m),
        radii=radii,
        preferred_standard_radius=str(args.preferred_standard_radius),
        out_stem=out_stem,
        ncols=int(args.ncols),
        include_ci=not bool(args.no_ci),
        connect_same_epsilon_radii=not bool(args.no_radius_spans),
        common_epsilon_only=bool(args.common_epsilon_only),
    )

    metadata["require_complete_radius_grid"] = bool(args.require_complete_radius_grid)
    metadata["expected_epsilon_grid"] = (
        [float(v) for v in args.expected_epsilon_grid]
        if args.expected_epsilon_grid is not None
        else None
    )
    metadata_path = Path(
        metadata.get(
            "metadata_json", output_dir / "accuracy_gamma_radius_grid_metadata.json"
        )
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
