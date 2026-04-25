#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp import (
    ball_rero,
    fit_convex,
    make_finite_identification_prior,
    summarize_embedding_ball_radii,
)
from quantbayes.ball_dp.serialization import save_dataframe
from quantbayes.ball_dp.experiments.run_attack_experiment import (
    DEFAULT_DELTA,
    DEFAULT_EMBEDDING_BOUND,
    DEFAULT_EPS_GRID,
    DEFAULT_FIXED_EPSILON,
    DEFAULT_FIXED_M,
    DEFAULT_LAM,
    DEFAULT_M_GRID,
    DEFAULT_MAX_ITER,
    DEFAULT_ORDERS,
    actual_ball_epsilon,
    actual_standard_epsilon_same_noise,
    canonicalize_model,
    configure_matplotlib,
    ensure_dir,
    load_embeddings,
    resolve_dataset,
    radius_value_from_report,
    savefig_stem,
    validate_model_dataset_compatibility,
    write_json,
)

RADIUS_ORDER = ("q50", "q80", "q95")
RADIUS_POSITION = {tag: i for i, tag in enumerate(RADIUS_ORDER)}

BALL_COLOR = "#0072B2"
STANDARD_COLOR = "#D55E00"
BASELINE_COLOR = "#4D4D4D"
RDP_COLOR = "#009E73"

LABEL_BALL_DP = "Ball-DP"
LABEL_STANDARD_DP = "Standard DP"
LABEL_BALL_DIRECT = "Ball direct"
LABEL_BALL_RDP = "Ball RDP"
LABEL_STANDARD_MATCHED = r"Standard direct (matched $\varepsilon$)"
LABEL_STANDARD_SAME_NOISE = r"Standard direct @ same $\sigma$ as Ball"

MATCHED_DIRECT_IDENTITY_NOTE = "Std direct at matched privacy equals Ball direct under Gaussian calibration; omitted."

SERIES_STYLES: dict[str, dict[str, Any]] = {
    LABEL_BALL_DP: {
        "color": BALL_COLOR,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
    },
    LABEL_STANDARD_DP: {
        "color": STANDARD_COLOR,
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.2,
    },
    LABEL_BALL_DIRECT: {
        "color": BALL_COLOR,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
    },
    LABEL_BALL_RDP: {
        "color": RDP_COLOR,
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 2.2,
    },
    LABEL_STANDARD_MATCHED: {
        "color": STANDARD_COLOR,
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.2,
    },
    LABEL_STANDARD_SAME_NOISE: {
        "color": BASELINE_COLOR,
        "linestyle": "--",
        "marker": "D",
        "linewidth": 2.2,
    },
}

DEFAULT_SERIES_STYLE: dict[str, Any] = {
    "color": "#000000",
    "linestyle": "-",
    "marker": "o",
    "linewidth": 2.2,
}


def get_series_style(label: str) -> dict[str, Any]:
    return dict(SERIES_STYLES.get(label, DEFAULT_SERIES_STYLE))


def fit_release(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_family: str,
    release_radius_value: float,
    epsilon: float,
    delta: float,
    lam: float,
    embedding_bound: float,
    num_classes: int,
    max_iter: int,
    solver: str,
    seed: int,
) -> Any:
    return fit_convex(
        X_train,
        y_train,
        X_eval=X_test,
        y_eval=y_test,
        model_family=model_family,
        privacy="ball_dp",
        radius=float(release_radius_value),
        lam=float(lam),
        epsilon=float(epsilon),
        delta=float(delta),
        embedding_bound=float(embedding_bound),
        num_classes=int(num_classes),
        orders=tuple(float(v) for v in DEFAULT_ORDERS),
        max_iter=int(max_iter),
        solver=str(solver),
        seed=int(seed),
    )


def make_finite_prior_for_m(m: int, feature_dim: int) -> Any:
    samples = np.zeros((int(m), int(feature_dim)), dtype=np.float32)
    return make_finite_identification_prior(samples, weights=None)


def get_rero_scalar(report: Any) -> float:
    return float(report.points[0].gamma_ball)


def summarize_mean_std(arr: Sequence[float]) -> tuple[float, float, float, float]:
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, 0.0, mean, mean
    std = float(np.std(vals, ddof=1))
    se = std / math.sqrt(float(vals.size))
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, std, lo, hi


def rebuild_erm_dataset_outputs(
    dataset_dir: Path,
    *,
    dataset_name: str,
    dataset_tag: str,
    model_family: str,
    fixed_radius_tag: str,
    fixed_epsilon: float,
    fixed_m: int,
) -> None:
    run_dirs = sorted((dataset_dir / "runs").glob("*"))
    raw_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "erm_seed_rows.csv"
        if path.exists():
            raw_frames.append(pd.read_csv(path))
    if not raw_frames:
        return

    raw_df = pd.concat(raw_frames, ignore_index=True)
    if "row_key" in raw_df.columns:
        raw_df = raw_df.drop_duplicates(subset=["row_key"], keep="last")
    save_dataframe(
        raw_df, dataset_dir / "erm_seed_rows.csv", save_parquet_if_possible=False
    )

    config_cols = [
        "dataset_tag",
        "dataset_name",
        "model_family",
        "radius_tag",
        "radius_value",
        "epsilon",
        "m",
        "delta",
        "lam",
        "embedding_bound",
        "solver",
        "max_iter",
    ]
    grouped = raw_df.groupby(config_cols, dropna=False)

    rows: list[dict[str, Any]] = []
    metric_cols = [
        "accuracy_ball",
        "accuracy_standard",
        "sigma_ball",
        "sigma_standard",
        "actual_epsilon_ball",
        "actual_epsilon_standard",
        "same_noise_standard_epsilon_from_ball",
        "bound_generic_dp_ball",
        "bound_generic_dp_standard",
        "bound_direct_ball",
        "bound_direct_standard",
        "bound_rdp_ball",
        "bound_rdp_standard",
        "bound_same_noise_standard_from_ball",
        "sensitivity_ball",
        "sensitivity_standard",
        "sigma_ratio_standard_over_ball",
        "accuracy_gain_ball_minus_standard",
        "bound_gap_direct_minus_rdp_ball",
        "bound_reduction_same_noise_pct",
    ]

    for key, frame in grouped:
        row = {col: val for col, val in zip(config_cols, key, strict=True)}
        row["n_runs"] = int(frame["run_id"].nunique())
        row["release_seeds"] = int(frame["release_seed"].nunique())
        for metric in metric_cols:
            mean, std, lo, hi = summarize_mean_std(frame[metric].astype(float).tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values(["radius_tag", "epsilon", "m"])
        .reset_index(drop=True)
    )
    save_dataframe(
        summary, dataset_dir / "erm_summary.csv", save_parquet_if_possible=False
    )

    figures_dir = ensure_dir(dataset_dir / "figures")
    dataset_title = f"{dataset_name} · {model_family}"

    def _series_equal(
        y1: np.ndarray,
        lo1: np.ndarray,
        hi1: np.ndarray,
        y2: np.ndarray,
        lo2: np.ndarray,
        hi2: np.ndarray,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-8,
    ) -> bool:
        return (
            np.allclose(y1, y2, atol=atol, rtol=rtol, equal_nan=True)
            and np.allclose(lo1, lo2, atol=atol, rtol=rtol, equal_nan=True)
            and np.allclose(hi1, hi2, atol=atol, rtol=rtol, equal_nan=True)
        )

    def plot_line(
        df: pd.DataFrame,
        *,
        x_col: str,
        x_label: str,
        y_label: str,
        y_specs: list[tuple[str, str, str, str]],
        title: str,
        stem_name: str,
        xscale: str | None = None,
        categorical_order: Sequence[str] | None = None,
        note: str | None = None,
    ) -> None:
        if df.empty:
            return

        fig, ax = plt.subplots()
        plot_df = df.copy()

        if categorical_order is not None:
            pos_map = {tag: i for i, tag in enumerate(categorical_order)}
            plot_df["_x"] = plot_df[x_col].map(pos_map)
            plot_df = plot_df.sort_values("_x")
            x_vals = plot_df["_x"].to_numpy(dtype=float)
            ax.set_xticks(range(len(categorical_order)))
            ax.set_xticklabels(list(categorical_order))
        else:
            plot_df = plot_df.sort_values(x_col)
            x_vals = plot_df[x_col].to_numpy(dtype=float)

        plotted: list[dict[str, Any]] = []
        overlap_notes: list[str] = []
        legend_handles: list[Line2D] = []

        for mean_col, low_col, high_col, label in y_specs:
            if (
                mean_col not in plot_df.columns
                or low_col not in plot_df.columns
                or high_col not in plot_df.columns
            ):
                continue

            y = plot_df[mean_col].to_numpy(dtype=float)
            lo = plot_df[low_col].to_numpy(dtype=float)
            hi = plot_df[high_col].to_numpy(dtype=float)

            if np.all(np.isnan(y)):
                continue

            style = get_series_style(label)

            overlapped_with = None
            for prev in plotted:
                if _series_equal(y, lo, hi, prev["y"], prev["lo"], prev["hi"]):
                    overlapped_with = prev["label"]
                    break

            if overlapped_with is not None:
                overlap_notes.append(f"{label} overlaps {overlapped_with}")
                continue

            ax.plot(
                x_vals,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=6,
            )
            ax.fill_between(x_vals, lo, hi, color=style["color"], alpha=0.10)

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    markersize=6,
                    label=label,
                )
            )

            plotted.append(
                {
                    "label": label,
                    "y": y,
                    "lo": lo,
                    "hi": hi,
                }
            )

        if xscale:
            finite_positive = np.isfinite(x_vals) & (x_vals > 0)
            if np.all(finite_positive):
                ax.set_xscale(xscale, base=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if legend_handles:
            ax.legend(handles=legend_handles, handlelength=3.2)

        rendered_notes: list[str] = []
        if note:
            rendered_notes.append(note)
        if overlap_notes:
            rendered_notes.append(
                "Overlapping series not redrawn:\n"
                + "\n".join(f"- {msg}" for msg in overlap_notes)
            )

        if rendered_notes:
            ax.text(
                0.01,
                0.01,
                "\n".join(rendered_notes),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

        savefig_stem(fig, figures_dir / stem_name)

    matched_privacy_bound_specs = [
        (
            "bound_direct_ball_mean",
            "bound_direct_ball_ci_low",
            "bound_direct_ball_ci_high",
            LABEL_BALL_DIRECT,
        ),
        (
            "bound_rdp_ball_mean",
            "bound_rdp_ball_ci_low",
            "bound_rdp_ball_ci_high",
            LABEL_BALL_RDP,
        ),
    ]

    same_noise_bound_specs = [
        (
            "bound_direct_ball_mean",
            "bound_direct_ball_ci_low",
            "bound_direct_ball_ci_high",
            LABEL_BALL_DIRECT,
        ),
        (
            "bound_same_noise_standard_from_ball_mean",
            "bound_same_noise_standard_from_ball_ci_low",
            "bound_same_noise_standard_from_ball_ci_high",
            LABEL_STANDARD_SAME_NOISE,
        ),
    ]

    eps_df = summary[
        (summary["radius_tag"] == fixed_radius_tag)
        & np.isclose(summary["m"], float(fixed_m))
    ]
    plot_line(
        eps_df,
        x_col="epsilon",
        x_label="$\\varepsilon$",
        y_label=r"Noise scale $\sigma$",
        y_specs=[
            (
                "sigma_ball_mean",
                "sigma_ball_ci_low",
                "sigma_ball_ci_high",
                LABEL_BALL_DP,
            ),
            (
                "sigma_standard_mean",
                "sigma_standard_ci_low",
                "sigma_standard_ci_high",
                LABEL_STANDARD_DP,
            ),
        ],
        title=f"Noise scale vs $\\varepsilon$\n{dataset_title}, radius={fixed_radius_tag}, m={fixed_m}",
        stem_name=f"fig_sigma_vs_epsilon_{dataset_tag}_{model_family}",
        xscale="log",
    )
    plot_line(
        eps_df,
        x_col="epsilon",
        x_label="$\\varepsilon$",
        y_label="Accuracy",
        y_specs=[
            (
                "accuracy_ball_mean",
                "accuracy_ball_ci_low",
                "accuracy_ball_ci_high",
                LABEL_BALL_DP,
            ),
            (
                "accuracy_standard_mean",
                "accuracy_standard_ci_low",
                "accuracy_standard_ci_high",
                LABEL_STANDARD_DP,
            ),
        ],
        title=f"Accuracy vs $\\varepsilon$\n{dataset_title}, radius={fixed_radius_tag}, m={fixed_m}",
        stem_name=f"fig_accuracy_vs_epsilon_{dataset_tag}_{model_family}",
        xscale="log",
    )
    plot_line(
        eps_df,
        x_col="epsilon",
        x_label="$\\varepsilon$",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=matched_privacy_bound_specs,
        title=(
            f"ReRo bound vs $\\varepsilon$ (matched privacy)\n"
            f"{dataset_title}, radius={fixed_radius_tag}, m={fixed_m}"
        ),
        stem_name=f"fig_bound_vs_epsilon_{dataset_tag}_{model_family}",
        xscale="log",
        note=MATCHED_DIRECT_IDENTITY_NOTE,
    )
    plot_line(
        eps_df,
        x_col="epsilon",
        x_label="$\\varepsilon$",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=same_noise_bound_specs,
        title=(
            f"ReRo bound vs $\\varepsilon$ (same noise; privacy not matched)\n"
            f"{dataset_title}, radius={fixed_radius_tag}, m={fixed_m}"
        ),
        stem_name=f"fig_bound_same_noise_vs_epsilon_{dataset_tag}_{model_family}",
        xscale="log",
    )

    m_df = summary[
        (summary["radius_tag"] == fixed_radius_tag)
        & np.isclose(summary["epsilon"], float(fixed_epsilon))
    ]
    plot_line(
        m_df,
        x_col="m",
        x_label="m",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=matched_privacy_bound_specs,
        title=(
            f"ReRo bound vs m (matched privacy)\n"
            f"{dataset_title}, radius={fixed_radius_tag}, $\\varepsilon$={fixed_epsilon:g}"
        ),
        stem_name=f"fig_bound_vs_m_{dataset_tag}_{model_family}",
        xscale="log",
        note=MATCHED_DIRECT_IDENTITY_NOTE,
    )
    plot_line(
        m_df,
        x_col="m",
        x_label="m",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=same_noise_bound_specs,
        title=(
            f"ReRo bound vs m (same noise; privacy not matched)\n"
            f"{dataset_title}, radius={fixed_radius_tag}, $\\varepsilon$={fixed_epsilon:g}"
        ),
        stem_name=f"fig_bound_same_noise_vs_m_{dataset_tag}_{model_family}",
        xscale="log",
    )

    radius_df = summary[
        np.isclose(summary["epsilon"], float(fixed_epsilon))
        & np.isclose(summary["m"], float(fixed_m))
    ].copy()
    if not radius_df.empty:
        radius_df["_radius_pos"] = radius_df["radius_tag"].map(RADIUS_POSITION)
        radius_df = radius_df.sort_values("_radius_pos")
    plot_line(
        radius_df,
        x_col="radius_tag",
        x_label="Radius quantile",
        y_label=r"Noise scale $\sigma$",
        y_specs=[
            (
                "sigma_ball_mean",
                "sigma_ball_ci_low",
                "sigma_ball_ci_high",
                LABEL_BALL_DP,
            ),
            (
                "sigma_standard_mean",
                "sigma_standard_ci_low",
                "sigma_standard_ci_high",
                LABEL_STANDARD_DP,
            ),
        ],
        title=f"Noise scale vs radius\n{dataset_title}, $\\varepsilon$={fixed_epsilon:g}, m={fixed_m}",
        stem_name=f"fig_sigma_vs_radius_{dataset_tag}_{model_family}",
        categorical_order=RADIUS_ORDER,
    )
    plot_line(
        radius_df,
        x_col="radius_tag",
        x_label="Radius quantile",
        y_label="Accuracy",
        y_specs=[
            (
                "accuracy_ball_mean",
                "accuracy_ball_ci_low",
                "accuracy_ball_ci_high",
                LABEL_BALL_DP,
            ),
            (
                "accuracy_standard_mean",
                "accuracy_standard_ci_low",
                "accuracy_standard_ci_high",
                LABEL_STANDARD_DP,
            ),
        ],
        title=f"Accuracy vs radius\n{dataset_title}, $\\varepsilon$={fixed_epsilon:g}, m={fixed_m}",
        stem_name=f"fig_accuracy_vs_radius_{dataset_tag}_{model_family}",
        categorical_order=RADIUS_ORDER,
    )
    plot_line(
        radius_df,
        x_col="radius_tag",
        x_label="Radius quantile",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=matched_privacy_bound_specs,
        title=(
            f"ReRo bound vs radius (matched privacy)\n"
            f"{dataset_title}, $\\varepsilon$={fixed_epsilon:g}, m={fixed_m}"
        ),
        stem_name=f"fig_bound_vs_radius_{dataset_tag}_{model_family}",
        categorical_order=RADIUS_ORDER,
        note=MATCHED_DIRECT_IDENTITY_NOTE,
    )
    plot_line(
        radius_df,
        x_col="radius_tag",
        x_label="Radius quantile",
        y_label=r"Exact-ID upper bound $\gamma$",
        y_specs=same_noise_bound_specs,
        title=(
            f"ReRo bound vs radius (same noise; privacy not matched)\n"
            f"{dataset_title}, $\\varepsilon$={fixed_epsilon:g}, m={fixed_m}"
        ),
        stem_name=f"fig_bound_same_noise_vs_radius_{dataset_tag}_{model_family}",
        categorical_order=RADIUS_ORDER,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run convex Ball-DP vs matched standard-DP ERM experiments over epsilon, m, and radius; "
            "save per-seed raw rows, combined dataset summaries, and publication-style figures."
        )
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--results-root", type=str, default="results")

    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--embedding-bound", type=float, default=DEFAULT_EMBEDDING_BOUND
    )
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--solver",
        choices=("lbfgs_fullbatch", "gd_fullbatch"),
        default="lbfgs_fullbatch",
    )

    parser.add_argument(
        "--epsilon-grid", nargs="+", type=float, default=list(DEFAULT_EPS_GRID)
    )
    parser.add_argument("--m-grid", nargs="+", type=int, default=list(DEFAULT_M_GRID))
    parser.add_argument(
        "--radius-grid", nargs="+", type=str, default=list(RADIUS_ORDER)
    )
    parser.add_argument("--fixed-epsilon", type=float, default=DEFAULT_FIXED_EPSILON)
    parser.add_argument("--fixed-m", type=int, default=DEFAULT_FIXED_M)
    parser.add_argument("--fixed-radius", choices=RADIUS_ORDER, default="q80")
    parser.add_argument(
        "--sweep", choices=("epsilon", "m", "radius", "both"), default="both"
    )

    parser.add_argument(
        "--release-seeds", nargs="*", type=int, default=list((0, 1, 2, 3, 4))
    )

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--embedding-cache-path", type=str, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--max-exact-pairs", type=int, default=250_000)
    parser.add_argument("--max-sampled-pairs", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    spec = resolve_dataset(args.dataset)
    model_family = canonicalize_model(args.model)
    eps_grid = sorted(set(float(v) for v in args.epsilon_grid))
    m_grid = sorted(set(int(v) for v in args.m_grid))
    radius_grid = [str(v) for v in args.radius_grid]
    invalid_radius = [tag for tag in radius_grid if tag not in RADIUS_ORDER]
    if invalid_radius:
        raise ValueError(f"Unsupported radius tags in --radius-grid: {invalid_radius}")

    data = load_embeddings(args, spec)
    validate_model_dataset_compatibility(model_family, data)
    if float(data.empirical_embedding_bound) > float(args.embedding_bound) + 1e-3:
        raise ValueError(
            f"Empirical training embedding bound {data.empirical_embedding_bound:.6f} exceeds the supplied public bound "
            f"{float(args.embedding_bound):.6f}."
        )

    radius_report = summarize_embedding_ball_radii(
        data.X_train,
        data.y_train,
        quantiles=(0.50, 0.80, 0.95),
        max_exact_pairs=int(args.max_exact_pairs),
        max_sampled_pairs=int(args.max_sampled_pairs),
        seed=0,
    )
    radius_values = {
        tag: radius_value_from_report(radius_report, tag) for tag in RADIUS_ORDER
    }

    used_configs: set[tuple[str, float, int]] = set()
    if args.sweep in {"epsilon", "both"}:
        for epsilon in eps_grid:
            used_configs.add(
                (str(args.fixed_radius), float(epsilon), int(args.fixed_m))
            )
    if args.sweep in {"m", "both"}:
        for m in m_grid:
            used_configs.add(
                (str(args.fixed_radius), float(args.fixed_epsilon), int(m))
            )
    if args.sweep in {"radius", "both"}:
        for radius_tag in radius_grid:
            used_configs.add(
                (str(radius_tag), float(args.fixed_epsilon), int(args.fixed_m))
            )
    used_configs = sorted(
        used_configs, key=lambda item: (RADIUS_POSITION[item[0]], item[1], item[2])
    )

    used_epsilons = sorted({float(eps) for _, eps, _ in used_configs})
    used_radius_tags = sorted(
        {str(tag) for tag, _, _ in used_configs}, key=lambda tag: RADIUS_POSITION[tag]
    )

    results_root = Path(args.results_root)
    dataset_dir = ensure_dir(results_root / "erm" / model_family / spec.tag)
    run_id_payload = {
        "dataset": spec.tag,
        "model": model_family,
        "delta": float(args.delta),
        "embedding_bound": float(args.embedding_bound),
        "lam": float(args.lam),
        "epsilon_grid": eps_grid,
        "m_grid": m_grid,
        "radius_grid": radius_grid,
        "fixed_epsilon": float(args.fixed_epsilon),
        "fixed_m": int(args.fixed_m),
        "fixed_radius": str(args.fixed_radius),
        "sweep": str(args.sweep),
        "used_configs": [
            {
                "radius_tag": str(radius_tag),
                "epsilon": float(epsilon),
                "m": int(m),
            }
            for radius_tag, epsilon, m in used_configs
        ],
        "release_seeds": [int(v) for v in args.release_seeds],
        "solver": str(args.solver),
        "max_iter": int(args.max_iter),
    }
    run_id = hashlib.sha1(
        json.dumps(run_id_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = ensure_dir(dataset_dir / "runs" / run_id)

    write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "dataset_name": spec.display_name,
            "dataset_tag": spec.tag,
            "model_family": model_family,
            "radius_values": radius_values,
            "used_configs": [
                {
                    "radius_tag": str(radius_tag),
                    "epsilon": float(epsilon),
                    "m": int(m),
                }
                for radius_tag, epsilon, m in used_configs
            ],
        },
    )
    write_json(run_dir / "radius_report.json", radius_report)
    write_json(
        run_dir / "dataset_metadata.json",
        {
            "dataset_name": spec.display_name,
            "dataset_tag": spec.tag,
            "num_train": int(data.X_train.shape[0]),
            "num_test": int(data.X_test.shape[0]),
            "num_classes": int(data.num_classes),
            "feature_dim": int(data.feature_dim),
            "empirical_embedding_bound": float(data.empirical_embedding_bound),
            "public_embedding_bound": float(args.embedding_bound),
        },
    )

    ball_releases: dict[tuple[str, float, int], Any] = {}
    standard_releases: dict[tuple[float, int], Any] = {}

    for epsilon in used_epsilons:
        for release_seed in [int(v) for v in args.release_seeds]:
            standard_releases[(float(epsilon), int(release_seed))] = fit_release(
                X_train=data.X_train,
                y_train=data.y_train,
                X_test=data.X_test,
                y_test=data.y_test,
                model_family=model_family,
                release_radius_value=float(2.0 * float(args.embedding_bound)),
                epsilon=float(epsilon),
                delta=float(args.delta),
                lam=float(args.lam),
                embedding_bound=float(args.embedding_bound),
                num_classes=int(data.num_classes),
                max_iter=int(args.max_iter),
                solver=str(args.solver),
                seed=int(release_seed),
            )

    for radius_tag in used_radius_tags:
        for epsilon in used_epsilons:
            for release_seed in [int(v) for v in args.release_seeds]:
                ball_releases[(str(radius_tag), float(epsilon), int(release_seed))] = (
                    fit_release(
                        X_train=data.X_train,
                        y_train=data.y_train,
                        X_test=data.X_test,
                        y_test=data.y_test,
                        model_family=model_family,
                        release_radius_value=float(radius_values[str(radius_tag)]),
                        epsilon=float(epsilon),
                        delta=float(args.delta),
                        lam=float(args.lam),
                        embedding_bound=float(args.embedding_bound),
                        num_classes=int(data.num_classes),
                        max_iter=int(args.max_iter),
                        solver=str(args.solver),
                        seed=int(release_seed),
                    )
                )

    rows: list[dict[str, Any]] = []
    for radius_tag, epsilon, m in used_configs:
        finite_prior = make_finite_prior_for_m(int(m), int(data.feature_dim))
        radius_value = float(radius_values[str(radius_tag)])

        for release_seed in [int(v) for v in args.release_seeds]:
            ball_release = ball_releases[
                (str(radius_tag), float(epsilon), int(release_seed))
            ]
            standard_release = standard_releases[(float(epsilon), int(release_seed))]

            report_ball_dp = ball_rero(
                ball_release, prior=finite_prior, eta_grid=(0.5,), mode="dp"
            )
            report_ball_direct = ball_rero(
                ball_release,
                prior=finite_prior,
                eta_grid=(0.5,),
                mode="gaussian_direct",
            )
            report_ball_rdp = ball_rero(
                ball_release, prior=finite_prior, eta_grid=(0.5,), mode="rdp"
            )

            report_standard_dp = ball_rero(
                standard_release, prior=finite_prior, eta_grid=(0.5,), mode="dp"
            )
            report_standard_direct = ball_rero(
                standard_release,
                prior=finite_prior,
                eta_grid=(0.5,),
                mode="gaussian_direct",
            )
            report_standard_rdp = ball_rero(
                standard_release, prior=finite_prior, eta_grid=(0.5,), mode="rdp"
            )

            sigma_ball = float(ball_release.privacy.ball.sigma)
            sigma_standard = float(standard_release.privacy.ball.sigma)
            acc_ball = float(ball_release.utility_metrics.get("accuracy", float("nan")))
            acc_standard = float(
                standard_release.utility_metrics.get("accuracy", float("nan"))
            )
            direct_ball = get_rero_scalar(report_ball_direct)
            direct_standard = get_rero_scalar(report_standard_direct)
            rdp_ball = get_rero_scalar(report_ball_rdp)
            rdp_standard = get_rero_scalar(report_standard_rdp)
            dp_ball = get_rero_scalar(report_ball_dp)
            dp_standard = get_rero_scalar(report_standard_dp)
            same_noise_standard = float(report_ball_direct.points[0].gamma_standard)

            row_key_payload = {
                "dataset": spec.tag,
                "model": model_family,
                "radius_tag": str(radius_tag),
                "epsilon": float(epsilon),
                "m": int(m),
                "release_seed": int(release_seed),
                "lam": float(args.lam),
                "delta": float(args.delta),
                "embedding_bound": float(args.embedding_bound),
                "solver": str(args.solver),
                "max_iter": int(args.max_iter),
            }
            row_key = hashlib.sha1(
                json.dumps(row_key_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]

            rows.append(
                {
                    "row_key": row_key,
                    "run_id": run_id,
                    "dataset_tag": spec.tag,
                    "dataset_name": spec.display_name,
                    "model_family": model_family,
                    "radius_tag": str(radius_tag),
                    "radius_value": float(radius_value),
                    "epsilon": float(epsilon),
                    "m": int(m),
                    "delta": float(args.delta),
                    "lam": float(args.lam),
                    "embedding_bound": float(args.embedding_bound),
                    "solver": str(args.solver),
                    "max_iter": int(args.max_iter),
                    "release_seed": int(release_seed),
                    "accuracy_ball": acc_ball,
                    "accuracy_standard": acc_standard,
                    "sigma_ball": sigma_ball,
                    "sigma_standard": sigma_standard,
                    "actual_epsilon_ball": actual_ball_epsilon(ball_release),
                    "actual_epsilon_standard": actual_ball_epsilon(standard_release),
                    "same_noise_standard_epsilon_from_ball": actual_standard_epsilon_same_noise(
                        ball_release
                    ),
                    "bound_generic_dp_ball": dp_ball,
                    "bound_generic_dp_standard": dp_standard,
                    "bound_direct_ball": direct_ball,
                    "bound_direct_standard": direct_standard,
                    "bound_rdp_ball": rdp_ball,
                    "bound_rdp_standard": rdp_standard,
                    "bound_same_noise_standard_from_ball": same_noise_standard,
                    "sensitivity_ball": float(ball_release.sensitivity.delta_ball),
                    "sensitivity_standard": float(
                        standard_release.sensitivity.delta_ball
                    ),
                    "sigma_ratio_standard_over_ball": (
                        sigma_standard / sigma_ball if sigma_ball > 0 else float("nan")
                    ),
                    "accuracy_gain_ball_minus_standard": acc_ball - acc_standard,
                    "bound_gap_direct_minus_rdp_ball": direct_ball - rdp_ball,
                    "bound_reduction_same_noise_pct": (
                        100.0
                        * (same_noise_standard - direct_ball)
                        / same_noise_standard
                        if same_noise_standard > 0
                        else float("nan")
                    ),
                }
            )

    rows_df = pd.DataFrame(rows)
    save_dataframe(
        rows_df, run_dir / "erm_seed_rows.csv", save_parquet_if_possible=False
    )

    rebuild_erm_dataset_outputs(
        dataset_dir,
        dataset_name=spec.display_name,
        dataset_tag=spec.tag,
        model_family=model_family,
        fixed_radius_tag=str(args.fixed_radius),
        fixed_epsilon=float(args.fixed_epsilon),
        fixed_m=int(args.fixed_m),
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": spec.display_name,
                "model_family": model_family,
                "run_id": run_id,
                "results_dir": str(dataset_dir),
                "run_dir": str(run_dir),
                "num_seed_rows": int(len(rows_df)),
                "used_configs": len(used_configs),
                "representative_config": {
                    "radius_tag": str(args.fixed_radius),
                    "epsilon": float(args.fixed_epsilon),
                    "m": int(args.fixed_m),
                },
                "radius_values": {k: float(v) for k, v in radius_values.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
