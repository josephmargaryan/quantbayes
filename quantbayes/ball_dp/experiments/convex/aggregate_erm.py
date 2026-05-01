#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.experiments.convex.run_attack_experiment import (
    DEFAULT_FIXED_EPSILON,
    DEFAULT_FIXED_M,
    canonicalize_model,
)

BALL_COLOR = "#0072B2"
STANDARD_COLOR = "#D55E00"
BASELINE_COLOR = "#4D4D4D"
RDP_COLOR = "#009E73"

DEFAULT_RADIUS_TAG = "q80"
DEFAULT_EPSILON = float(DEFAULT_FIXED_EPSILON)
DEFAULT_M = int(DEFAULT_FIXED_M)

PAPER_DATASET_ORDER = [
    "AG News-embeddings",
    "BANKING77-embeddings",
    "CIFAR-10-embeddings",
    "DBpedia-14-embeddings",
    "Emotion-embeddings",
    "IMDb-embeddings",
    "MNIST-embeddings",
    "TREC-6-embeddings",
    "Yelp Review Full-embeddings",
]
DATASET_ORDER_INDEX = {name: i for i, name in enumerate(PAPER_DATASET_ORDER)}

Config = tuple[str, float, int]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 4.8),
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


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def latex_escape_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("_", "\\_") for c in out.columns]
    return out


def save_table(df: pd.DataFrame, out_stem: str | Path) -> None:
    out_stem = Path(out_stem)
    save_dataframe(df, out_stem.with_suffix(".csv"))
    tex_df = latex_escape_columns(df)
    tex = tex_df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}")
    out_stem.with_suffix(".tex").write_text(tex)


def dataset_sort_key(name: str) -> tuple[int, str]:
    return (
        DATASET_ORDER_INDEX.get(str(name), len(PAPER_DATASET_ORDER)),
        str(name).lower(),
    )


def close_enough(a: pd.Series, target: float, tol: float = 1e-9) -> pd.Series:
    return pd.Series(
        np.isclose(a.astype(float).to_numpy(), float(target), atol=tol, rtol=0.0),
        index=a.index,
    )


def discover_models(
    results_root: Path, requested_models: Optional[Iterable[str]]
) -> list[str]:
    erm_root = results_root / "erm"
    if requested_models:
        return [canonicalize_model(str(m)) for m in requested_models]
    return [p.name for p in sorted(erm_root.iterdir()) if p.is_dir()]


def config_from_run_config(path: Path) -> Config | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None

    if (
        "fixed_radius" not in payload
        or "fixed_epsilon" not in payload
        or "fixed_m" not in payload
    ):
        return None

    return (
        str(payload["fixed_radius"]),
        float(payload["fixed_epsilon"]),
        int(payload["fixed_m"]),
    )


def summary_configs(summary: pd.DataFrame) -> set[Config]:
    required = {"radius_tag", "epsilon", "m"}
    if not required.issubset(summary.columns):
        return set()

    out: set[Config] = set()
    for row in (
        summary[["radius_tag", "epsilon", "m"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        out.add((str(row.radius_tag), float(row.epsilon), int(row.m)))
    return out


def config_mask(summary: pd.DataFrame, cfg: Config) -> pd.Series:
    radius_tag, epsilon, m = cfg
    return (
        (summary["radius_tag"].astype(str) == str(radius_tag))
        & close_enough(summary["epsilon"], float(epsilon))
        & (summary["m"].astype(int) == int(m))
    )


def apply_overrides(
    cfg: Config,
    *,
    requested_radius: str | None,
    requested_epsilon: float | None,
    requested_m: int | None,
) -> Config:
    radius, epsilon, m = cfg
    if requested_radius is not None:
        radius = str(requested_radius)
    if requested_epsilon is not None:
        epsilon = float(requested_epsilon)
    if requested_m is not None:
        m = int(requested_m)
    return str(radius), float(epsilon), int(m)


def resolve_representative_config(
    model_dir: Path,
    *,
    requested_radius: str | None,
    requested_epsilon: float | None,
    requested_m: int | None,
) -> tuple[Config, dict[str, Any]]:
    dataset_dirs = sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    )

    summaries: dict[str, pd.DataFrame] = {}
    all_summary_configs: Counter[Config] = Counter()
    run_config_counts: Counter[Config] = Counter()

    for dataset_dir in dataset_dirs:
        summary_path = dataset_dir / "erm_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summaries[dataset_dir.name] = summary
            all_summary_configs.update(summary_configs(summary))

        for run_config_path in sorted((dataset_dir / "runs").glob("*/run_config.json")):
            cfg = config_from_run_config(run_config_path)
            if cfg is not None:
                run_config_counts[cfg] += 1

    default_cfg: Config = (DEFAULT_RADIUS_TAG, DEFAULT_EPSILON, DEFAULT_M)

    candidate_set: set[Config] = set()
    candidate_set.update(run_config_counts.keys())
    candidate_set.update(all_summary_configs.keys())
    candidate_set.add(default_cfg)

    if not candidate_set:
        raise RuntimeError(f"No candidate configs found under {model_dir}")

    scored: list[tuple[tuple[Any, ...], Config, dict[str, Any]]] = []

    for raw_cfg in candidate_set:
        cfg = apply_overrides(
            raw_cfg,
            requested_radius=requested_radius,
            requested_epsilon=requested_epsilon,
            requested_m=requested_m,
        )

        coverage = 0
        covered_datasets: list[str] = []
        for dataset_name, summary in summaries.items():
            if config_mask(summary, cfg).any():
                coverage += 1
                covered_datasets.append(dataset_name)

        source_count = run_config_counts.get(raw_cfg, 0)
        exact_default = int(cfg == default_cfg)
        q80_bonus = int(cfg[0] == DEFAULT_RADIUS_TAG)
        epsilon_distance = abs(float(cfg[1]) - DEFAULT_EPSILON)
        m_distance = abs(int(cfg[2]) - DEFAULT_M)

        score = (
            coverage,
            source_count,
            exact_default,
            q80_bonus,
            -epsilon_distance,
            -m_distance,
        )

        scored.append(
            (
                score,
                cfg,
                {
                    "coverage": coverage,
                    "covered_datasets": covered_datasets,
                    "source_count": int(source_count),
                    "raw_candidate": {
                        "radius_tag": raw_cfg[0],
                        "epsilon": raw_cfg[1],
                        "m": raw_cfg[2],
                    },
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cfg, best_meta = scored[0]

    if best_meta["coverage"] == 0:
        available_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for dataset_name, summary in summaries.items():
            available_by_dataset[dataset_name] = [
                {"radius_tag": r, "epsilon": e, "m": m}
                for r, e, m in sorted(summary_configs(summary))
            ]

        raise RuntimeError(
            "Could not infer a representative config that appears in any summary. "
            f"Best attempted config was radius={best_cfg[0]!r}, "
            f"epsilon={best_cfg[1]}, m={best_cfg[2]}. "
            f"Available configs by dataset: {json.dumps(available_by_dataset, indent=2)}"
        )

    diagnostics = {
        "selected_config": {
            "radius_tag": best_cfg[0],
            "epsilon": float(best_cfg[1]),
            "m": int(best_cfg[2]),
        },
        "selected_score": list(best_score),
        "selected_meta": best_meta,
        "num_dataset_summaries": len(summaries),
        "num_candidate_configs": len(candidate_set),
        "requested_overrides": {
            "radius": requested_radius,
            "epsilon": requested_epsilon,
            "m": requested_m,
        },
    }

    return best_cfg, diagnostics


def load_model_rows(
    model_dir: Path,
    *,
    radius_tag: str,
    epsilon: float,
    m: int,
    strict: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        summary_path = dataset_dir / "erm_summary.csv"

        if not summary_path.exists():
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": "missing erm_summary.csv",
            }
            issues.append(issue)
            if strict:
                raise FileNotFoundError(f"{summary_path} does not exist.")
            continue

        summary = pd.read_csv(summary_path)
        cfg = (str(radius_tag), float(epsilon), int(m))
        sub = summary[config_mask(summary, cfg)].copy()

        if sub.empty:
            available_configs = [
                {"radius_tag": r, "epsilon": e, "m": mm}
                for r, e, mm in sorted(summary_configs(summary))
            ]
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": "missing representative config",
                "required": {
                    "radius_tag": str(radius_tag),
                    "epsilon": float(epsilon),
                    "m": int(m),
                },
                "available_configs": available_configs,
            }
            issues.append(issue)
            if strict:
                raise RuntimeError(
                    f"{summary_path} has no row for radius={radius_tag!r}, "
                    f"epsilon={float(epsilon)}, m={int(m)}."
                )
            continue

        if len(sub) > 1:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": "multiple representative rows; using first",
                    "num_rows": int(len(sub)),
                    "required": {
                        "radius_tag": str(radius_tag),
                        "epsilon": float(epsilon),
                        "m": int(m),
                    },
                }
            )

        row = sub.iloc[0]

        rows.append(
            {
                "model_family": str(row["model_family"]),
                "dataset_tag": str(row["dataset_tag"]),
                "dataset_name": str(row["dataset_name"]),
                "radius_tag": str(radius_tag),
                "epsilon": float(epsilon),
                "m": int(m),
                "accuracy_ball": float(row["accuracy_ball_mean"]),
                "accuracy_ball_ci_low": float(row["accuracy_ball_ci_low"]),
                "accuracy_ball_ci_high": float(row["accuracy_ball_ci_high"]),
                "accuracy_standard": float(row["accuracy_standard_mean"]),
                "accuracy_standard_ci_low": float(row["accuracy_standard_ci_low"]),
                "accuracy_standard_ci_high": float(row["accuracy_standard_ci_high"]),
                "accuracy_gain": float(row["accuracy_gain_ball_minus_standard_mean"]),
                "sigma_ball": float(row["sigma_ball_mean"]),
                "sigma_ball_ci_low": float(row["sigma_ball_ci_low"]),
                "sigma_ball_ci_high": float(row["sigma_ball_ci_high"]),
                "sigma_standard": float(row["sigma_standard_mean"]),
                "sigma_standard_ci_low": float(row["sigma_standard_ci_low"]),
                "sigma_standard_ci_high": float(row["sigma_standard_ci_high"]),
                "sigma_ratio_standard_over_ball": float(
                    row["sigma_ratio_standard_over_ball_mean"]
                ),
                "bound_ball_direct": float(row["bound_direct_ball_mean"]),
                "bound_ball_direct_ci_low": float(row["bound_direct_ball_ci_low"]),
                "bound_ball_direct_ci_high": float(row["bound_direct_ball_ci_high"]),
                "bound_ball_rdp": float(row["bound_rdp_ball_mean"]),
                "bound_ball_rdp_ci_low": float(row["bound_rdp_ball_ci_low"]),
                "bound_ball_rdp_ci_high": float(row["bound_rdp_ball_ci_high"]),
                "bound_standard_direct": float(row["bound_direct_standard_mean"]),
                "bound_standard_direct_ci_low": float(
                    row["bound_direct_standard_ci_low"]
                ),
                "bound_standard_direct_ci_high": float(
                    row["bound_direct_standard_ci_high"]
                ),
                "bound_standard_same_noise": float(
                    row["bound_same_noise_standard_from_ball_mean"]
                ),
                "bound_standard_same_noise_ci_low": float(
                    row["bound_same_noise_standard_from_ball_ci_low"]
                ),
                "bound_standard_same_noise_ci_high": float(
                    row["bound_same_noise_standard_from_ball_ci_high"]
                ),
                "bound_reduction_same_noise_pct": float(
                    row["bound_reduction_same_noise_pct_mean"]
                ),
                "actual_epsilon_ball": float(row["actual_epsilon_ball_mean"]),
                "actual_epsilon_standard": float(row["actual_epsilon_standard_mean"]),
                "same_noise_standard_epsilon_from_ball": float(
                    row["same_noise_standard_epsilon_from_ball_mean"]
                ),
                "sensitivity_ball": float(row["sensitivity_ball_mean"]),
                "sensitivity_ball_ci_low": float(row["sensitivity_ball_ci_low"]),
                "sensitivity_ball_ci_high": float(row["sensitivity_ball_ci_high"]),
                "sensitivity_standard": float(row["sensitivity_standard_mean"]),
                "sensitivity_standard_ci_low": float(
                    row["sensitivity_standard_ci_low"]
                ),
                "sensitivity_standard_ci_high": float(
                    row["sensitivity_standard_ci_high"]
                ),
                "release_seeds": int(row["release_seeds"]),
            }
        )

    agg = pd.DataFrame(rows)

    if agg.empty:
        if strict:
            raise RuntimeError(
                f"No ERM summaries for model {model_dir.name!r} matched "
                f"radius={radius_tag}, epsilon={epsilon}, m={m}."
            )
        return agg, issues

    agg = agg.sort_values(
        by="dataset_name", key=lambda s: s.map(dataset_sort_key)
    ).reset_index(drop=True)

    return agg, issues


def grouped_bar_figure(
    df: pd.DataFrame,
    *,
    left_mean: str,
    right_mean: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    y_label: str,
    title: str,
    out_stem: str | Path,
    left_ci_low: Optional[str] = None,
    left_ci_high: Optional[str] = None,
    right_ci_low: Optional[str] = None,
    right_ci_high: Optional[str] = None,
    yscale: Optional[str] = None,
    left_hatch: Optional[str] = None,
    right_hatch: Optional[str] = None,
) -> None:
    if df.empty:
        return

    x = np.arange(len(df), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(df) + 3.5), 4.8))

    left_yerr = None
    right_yerr = None

    if left_ci_low is not None and left_ci_high is not None:
        left_yerr = np.vstack(
            [df[left_mean] - df[left_ci_low], df[left_ci_high] - df[left_mean]]
        )

    if right_ci_low is not None and right_ci_high is not None:
        right_yerr = np.vstack(
            [df[right_mean] - df[right_ci_low], df[right_ci_high] - df[right_mean]]
        )

    ax.bar(
        x - width / 2.0,
        df[left_mean],
        width=width,
        color=left_color,
        edgecolor="#222222",
        linewidth=0.5,
        hatch=left_hatch,
        label=left_label,
        yerr=left_yerr,
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    ax.bar(
        x + width / 2.0,
        df[right_mean],
        width=width,
        color=right_color,
        edgecolor="#222222",
        linewidth=0.5,
        hatch=right_hatch,
        label=right_label,
        yerr=right_yerr,
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if yscale:
        finite_positive = (
            np.isfinite(df[left_mean].to_numpy(dtype=float))
            & np.isfinite(df[right_mean].to_numpy(dtype=float))
            & (df[left_mean].to_numpy(dtype=float) > 0)
            & (df[right_mean].to_numpy(dtype=float) > 0)
        )
        if np.all(finite_positive):
            ax.set_yscale(yscale)

    ax.legend(ncol=2)
    savefig_stem(fig, out_stem)


def build_matched_privacy_table(df: pd.DataFrame) -> pd.DataFrame:
    direct_abs_diff = np.abs(
        df["bound_ball_direct"].to_numpy(dtype=float)
        - df["bound_standard_direct"].to_numpy(dtype=float)
    )

    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Acc (Ball)": df["accuracy_ball"],
            "Acc (Std matched)": df["accuracy_standard"],
            "Acc gain": df["accuracy_gain"],
            "Direct bound (matched; common)": df["bound_ball_direct"],
            "Direct bound abs diff": direct_abs_diff,
            "$\\sigma$ (Ball)": df["sigma_ball"],
            "$\\sigma$ (Std matched)": df["sigma_standard"],
            "$\\sigma$ ratio": df["sigma_ratio_standard_over_ball"],
        }
    )


def build_same_noise_geometry_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Bound (Ball direct @ $\\sigma_{Ball}$)": df["bound_ball_direct"],
            "Bound (Std direct @ $\\sigma_{Ball}$)": df["bound_standard_same_noise"],
            "Bound red. %": df["bound_reduction_same_noise_pct"],
            "$\\varepsilon_{\\mathrm{std}|\\sigma_{Ball}}$": df[
                "same_noise_standard_epsilon_from_ball"
            ],
            "$\\Delta_{Ball}$": df["sensitivity_ball"],
            "$\\Delta_{Std}$": df["sensitivity_standard"],
            "$\\sigma_{Ball}$": df["sigma_ball"],
        }
    )


def build_extended_table(df: pd.DataFrame) -> pd.DataFrame:
    direct_abs_diff = np.abs(
        df["bound_ball_direct"].to_numpy(dtype=float)
        - df["bound_standard_direct"].to_numpy(dtype=float)
    )

    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Acc (Ball)": df["accuracy_ball"],
            "Acc (Std)": df["accuracy_standard"],
            "Direct bound (matched; common)": df["bound_ball_direct"],
            "Direct bound abs diff": direct_abs_diff,
            "Bound (Ball RDP)": df["bound_ball_rdp"],
            "Bound (Std same-noise)": df["bound_standard_same_noise"],
            "$\\sigma$ (Ball)": df["sigma_ball"],
            "$\\sigma$ (Std)": df["sigma_standard"],
            "$\\sigma$ ratio": df["sigma_ratio_standard_over_ball"],
            "$\\varepsilon_{\\mathrm{std}|\\sigma_{Ball}}$": df[
                "same_noise_standard_epsilon_from_ball"
            ],
        }
    )


def build_matched_direct_identity_table(df: pd.DataFrame) -> pd.DataFrame:
    ball = df["bound_ball_direct"].to_numpy(dtype=float)
    std = df["bound_standard_direct"].to_numpy(dtype=float)
    abs_diff = np.abs(ball - std)
    rel_diff = abs_diff / np.maximum(np.abs(ball), 1e-300)

    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Ball direct bound": ball,
            "Std matched direct bound": std,
            "Abs diff": abs_diff,
            "Rel diff": rel_diff,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset convex ERM results into model-specific "
            "publication tables and figures."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--model", nargs="*", type=str, default=None)

    parser.add_argument(
        "--radius",
        type=str,
        default=None,
        help=(
            "Representative radius tag. If omitted, inferred from per-dataset run_config.json files."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Representative epsilon. If omitted, inferred from per-dataset run_config.json files."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help=(
            "Representative prior size m. If omitted, inferred from per-dataset run_config.json files."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    results_root = Path(args.results_root)
    erm_root = results_root / "erm"

    if not erm_root.exists():
        raise FileNotFoundError(f"ERM results root not found: {erm_root}")

    summary_manifest: dict[str, Any] = {
        "results_root": str(results_root),
        "requested_radius": args.radius,
        "requested_epsilon": args.epsilon,
        "requested_m": args.m,
        "models": {},
    }

    models = discover_models(results_root, args.model)

    if not models:
        raise RuntimeError(f"No ERM model directories found under {erm_root}")

    any_success = False

    for model_family in models:
        model_dir = erm_root / model_family

        if not model_dir.exists() or not model_dir.is_dir():
            issue = {
                "status": "missing_model_dir",
                "model_dir": str(model_dir),
                "issues": [
                    {
                        "reason": "missing ERM model directory",
                        "model_dir": str(model_dir),
                    }
                ],
            }
            summary_manifest["models"][model_family] = issue

            if args.strict:
                raise FileNotFoundError(f"Missing ERM model directory: {model_dir}")

            continue

        selected_cfg, config_diagnostics = resolve_representative_config(
            model_dir,
            requested_radius=args.radius,
            requested_epsilon=args.epsilon,
            requested_m=args.m,
        )
        radius_tag, epsilon, m = selected_cfg

        agg_df, issues = load_model_rows(
            model_dir,
            radius_tag=radius_tag,
            epsilon=epsilon,
            m=m,
            strict=bool(args.strict),
        )

        if agg_df.empty:
            summary_manifest["models"][model_family] = {
                "included_datasets": [],
                "issues": issues,
                "status": "empty",
                "representative_config": {
                    "radius_tag": radius_tag,
                    "epsilon": float(epsilon),
                    "m": int(m),
                },
                "config_diagnostics": config_diagnostics,
            }
            continue

        any_success = True

        out_dir = ensure_dir(model_dir / "_aggregate")
        save_dataframe(agg_df, out_dir / "aggregate_summary.csv")

        save_table(
            build_matched_privacy_table(agg_df),
            out_dir / "aggregate_matched_privacy_table",
        )
        save_table(
            build_same_noise_geometry_table(agg_df),
            out_dir / "aggregate_same_noise_geometry_table",
        )
        save_table(
            build_extended_table(agg_df),
            out_dir / "aggregate_extended_table",
        )
        save_table(
            build_matched_direct_identity_table(agg_df),
            out_dir / "aggregate_matched_direct_identity_check",
        )

        sigma_vals = np.concatenate(
            [
                agg_df["sigma_ball"].to_numpy(dtype=float),
                agg_df["sigma_standard"].to_numpy(dtype=float),
            ]
        )
        sigma_positive = sigma_vals[np.isfinite(sigma_vals) & (sigma_vals > 0)]
        sigma_yscale = (
            "log"
            if sigma_positive.size
            and (np.max(sigma_positive) / np.min(sigma_positive) > 20.0)
            else None
        )

        grouped_bar_figure(
            agg_df,
            left_mean="sigma_ball",
            right_mean="sigma_standard",
            left_label="Ball-DP",
            right_label=r"Standard DP (matched $\varepsilon$)",
            left_color=BALL_COLOR,
            right_color=STANDARD_COLOR,
            y_label="Noise scale $\\sigma$",
            title="Gaussian noise scale at matched privacy",
            out_stem=out_dir / "fig_agg_sigma_grouped_bar",
            left_ci_low="sigma_ball_ci_low",
            left_ci_high="sigma_ball_ci_high",
            right_ci_low="sigma_standard_ci_low",
            right_ci_high="sigma_standard_ci_high",
            yscale=sigma_yscale,
        )
        grouped_bar_figure(
            agg_df,
            left_mean="bound_ball_direct",
            right_mean="bound_standard_same_noise",
            left_label="Ball direct",
            right_label=r"Standard direct @ same $\sigma$ as Ball",
            left_color=BALL_COLOR,
            right_color=BASELINE_COLOR,
            y_label=r"Exact-ID upper bound $\gamma$",
            title=(
                r"Direct exact-ID upper bound at Ball noise scale "
                r"(same $\sigma$; privacy not matched)"
            ),
            out_stem=out_dir / "fig_agg_bound_same_noise_grouped_bar",
            left_ci_low="bound_ball_direct_ci_low",
            left_ci_high="bound_ball_direct_ci_high",
            right_ci_low="bound_standard_same_noise_ci_low",
            right_ci_high="bound_standard_same_noise_ci_high",
            right_hatch="//",
        )
        grouped_bar_figure(
            agg_df,
            left_mean="accuracy_ball",
            right_mean="accuracy_standard",
            left_label="Ball-DP",
            right_label=r"Standard DP (matched $\varepsilon$)",
            left_color=BALL_COLOR,
            right_color=STANDARD_COLOR,
            y_label="Accuracy",
            title="Predictive accuracy at matched privacy",
            out_stem=out_dir / "fig_agg_accuracy_grouped_bar",
            left_ci_low="accuracy_ball_ci_low",
            left_ci_high="accuracy_ball_ci_high",
            right_ci_low="accuracy_standard_ci_low",
            right_ci_high="accuracy_standard_ci_high",
        )
        grouped_bar_figure(
            agg_df,
            left_mean="bound_ball_direct",
            right_mean="bound_ball_rdp",
            left_label="Ball direct",
            right_label="Ball RDP",
            left_color=BALL_COLOR,
            right_color=RDP_COLOR,
            y_label="Exact-ID upper bound",
            title="Ball direct vs Ball RDP exact-ID upper bound",
            out_stem=out_dir / "fig_agg_direct_vs_rdp_ball_grouped_bar",
            left_ci_low="bound_ball_direct_ci_low",
            left_ci_high="bound_ball_direct_ci_high",
            right_ci_low="bound_ball_rdp_ci_low",
            right_ci_high="bound_ball_rdp_ci_high",
            right_hatch="..",
        )

        summary_manifest["models"][model_family] = {
            "status": "ok",
            "representative_config": {
                "radius_tag": radius_tag,
                "epsilon": float(epsilon),
                "m": int(m),
            },
            "config_diagnostics": config_diagnostics,
            "included_datasets": agg_df["dataset_name"].tolist(),
            "num_datasets": int(len(agg_df)),
            "issues": issues,
            "output_dir": str(out_dir),
            "files": [
                str(out_dir / "aggregate_summary.csv"),
                str(out_dir / "aggregate_matched_privacy_table.csv"),
                str(out_dir / "aggregate_matched_privacy_table.tex"),
                str(out_dir / "aggregate_same_noise_geometry_table.csv"),
                str(out_dir / "aggregate_same_noise_geometry_table.tex"),
                str(out_dir / "aggregate_extended_table.csv"),
                str(out_dir / "aggregate_extended_table.tex"),
                str(out_dir / "aggregate_matched_direct_identity_check.csv"),
                str(out_dir / "aggregate_matched_direct_identity_check.tex"),
                str((out_dir / "fig_agg_sigma_grouped_bar").with_suffix(".pdf")),
                str(
                    (out_dir / "fig_agg_bound_same_noise_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
                str((out_dir / "fig_agg_accuracy_grouped_bar").with_suffix(".pdf")),
                str(
                    (out_dir / "fig_agg_direct_vs_rdp_ball_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
            ],
        }

    manifest_path = erm_root / "aggregate_erm_manifest.json"
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, sort_keys=True))

    if not any_success:
        raise RuntimeError(
            "No aggregate outputs were produced. "
            f"See diagnostic manifest: {manifest_path}"
        )

    print(json.dumps(summary_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
