#!/usr/bin/env python3
"""Aggregate Paper 2 nonconvex transcript attack outputs.

Compatible with outputs from:

    run_nonconvex_transcript_attack_experiment.py

Expected layout:

    <results-root>/paper2/nonconvex_transcript_attack/<model_tag>/<dataset_tag>/
        attack_summary.csv
        attack_trial_rows.csv
        release_rows.csv
        runs/<run_id>/...

This script writes per-model aggregate CSVs, LaTeX tables, publication-style
figures, and a cross-model aggregate directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.experiments.run_attack_experiment import (
    DEFAULT_FIXED_EPSILON,
    DEFAULT_FIXED_M,
)
from quantbayes.ball_dp.serialization import save_dataframe

SUMMARY_FILENAME = "attack_summary.csv"
TRIAL_FILENAME = "attack_trial_rows.csv"
RELEASE_FILENAME = "release_rows.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper2")
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help=(
            "Optional model tag to aggregate. If omitted, every model directory "
            "under <results-root>/paper2/nonconvex_transcript_attack is aggregated."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset tags or display names to include. With --strict, "
            "missing requested datasets raise an error."
        ),
    )
    parser.add_argument(
        "--radius",
        type=str,
        default=None,
        help=(
            "Representative radius tag for publication tables/figures. "
            "If omitted, inferred from available summaries."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Representative epsilon for publication tables/figures. "
            "If omitted, inferred from available summaries."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help=(
            "Representative m for publication tables/figures. "
            "If omitted, inferred from available summaries."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir-name", type=str, default="_aggregate")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 4.9),
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


def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def dataset_sort_key(name: str) -> tuple[int, str]:
    return (
        DATASET_ORDER_INDEX.get(str(name), len(PAPER_DATASET_ORDER)),
        str(name).lower(),
    )


def wanted_dataset_filter(datasets: Iterable[str] | None) -> set[str] | None:
    if datasets is None:
        return None
    return {normalize_name(ds) for ds in datasets}


def model_dirs(root: Path, model_tag: str | None) -> list[Path]:
    if model_tag is not None:
        return [root / str(model_tag)]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_aggregate")


def read_csv_if_exists(path: Path, *, strict: bool) -> pd.DataFrame | None:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Expected file not found: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
        print(f"Skipping unreadable {path}: {exc}", flush=True)
        return None


def dataset_matches(
    dataset_dir: Path,
    summary_df: pd.DataFrame | None,
    wanted: set[str] | None,
) -> bool:
    if wanted is None:
        return True

    keys = {normalize_name(dataset_dir.name)}

    if summary_df is not None and not summary_df.empty:
        if "dataset_tag" in summary_df.columns:
            keys.update(
                normalize_name(v)
                for v in summary_df["dataset_tag"].dropna().astype(str).unique()
            )
        if "dataset_name" in summary_df.columns:
            keys.update(
                normalize_name(v)
                for v in summary_df["dataset_name"].dropna().astype(str).unique()
            )

    return bool(keys & wanted)


def add_source_columns(
    df: pd.DataFrame,
    *,
    model_tag: str,
    dataset_dir: Path,
    source_path: Path,
) -> pd.DataFrame:
    out = df.copy()

    if "model_tag" not in out.columns:
        out["model_tag"] = str(model_tag)
    if "dataset_tag" not in out.columns:
        out["dataset_tag"] = str(dataset_dir.name)

    out["source_path"] = str(source_path)
    out["source_dataset_dir"] = str(dataset_dir)
    out["source_model_tag"] = str(model_tag)
    return out


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def deduplicate(df: pd.DataFrame, preferred_keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    for key in preferred_keys:
        if key in df.columns:
            return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)

    return df.drop_duplicates().reset_index(drop=True)


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "model_tag",
            "task",
            "mechanism",
            "attack_mode",
            "radius_tag",
            "epsilon",
            "m",
            "num_steps",
            "batch_size",
            "clip_norm",
        ]
        if c in df.columns
    ]

    if not sort_cols:
        return df.reset_index(drop=True)

    return df.sort_values(sort_cols).reset_index(drop=True)


def sort_trial_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "model_tag",
            "task",
            "mechanism",
            "attack_mode",
            "radius_tag",
            "epsilon",
            "m",
            "release_seed",
            "anchor_index",
            "target_position",
        ]
        if c in df.columns
    ]

    if not sort_cols:
        return df.reset_index(drop=True)

    return df.sort_values(sort_cols).reset_index(drop=True)


def sort_release_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "model_tag",
            "task",
            "mechanism",
            "radius_tag",
            "epsilon",
            "m",
            "release_seed",
            "anchor_index",
            "target_source_id",
        ]
        if c in df.columns
    ]

    if not sort_cols:
        return df.reset_index(drop=True)

    return df.sort_values(sort_cols).reset_index(drop=True)


def latex_escape_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("_", "\\_") for c in out.columns]
    return out


def save_table(df: pd.DataFrame, out_stem: str | Path) -> None:
    out_stem = Path(out_stem)
    save_dataframe(df, out_stem.with_suffix(".csv"), save_parquet_if_possible=False)

    tex_df = latex_escape_columns(df)
    tex = tex_df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}")
    out_stem.with_suffix(".tex").write_text(tex)


def savefig_stem(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def close_enough(a: pd.Series, target: float, tol: float = 1e-9) -> pd.Series:
    return pd.Series(
        np.isclose(a.astype(float).to_numpy(), float(target), atol=tol, rtol=0.0),
        index=a.index,
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
    summary: pd.DataFrame,
    *,
    requested_radius: str | None,
    requested_epsilon: float | None,
    requested_m: int | None,
) -> tuple[Config, dict[str, Any]]:
    default_cfg: Config = (DEFAULT_RADIUS_TAG, DEFAULT_EPSILON, DEFAULT_M)

    raw_candidates = summary_configs(summary)
    if not raw_candidates:
        raise RuntimeError("No candidate (radius_tag, epsilon, m) configs in summary.")

    candidates: Counter[Config] = Counter()
    for cfg in raw_candidates:
        candidates[
            apply_overrides(
                cfg,
                requested_radius=requested_radius,
                requested_epsilon=requested_epsilon,
                requested_m=requested_m,
            )
        ] += 1

    candidates[
        apply_overrides(
            default_cfg,
            requested_radius=requested_radius,
            requested_epsilon=requested_epsilon,
            requested_m=requested_m,
        )
    ] += 1

    scored: list[tuple[tuple[Any, ...], Config, dict[str, Any]]] = []
    for cfg in candidates:
        sub = summary[config_mask(summary, cfg)].copy()
        if "dataset_tag" in sub.columns:
            coverage = int(sub["dataset_tag"].astype(str).nunique())
        else:
            coverage = int(len(sub))

        exact_default = int(cfg == default_cfg)
        q80_bonus = int(cfg[0] == DEFAULT_RADIUS_TAG)
        epsilon_distance = abs(float(cfg[1]) - DEFAULT_EPSILON)
        m_distance = abs(int(cfg[2]) - DEFAULT_M)

        score = (
            coverage,
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
                    "candidate_count": int(candidates[cfg]),
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cfg, best_meta = scored[0]

    if best_meta["coverage"] == 0:
        available = [
            {"radius_tag": r, "epsilon": e, "m": m}
            for r, e, m in sorted(raw_candidates)
        ]
        raise RuntimeError(
            "Could not select a representative config with nonzero coverage. "
            f"Best attempted config was radius={best_cfg[0]!r}, "
            f"epsilon={best_cfg[1]}, m={best_cfg[2]}. "
            f"Available configs: {json.dumps(available, indent=2)}"
        )

    diagnostics = {
        "selected_config": {
            "radius_tag": best_cfg[0],
            "epsilon": float(best_cfg[1]),
            "m": int(best_cfg[2]),
        },
        "selected_score": list(best_score),
        "selected_meta": best_meta,
        "requested_overrides": {
            "radius": requested_radius,
            "epsilon": requested_epsilon,
            "m": requested_m,
        },
    }
    return best_cfg, diagnostics


def representative_summary(
    summary: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sub = summary[config_mask(summary, cfg)].copy()

    group_cols = [
        c
        for c in ["dataset_tag", "dataset_name", "mechanism", "attack_mode"]
        if c in sub.columns
    ]
    duplicate_count = 0

    if group_cols:
        duplicate_count = int(sub.duplicated(group_cols, keep=False).sum())
        if duplicate_count:
            sort_cols = [
                c
                for c in ["source_path", "num_steps", "batch_size"]
                if c in sub.columns
            ]
            if sort_cols:
                sub = sub.sort_values(sort_cols)
            sub = sub.drop_duplicates(subset=group_cols, keep="last")

    if "dataset_name" in sub.columns:
        sub = sub.sort_values(
            by="dataset_name",
            key=lambda s: s.map(dataset_sort_key),
        ).reset_index(drop=True)

    diagnostics = {
        "representative_rows": int(len(sub)),
        "duplicate_rows_dropped": int(duplicate_count),
    }
    return sub, diagnostics


def mode_short(mode: str) -> str:
    mode = str(mode)
    if mode == "known_inclusion":
        return "ki"
    if mode == "unknown_inclusion":
        return "ui"
    return normalize_name(mode)


def attack_wide(rep: pd.DataFrame) -> pd.DataFrame:
    if rep.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for dataset_tag, g in rep.groupby("dataset_tag", dropna=False):
        rec: dict[str, Any] = {
            "dataset_tag": str(dataset_tag),
            "dataset_name": (
                str(g["dataset_name"].iloc[0])
                if "dataset_name" in g.columns
                else str(dataset_tag)
            ),
        }

        if "m" in g.columns:
            rec["m"] = int(g["m"].iloc[0])

        for _, row in g.iterrows():
            mechanism = str(row.get("mechanism", "unknown"))
            attack_mode = mode_short(str(row.get("attack_mode", "unknown")))
            prefix = f"{mechanism}_{attack_mode}"

            exact_mean = row.get("exact_identification_success_mean", np.nan)
            exact_low = row.get(
                "exact_id_bernoulli_ci_low",
                row.get("exact_identification_success_ci_low", np.nan),
            )
            exact_high = row.get(
                "exact_id_bernoulli_ci_high",
                row.get("exact_identification_success_ci_high", np.nan),
            )

            rec[f"exact_id_{prefix}_mean"] = exact_mean
            rec[f"exact_id_{prefix}_ci_low"] = exact_low
            rec[f"exact_id_{prefix}_ci_high"] = exact_high

            for metric in [
                "prior_rank",
                "prior_hit@5",
                "target_inclusion_count",
                "selected_step_count",
                "accuracy",
                "noise_multiplier",
            ]:
                for suffix in ["mean", "std", "ci_low", "ci_high"]:
                    src = f"{metric}_{suffix}"
                    if src in row.index:
                        safe_metric = metric.replace("@", "_at_")
                        rec[f"{safe_metric}_{prefix}_{suffix}"] = row[src]

            if mechanism == "ball":
                rec[f"hayes_bound_{prefix}_mean"] = row.get(
                    "bound_hayes_ball_mean", np.nan
                )
                rec[f"hayes_bound_{prefix}_ci_low"] = row.get(
                    "bound_hayes_ball_ci_low", np.nan
                )
                rec[f"hayes_bound_{prefix}_ci_high"] = row.get(
                    "bound_hayes_ball_ci_high", np.nan
                )
            elif mechanism == "standard":
                rec[f"hayes_bound_{prefix}_mean"] = row.get(
                    "bound_hayes_standard_mean", np.nan
                )
                rec[f"hayes_bound_{prefix}_ci_low"] = row.get(
                    "bound_hayes_standard_ci_low", np.nan
                )
                rec[f"hayes_bound_{prefix}_ci_high"] = row.get(
                    "bound_hayes_standard_ci_high", np.nan
                )

        records.append(rec)

    wide = pd.DataFrame(records)
    if wide.empty:
        return wide

    wide = wide.sort_values(
        by="dataset_name",
        key=lambda s: s.map(dataset_sort_key),
    ).reset_index(drop=True)

    if "m" in wide.columns:
        wide["chance_baseline"] = 1.0 / wide["m"].astype(float)

    return wide


def choose_log_yscale(
    df: pd.DataFrame, cols: list[str], threshold: float = 20.0
) -> str | None:
    vals: list[float] = []
    for col in cols:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(float(v) for v in arr if np.isfinite(v) and v > 0)

    if not vals:
        return None
    if max(vals) / max(min(vals), 1e-300) > threshold:
        return "log"
    return None


def symmetric_yerr(
    df: pd.DataFrame, mean_col: str, low_col: str, high_col: str
) -> np.ndarray | None:
    if (
        mean_col not in df.columns
        or low_col not in df.columns
        or high_col not in df.columns
    ):
        return None

    mean = pd.to_numeric(df[mean_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    high = pd.to_numeric(df[high_col], errors="coerce")

    lower = np.maximum(mean.to_numpy(dtype=float) - low.to_numpy(dtype=float), 0.0)
    upper = np.maximum(high.to_numpy(dtype=float) - mean.to_numpy(dtype=float), 0.0)

    if not np.isfinite(lower).any() and not np.isfinite(upper).any():
        return None
    return np.vstack([lower, upper])


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
    left_ci_low: str | None = None,
    left_ci_high: str | None = None,
    right_ci_low: str | None = None,
    right_ci_high: str | None = None,
    yscale: str | None = None,
    left_hatch: str | None = None,
    right_hatch: str | None = None,
    ylim: tuple[float, float] | None = None,
    baseline_col: str | None = None,
    baseline_label: str | None = None,
) -> None:
    if df.empty or left_mean not in df.columns or right_mean not in df.columns:
        return

    x = np.arange(len(df), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(8.4, 0.9 * len(df) + 3.5), 4.9))

    left_yerr = (
        symmetric_yerr(df, left_mean, left_ci_low, left_ci_high)
        if left_ci_low and left_ci_high
        else None
    )
    right_yerr = (
        symmetric_yerr(df, right_mean, right_ci_low, right_ci_high)
        if right_ci_low and right_ci_high
        else None
    )

    ax.bar(
        x - width / 2.0,
        pd.to_numeric(df[left_mean], errors="coerce"),
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
        pd.to_numeric(df[right_mean], errors="coerce"),
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

    if baseline_col and baseline_col in df.columns:
        ax.plot(
            x,
            pd.to_numeric(df[baseline_col], errors="coerce"),
            color=BASELINE_COLOR,
            linestyle="--",
            marker=".",
            linewidth=1.5,
            label=baseline_label or "Chance baseline",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if yscale:
        ax.set_yscale(yscale)
    if ylim:
        ax.set_ylim(*ylim)

    ax.legend(ncol=2)
    savefig_stem(fig, out_stem)


def scatter_success_vs_bound(
    wide: pd.DataFrame,
    *,
    out_stem: str | Path,
    title: str,
) -> None:
    if wide.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    specs = [
        (
            "exact_id_ball_ki_mean",
            "hayes_bound_ball_ki_mean",
            "Ball KI",
            BALL_COLOR,
            "o",
        ),
        (
            "exact_id_standard_ki_mean",
            "hayes_bound_standard_ki_mean",
            "Std KI",
            STANDARD_COLOR,
            "s",
        ),
        (
            "exact_id_ball_ui_mean",
            "hayes_bound_ball_ui_mean",
            "Ball UI",
            RDP_COLOR,
            "^",
        ),
        (
            "exact_id_standard_ui_mean",
            "hayes_bound_standard_ui_mean",
            "Std UI",
            BASELINE_COLOR,
            "D",
        ),
    ]

    any_points = False
    for success_col, bound_col, label, color, marker in specs:
        if success_col not in wide.columns or bound_col not in wide.columns:
            continue

        x = pd.to_numeric(wide[bound_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(wide[success_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)

        if not mask.any():
            continue

        any_points = True
        ax.scatter(x[mask], y[mask], color=color, marker=marker, label=label, s=48)

        for i in np.flatnonzero(mask):
            ax.annotate(
                str(wide["dataset_name"].iloc[i]).split("-")[0],
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    if not any_points:
        plt.close(fig)
        return

    ax.set_xlabel(r"Hayes transcript bound $\gamma$")
    ax.set_ylabel("Empirical exact-ID success")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=9)
    savefig_stem(fig, out_stem)


def build_publication_tables(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if wide.empty:
        return {}

    exact_id = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Chance": wide.get("chance_baseline"),
            "Ball KI": wide.get("exact_id_ball_ki_mean"),
            "Std KI": wide.get("exact_id_standard_ki_mean"),
            "Ball UI": wide.get("exact_id_ball_ui_mean"),
            "Std UI": wide.get("exact_id_standard_ui_mean"),
            "Ball KI - Chance": wide.get("exact_id_ball_ki_mean")
            - wide.get("chance_baseline"),
            "Std KI - Chance": wide.get("exact_id_standard_ki_mean")
            - wide.get("chance_baseline"),
        }
    )

    rank = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Ball KI rank": wide.get("prior_rank_ball_ki_mean"),
            "Std KI rank": wide.get("prior_rank_standard_ki_mean"),
            "Ball UI rank": wide.get("prior_rank_ball_ui_mean"),
            "Std UI rank": wide.get("prior_rank_standard_ui_mean"),
            "Ball KI hit@5": wide.get("prior_hit_at_5_ball_ki_mean"),
            "Std KI hit@5": wide.get("prior_hit_at_5_standard_ki_mean"),
            "Ball UI hit@5": wide.get("prior_hit_at_5_ball_ui_mean"),
            "Std UI hit@5": wide.get("prior_hit_at_5_standard_ui_mean"),
        }
    )

    bounds = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Ball KI success": wide.get("exact_id_ball_ki_mean"),
            "Ball KI bound": wide.get("hayes_bound_ball_ki_mean"),
            "Std KI success": wide.get("exact_id_standard_ki_mean"),
            "Std KI bound": wide.get("hayes_bound_standard_ki_mean"),
            "Ball UI success": wide.get("exact_id_ball_ui_mean"),
            "Ball UI bound": wide.get("hayes_bound_ball_ui_mean"),
            "Std UI success": wide.get("exact_id_standard_ui_mean"),
            "Std UI bound": wide.get("hayes_bound_standard_ui_mean"),
        }
    )

    training = pd.DataFrame(
        {
            "Dataset": wide["dataset_name"],
            "Accuracy Ball KI": wide.get("accuracy_ball_ki_mean"),
            "Accuracy Std KI": wide.get("accuracy_standard_ki_mean"),
            "Noise Ball KI": wide.get("noise_multiplier_ball_ki_mean"),
            "Noise Std KI": wide.get("noise_multiplier_standard_ki_mean"),
        }
    )

    return {
        "aggregate_publication_attack_exact_id_table": exact_id,
        "aggregate_publication_attack_rank_table": rank,
        "aggregate_publication_attack_bound_table": bounds,
        "aggregate_publication_attack_training_table": training,
    }


def write_publication_outputs(
    *,
    out_dir: Path,
    rep: pd.DataFrame,
    cfg: Config,
    no_plots: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = attack_wide(rep)

    publication_manifest: dict[str, Any] = {
        "representative_config": {
            "radius_tag": cfg[0],
            "epsilon": float(cfg[1]),
            "m": int(cfg[2]),
        },
        "num_representative_rows": int(len(rep)),
        "num_representative_datasets": (
            int(rep["dataset_tag"].astype(str).nunique())
            if "dataset_tag" in rep.columns and not rep.empty
            else 0
        ),
        "files": [],
    }

    if wide.empty:
        publication_manifest["status"] = "empty"
        return publication_manifest

    save_dataframe(
        rep,
        out_dir / "aggregate_representative_attack_summary.csv",
        save_parquet_if_possible=False,
    )
    save_dataframe(
        wide,
        out_dir / "aggregate_representative_attack_wide.csv",
        save_parquet_if_possible=False,
    )
    publication_manifest["files"].extend(
        [
            str(out_dir / "aggregate_representative_attack_summary.csv"),
            str(out_dir / "aggregate_representative_attack_wide.csv"),
        ]
    )

    tables = build_publication_tables(wide)
    for stem, table in tables.items():
        save_table(table, out_dir / stem)
        publication_manifest["files"].extend(
            [
                str((out_dir / stem).with_suffix(".csv")),
                str((out_dir / stem).with_suffix(".tex")),
            ]
        )

    if not no_plots:
        grouped_bar_figure(
            wide,
            left_mean="exact_id_ball_ki_mean",
            right_mean="exact_id_standard_ki_mean",
            left_label="Ball KI",
            right_label="Standard KI",
            left_color=BALL_COLOR,
            right_color=STANDARD_COLOR,
            y_label="Exact-ID success",
            title=(
                "Known-inclusion transcript attack success\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_success_ki_grouped_bar",
            left_ci_low="exact_id_ball_ki_ci_low",
            left_ci_high="exact_id_ball_ki_ci_high",
            right_ci_low="exact_id_standard_ki_ci_low",
            right_ci_high="exact_id_standard_ki_ci_high",
            ylim=(0.0, 1.02),
            baseline_col="chance_baseline",
            baseline_label="Chance",
        )
        grouped_bar_figure(
            wide,
            left_mean="exact_id_ball_ui_mean",
            right_mean="exact_id_standard_ui_mean",
            left_label="Ball UI",
            right_label="Standard UI",
            left_color=BALL_COLOR,
            right_color=STANDARD_COLOR,
            y_label="Exact-ID success",
            title=(
                "Unknown-inclusion transcript attack success\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_success_ui_grouped_bar",
            left_ci_low="exact_id_ball_ui_ci_low",
            left_ci_high="exact_id_ball_ui_ci_high",
            right_ci_low="exact_id_standard_ui_ci_low",
            right_ci_high="exact_id_standard_ui_ci_high",
            ylim=(0.0, 1.02),
            baseline_col="chance_baseline",
            baseline_label="Chance",
        )
        grouped_bar_figure(
            wide,
            left_mean="exact_id_ball_ki_mean",
            right_mean="exact_id_ball_ui_mean",
            left_label="Ball KI",
            right_label="Ball UI",
            left_color=BALL_COLOR,
            right_color=RDP_COLOR,
            y_label="Exact-ID success",
            title=(
                "Ball-DP attack success: KI vs UI\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_success_ball_ki_vs_ui",
            left_ci_low="exact_id_ball_ki_ci_low",
            left_ci_high="exact_id_ball_ki_ci_high",
            right_ci_low="exact_id_ball_ui_ci_low",
            right_ci_high="exact_id_ball_ui_ci_high",
            ylim=(0.0, 1.02),
            baseline_col="chance_baseline",
            baseline_label="Chance",
            right_hatch="..",
        )
        grouped_bar_figure(
            wide,
            left_mean="exact_id_standard_ki_mean",
            right_mean="exact_id_standard_ui_mean",
            left_label="Standard KI",
            right_label="Standard UI",
            left_color=STANDARD_COLOR,
            right_color=BASELINE_COLOR,
            y_label="Exact-ID success",
            title=(
                "Standard-DP attack success: KI vs UI\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_success_standard_ki_vs_ui",
            left_ci_low="exact_id_standard_ki_ci_low",
            left_ci_high="exact_id_standard_ki_ci_high",
            right_ci_low="exact_id_standard_ui_ci_low",
            right_ci_high="exact_id_standard_ui_ci_high",
            ylim=(0.0, 1.02),
            baseline_col="chance_baseline",
            baseline_label="Chance",
            right_hatch="//",
        )
        grouped_bar_figure(
            wide,
            left_mean="prior_rank_ball_ki_mean",
            right_mean="prior_rank_standard_ki_mean",
            left_label="Ball KI",
            right_label="Standard KI",
            left_color=BALL_COLOR,
            right_color=STANDARD_COLOR,
            y_label="Mean prior rank",
            title=(
                "Known-inclusion attack rank\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_prior_rank_ki_grouped_bar",
            left_ci_low="prior_rank_ball_ki_ci_low",
            left_ci_high="prior_rank_ball_ki_ci_high",
            right_ci_low="prior_rank_standard_ki_ci_low",
            right_ci_high="prior_rank_standard_ki_ci_high",
        )
        grouped_bar_figure(
            wide,
            left_mean="hayes_bound_ball_ki_mean",
            right_mean="hayes_bound_standard_ki_mean",
            left_label="Ball KI bound",
            right_label="Standard KI bound",
            left_color=BALL_COLOR,
            right_color=BASELINE_COLOR,
            y_label=r"Hayes bound $\gamma$",
            title=(
                "Known-inclusion Hayes transcript bound\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
            out_stem=out_dir / "fig_agg_attack_hayes_bound_ki_grouped_bar",
            left_ci_low="hayes_bound_ball_ki_ci_low",
            left_ci_high="hayes_bound_ball_ki_ci_high",
            right_ci_low="hayes_bound_standard_ki_ci_low",
            right_ci_high="hayes_bound_standard_ki_ci_high",
            right_hatch="//",
        )
        scatter_success_vs_bound(
            wide,
            out_stem=out_dir / "fig_agg_attack_success_vs_hayes_bound",
            title=(
                "Empirical attack success vs Hayes transcript bound\n"
                f"radius={cfg[0]}, $\\varepsilon$={cfg[1]:g}, m={cfg[2]}"
            ),
        )

        publication_manifest["files"].extend(
            [
                str(
                    (out_dir / "fig_agg_attack_success_ki_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_success_ui_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_success_ball_ki_vs_ui").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_success_standard_ki_vs_ui").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_prior_rank_ki_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_hayes_bound_ki_grouped_bar").with_suffix(
                        ".pdf"
                    )
                ),
                str(
                    (out_dir / "fig_agg_attack_success_vs_hayes_bound").with_suffix(
                        ".pdf"
                    )
                ),
            ]
        )

    publication_manifest["status"] = "ok"
    return publication_manifest


def aggregate_one_model(
    model_dir: Path,
    *,
    wanted_datasets: set[str] | None,
    requested_radius: str | None,
    requested_epsilon: float | None,
    requested_m: int | None,
    strict: bool,
    no_plots: bool,
    out_dir_name: str,
) -> dict[str, Any]:
    if not model_dir.exists() or not model_dir.is_dir():
        if strict:
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return {
            "model_tag": model_dir.name,
            "status": "missing_model_dir",
            "model_dir": str(model_dir),
            "included_datasets": [],
            "issues": [{"reason": "missing model directory", "path": str(model_dir)}],
        }

    summary_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    release_frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []
    included_datasets: list[str] = []

    dataset_dirs = sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != str(out_dir_name)
    )

    for dataset_dir in dataset_dirs:
        summary_path = dataset_dir / SUMMARY_FILENAME
        summary_df = read_csv_if_exists(summary_path, strict=False)

        if not dataset_matches(dataset_dir, summary_df, wanted_datasets):
            continue

        if summary_df is None:
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": f"missing {SUMMARY_FILENAME}",
                "path": str(summary_path),
            }
            issues.append(issue)
            if strict:
                raise FileNotFoundError(f"Missing {SUMMARY_FILENAME}: {summary_path}")
            continue

        included_datasets.append(dataset_dir.name)
        summary_frames.append(
            add_source_columns(
                summary_df,
                model_tag=model_dir.name,
                dataset_dir=dataset_dir,
                source_path=summary_path,
            )
        )

        trial_path = dataset_dir / TRIAL_FILENAME
        trial_df = read_csv_if_exists(trial_path, strict=strict)
        if trial_df is not None:
            trial_frames.append(
                add_source_columns(
                    trial_df,
                    model_tag=model_dir.name,
                    dataset_dir=dataset_dir,
                    source_path=trial_path,
                )
            )
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": f"missing {TRIAL_FILENAME}",
                    "path": str(trial_path),
                }
            )

        release_path = dataset_dir / RELEASE_FILENAME
        release_df = read_csv_if_exists(release_path, strict=strict)
        if release_df is not None:
            release_frames.append(
                add_source_columns(
                    release_df,
                    model_tag=model_dir.name,
                    dataset_dir=dataset_dir,
                    source_path=release_path,
                )
            )
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": f"missing {RELEASE_FILENAME}",
                    "path": str(release_path),
                }
            )

    if wanted_datasets is not None:
        found = {normalize_name(ds) for ds in included_datasets}
        missing = sorted(wanted_datasets - found)
        if missing:
            issue = {
                "reason": "requested datasets missing",
                "missing_normalized_dataset_names": missing,
            }
            issues.append(issue)
            if strict:
                raise RuntimeError(
                    f"Missing requested datasets for model {model_dir.name}: {missing}"
                )

    summary = sort_summary(concat_or_empty(summary_frames))
    trials = sort_trial_rows(
        deduplicate(concat_or_empty(trial_frames), ["trial_key", "row_key"])
    )
    releases = sort_release_rows(
        deduplicate(concat_or_empty(release_frames), ["release_key"])
    )

    if summary.empty:
        return {
            "model_tag": model_dir.name,
            "status": "empty",
            "model_dir": str(model_dir),
            "included_datasets": included_datasets,
            "num_datasets": int(len(included_datasets)),
            "issues": issues,
        }

    out_dir = model_dir / str(out_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(
        summary,
        out_dir / "aggregate_attack_summary.csv",
        save_parquet_if_possible=False,
    )
    if not trials.empty:
        save_dataframe(
            trials,
            out_dir / "aggregate_attack_trial_rows.csv",
            save_parquet_if_possible=False,
        )
    if not releases.empty:
        save_dataframe(
            releases,
            out_dir / "aggregate_release_rows.csv",
            save_parquet_if_possible=False,
        )

    selected_cfg, cfg_diagnostics = resolve_representative_config(
        summary,
        requested_radius=requested_radius,
        requested_epsilon=requested_epsilon,
        requested_m=requested_m,
    )
    rep, rep_diagnostics = representative_summary(summary, selected_cfg)
    publication_manifest = write_publication_outputs(
        out_dir=out_dir,
        rep=rep,
        cfg=selected_cfg,
        no_plots=bool(no_plots),
    )

    manifest: dict[str, Any] = {
        "model_tag": model_dir.name,
        "status": "ok",
        "model_dir": str(model_dir),
        "output_dir": str(out_dir),
        "included_datasets": included_datasets,
        "num_datasets": int(len(included_datasets)),
        "num_summary_rows": int(len(summary)),
        "num_trial_rows": int(len(trials)),
        "num_release_rows": int(len(releases)),
        "issues": issues,
        "representative_config": {
            "radius_tag": selected_cfg[0],
            "epsilon": float(selected_cfg[1]),
            "m": int(selected_cfg[2]),
        },
        "config_diagnostics": cfg_diagnostics,
        "representative_diagnostics": rep_diagnostics,
        "publication_outputs": publication_manifest,
        "files": [
            str(out_dir / "aggregate_attack_summary.csv"),
        ],
    }

    if not trials.empty:
        manifest["files"].append(str(out_dir / "aggregate_attack_trial_rows.csv"))
    if not releases.empty:
        manifest["files"].append(str(out_dir / "aggregate_release_rows.csv"))

    manifest["files"].extend(publication_manifest.get("files", []))
    return manifest


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    results_root = Path(args.results_root)
    root = results_root / "paper2" / "nonconvex_transcript_attack"

    if not root.exists():
        raise FileNotFoundError(f"Nonconvex transcript attack root not found: {root}")

    wanted = wanted_dataset_filter(args.datasets)
    models = model_dirs(root, args.model_tag)

    if not models:
        raise RuntimeError(f"No model directories found under {root}")

    manifest: dict[str, Any] = {
        "results_root": str(results_root),
        "search_root": str(root),
        "requested_model_tag": args.model_tag,
        "requested_datasets": (
            list(args.datasets) if args.datasets is not None else None
        ),
        "requested_radius": args.radius,
        "requested_epsilon": args.epsilon,
        "requested_m": args.m,
        "strict": bool(args.strict),
        "no_plots": bool(args.no_plots),
        "models": {},
    }

    any_success = False
    combined_summaries: list[pd.DataFrame] = []
    combined_trials: list[pd.DataFrame] = []
    combined_releases: list[pd.DataFrame] = []

    for model_dir in models:
        model_manifest = aggregate_one_model(
            model_dir,
            wanted_datasets=wanted,
            requested_radius=args.radius,
            requested_epsilon=args.epsilon,
            requested_m=args.m,
            strict=bool(args.strict),
            no_plots=bool(args.no_plots),
            out_dir_name=str(args.out_dir_name),
        )
        manifest["models"][model_dir.name] = model_manifest

        if model_manifest.get("status") == "ok":
            any_success = True
            out_dir = Path(str(model_manifest["output_dir"]))

            summary_path = out_dir / "aggregate_attack_summary.csv"
            if summary_path.exists():
                combined_summaries.append(pd.read_csv(summary_path))

            trial_path = out_dir / "aggregate_attack_trial_rows.csv"
            if trial_path.exists():
                combined_trials.append(pd.read_csv(trial_path))

            release_path = out_dir / "aggregate_release_rows.csv"
            if release_path.exists():
                combined_releases.append(pd.read_csv(release_path))

    root_aggregate_dir = root / "_aggregate"
    root_aggregate_dir.mkdir(parents=True, exist_ok=True)

    if combined_summaries:
        all_summary = sort_summary(concat_or_empty(combined_summaries))
        save_dataframe(
            all_summary,
            root_aggregate_dir / "all_models_attack_summary.csv",
            save_parquet_if_possible=False,
        )
        manifest["all_models_attack_summary"] = str(
            root_aggregate_dir / "all_models_attack_summary.csv"
        )

    if combined_trials:
        all_trials = sort_trial_rows(
            deduplicate(concat_or_empty(combined_trials), ["trial_key", "row_key"])
        )
        save_dataframe(
            all_trials,
            root_aggregate_dir / "all_models_attack_trial_rows.csv",
            save_parquet_if_possible=False,
        )
        manifest["all_models_attack_trial_rows"] = str(
            root_aggregate_dir / "all_models_attack_trial_rows.csv"
        )

    if combined_releases:
        all_releases = sort_release_rows(
            deduplicate(concat_or_empty(combined_releases), ["release_key"])
        )
        save_dataframe(
            all_releases,
            root_aggregate_dir / "all_models_release_rows.csv",
            save_parquet_if_possible=False,
        )
        manifest["all_models_release_rows"] = str(
            root_aggregate_dir / "all_models_release_rows.csv"
        )

    manifest_path = root / "aggregate_nonconvex_transcript_attack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not any_success:
        raise RuntimeError(
            "No nonconvex transcript attack aggregate outputs were produced. "
            f"See manifest: {manifest_path}"
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
