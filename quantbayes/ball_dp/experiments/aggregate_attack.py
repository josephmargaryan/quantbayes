#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

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
    DEFAULT_RADIUS_TAG,
    DEFAULT_RIDGE_SENSITIVITY_MODE,
    DEFAULT_SUPPORT_SELECTION,
    canonicalize_model,
)

BALL_COLOR = "#0072B2"
STANDARD_COLOR = "#D55E00"
BASELINE_COLOR = "#4D4D4D"
RDP_COLOR = "#009E73"

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


def normalize_optional_filter(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "any", "all", "*"}:
        return None
    return text


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_int(value: Any) -> float | int:
    try:
        if value is None:
            return float("nan")
        v = float(value)
        if not math.isfinite(v):
            return float("nan")
        return int(v)
    except Exception:
        return float("nan")


def discover_models(
    results_root: Path, requested_models: Optional[Iterable[str]]
) -> list[str]:
    attack_root = results_root / "attack"
    if requested_models:
        return [canonicalize_model(str(m)) for m in requested_models]
    return [p.name for p in sorted(attack_root.iterdir()) if p.is_dir()]


def available_configs(summary: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        c
        for c in [
            "candidate_radius_tag",
            "epsilon",
            "m",
            "support_source_mode",
            "support_selection",
            "ridge_sensitivity_mode",
            "mechanism",
        ]
        if c in summary.columns
    ]
    if not cols:
        return []
    return summary[cols].drop_duplicates().sort_values(cols).to_dict(orient="records")


def apply_config_filter(
    summary: pd.DataFrame,
    *,
    radius_tag: str,
    epsilon: float,
    m: int,
    support_source_mode: str | None,
    support_selection: str | None,
    ridge_sensitivity_mode: str | None,
) -> pd.Series:
    mask = (
        (summary["candidate_radius_tag"].astype(str) == str(radius_tag))
        & close_enough(summary["epsilon"], float(epsilon))
        & (summary["m"].astype(int) == int(m))
    )

    optional_filters = {
        "support_source_mode": support_source_mode,
        "support_selection": support_selection,
        "ridge_sensitivity_mode": ridge_sensitivity_mode,
    }

    for col, expected in optional_filters.items():
        if expected is None:
            continue
        if col not in summary.columns:
            return pd.Series(False, index=summary.index)
        mask = mask & (summary[col].astype(str) == str(expected))

    return mask


def select_mechanism_row(
    sub: pd.DataFrame,
    *,
    mechanism: str,
    dataset_dir_name: str,
    strict: bool,
    issues: list[dict[str, Any]],
) -> pd.Series | None:
    mm = sub[sub["mechanism"].astype(str) == str(mechanism)].copy()

    if mm.empty:
        issues.append(
            {
                "dataset_dir": dataset_dir_name,
                "reason": "missing mechanism row",
                "mechanism": str(mechanism),
            }
        )
        return None

    if len(mm) > 1:
        issue = {
            "dataset_dir": dataset_dir_name,
            "reason": "multiple mechanism rows for representative config; using first",
            "mechanism": str(mechanism),
            "num_rows": int(len(mm)),
        }
        issues.append(issue)
        if strict:
            raise RuntimeError(json.dumps(issue, indent=2, sort_keys=True))

    sort_cols = [
        c
        for c in [
            "support_source_mode",
            "support_selection",
            "ridge_sensitivity_mode",
            "candidate_radius_value",
            "release_radius_value",
            "standard_radius_value",
        ]
        if c in mm.columns
    ]
    if sort_cols:
        mm = mm.sort_values(sort_cols)

    return mm.iloc[0]


def load_raw_dataset_frames(
    model_dir: Path,
    *,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    trial_frames: list[pd.DataFrame] = []
    release_frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []

    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        trial_path = dataset_dir / "attack_trial_rows.csv"
        release_path = dataset_dir / "release_rows.csv"

        if trial_path.exists():
            trial_frames.append(pd.read_csv(trial_path))
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": "missing attack_trial_rows.csv",
                }
            )
            if strict:
                raise FileNotFoundError(f"Missing {trial_path}")

        if release_path.exists():
            release_frames.append(pd.read_csv(release_path))
        else:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": "missing release_rows.csv",
                }
            )
            if strict:
                raise FileNotFoundError(f"Missing {release_path}")

    trials = (
        pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    )
    releases = (
        pd.concat(release_frames, ignore_index=True)
        if release_frames
        else pd.DataFrame()
    )

    if not trials.empty and "trial_key" in trials.columns:
        trials = trials.drop_duplicates(subset=["trial_key"], keep="last").reset_index(
            drop=True
        )
    if not releases.empty and "release_key" in releases.columns:
        releases = releases.drop_duplicates(
            subset=["release_key"], keep="last"
        ).reset_index(drop=True)

    return trials, releases, issues


def load_model_rows(
    model_dir: Path,
    *,
    radius_tag: str,
    epsilon: float,
    m: int,
    support_source_mode: str | None,
    support_selection: str | None,
    ridge_sensitivity_mode: str | None,
    strict: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        summary_path = dataset_dir / "attack_summary.csv"

        if not summary_path.exists():
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": "missing attack_summary.csv",
            }
            issues.append(issue)
            if strict:
                raise FileNotFoundError(f"Missing {summary_path}")
            continue

        summary = pd.read_csv(summary_path)
        required_cols = {"candidate_radius_tag", "epsilon", "m", "mechanism"}
        missing_cols = sorted(required_cols - set(summary.columns))
        if missing_cols:
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": "attack_summary.csv missing required columns",
                "missing_columns": missing_cols,
            }
            issues.append(issue)
            if strict:
                raise RuntimeError(json.dumps(issue, indent=2, sort_keys=True))
            continue

        mask = apply_config_filter(
            summary,
            radius_tag=radius_tag,
            epsilon=epsilon,
            m=m,
            support_source_mode=support_source_mode,
            support_selection=support_selection,
            ridge_sensitivity_mode=ridge_sensitivity_mode,
        )
        sub = summary[mask].copy()

        if sub.empty:
            issue = {
                "dataset_dir": dataset_dir.name,
                "reason": "missing representative config",
                "required": {
                    "radius_tag": radius_tag,
                    "epsilon": float(epsilon),
                    "m": int(m),
                    "support_source_mode": support_source_mode,
                    "support_selection": support_selection,
                    "ridge_sensitivity_mode": ridge_sensitivity_mode,
                },
                "available_configs": available_configs(summary),
            }
            issues.append(issue)
            if strict:
                raise RuntimeError(json.dumps(issue, indent=2, sort_keys=True))
            continue

        ball_row = select_mechanism_row(
            sub,
            mechanism="ball",
            dataset_dir_name=dataset_dir.name,
            strict=strict,
            issues=issues,
        )
        std_row = select_mechanism_row(
            sub,
            mechanism="standard",
            dataset_dir_name=dataset_dir.name,
            strict=strict,
            issues=issues,
        )

        if ball_row is None or std_row is None:
            continue

        rows.append(
            {
                "model_family": str(ball_row["model_family"]),
                "dataset_tag": str(ball_row["dataset_tag"]),
                "dataset_name": str(ball_row["dataset_name"]),
                "radius_tag": str(radius_tag),
                "epsilon": float(epsilon),
                "m": int(m),
                "support_source_mode": str(
                    ball_row.get("support_source_mode", "unknown")
                ),
                "support_selection": str(ball_row.get("support_selection", "unknown")),
                "ridge_sensitivity_mode": str(
                    ball_row.get("ridge_sensitivity_mode", "unknown")
                ),
                "exact_id_ball": safe_float(ball_row.get("exact_id_mean")),
                "exact_id_ball_ci_low": safe_float(ball_row.get("exact_id_ci_low")),
                "exact_id_ball_ci_high": safe_float(ball_row.get("exact_id_ci_high")),
                "exact_id_standard": safe_float(std_row.get("exact_id_mean")),
                "exact_id_standard_ci_low": safe_float(std_row.get("exact_id_ci_low")),
                "exact_id_standard_ci_high": safe_float(
                    std_row.get("exact_id_ci_high")
                ),
                "oblivious_baseline": safe_float(ball_row.get("oblivious_kappa")),
                "advantage_ball": safe_float(ball_row.get("attack_advantage")),
                "advantage_standard": safe_float(std_row.get("attack_advantage")),
                "n_trials_ball": safe_int(ball_row.get("n_trials")),
                "n_trials_standard": safe_int(std_row.get("n_trials")),
                "num_support_anchors_ball": safe_int(
                    ball_row.get("num_support_anchors")
                ),
                "num_support_anchors_standard": safe_int(
                    std_row.get("num_support_anchors")
                ),
                "support_draws_ball": safe_int(ball_row.get("support_draws")),
                "support_draws_standard": safe_int(std_row.get("support_draws")),
                "release_seeds_ball": safe_int(ball_row.get("release_seeds")),
                "release_seeds_standard": safe_int(std_row.get("release_seeds")),
                "predicted_source_mode_share_ball": safe_float(
                    ball_row.get("predicted_source_mode_share")
                ),
                "predicted_source_mode_share_standard": safe_float(
                    std_row.get("predicted_source_mode_share")
                ),
                "predicted_anchor_rate_ball": safe_float(
                    ball_row.get("predicted_anchor_rate")
                ),
                "predicted_anchor_rate_standard": safe_float(
                    std_row.get("predicted_anchor_rate")
                ),
                "support_contains_anchor_rate_ball": safe_float(
                    ball_row.get("support_contains_anchor_rate")
                ),
                "support_contains_anchor_rate_standard": safe_float(
                    std_row.get("support_contains_anchor_rate")
                ),
                "sigma_ball": safe_float(ball_row.get("release_sigma_mean")),
                "sigma_standard": safe_float(std_row.get("release_sigma_mean")),
                "accuracy_ball": safe_float(ball_row.get("release_accuracy_mean")),
                "accuracy_standard": safe_float(std_row.get("release_accuracy_mean")),
                "bound_direct_ball": safe_float(ball_row.get("bound_direct_mean")),
                "bound_direct_standard": safe_float(std_row.get("bound_direct_mean")),
                "bound_rdp_ball": safe_float(ball_row.get("bound_rdp_mean")),
                "bound_rdp_standard": safe_float(std_row.get("bound_rdp_mean")),
                "bound_direct_standard_same_noise": safe_float(
                    ball_row.get("bound_direct_standard_same_noise_mean")
                ),
                "bound_direct_instance_ball": safe_float(
                    ball_row.get("bound_direct_instance_finite_opt_mean")
                ),
                "bound_direct_instance_standard": safe_float(
                    std_row.get("bound_direct_instance_finite_opt_mean")
                ),
                "model_pairwise_snr_ball": safe_float(
                    ball_row.get("model_pairwise_snr_median_mean")
                ),
                "model_pairwise_snr_standard": safe_float(
                    std_row.get("model_pairwise_snr_median_mean")
                ),
                "model_nn_snr_ball": safe_float(
                    ball_row.get("model_nn_snr_median_mean")
                ),
                "model_nn_snr_standard": safe_float(
                    std_row.get("model_nn_snr_median_mean")
                ),
                "feature_pairwise_distance_ball": safe_float(
                    ball_row.get("feature_pairwise_distance_median_mean")
                ),
                "feature_pairwise_distance_standard": safe_float(
                    std_row.get("feature_pairwise_distance_median_mean")
                ),
                "ridge_count_dilution_ball": safe_float(
                    ball_row.get("ridge_count_dilution_mean")
                ),
                "ridge_count_dilution_standard": safe_float(
                    std_row.get("ridge_count_dilution_mean")
                ),
                "ridge_inverse_tau_ball": safe_float(
                    ball_row.get("ridge_inverse_noise_tau_mean")
                ),
                "ridge_inverse_tau_standard": safe_float(
                    std_row.get("ridge_inverse_noise_tau_mean")
                ),
                "posterior_effective_candidates_ball": safe_float(
                    ball_row.get("posterior_effective_candidates_mean")
                ),
                "posterior_effective_candidates_standard": safe_float(
                    std_row.get("posterior_effective_candidates_mean")
                ),
                "posterior_true_probability_ball": safe_float(
                    ball_row.get("posterior_true_probability_mean")
                ),
                "posterior_true_probability_standard": safe_float(
                    std_row.get("posterior_true_probability_mean")
                ),
                "posterior_top1_probability_ball": safe_float(
                    ball_row.get("posterior_top1_probability_mean")
                ),
                "posterior_top1_probability_standard": safe_float(
                    std_row.get("posterior_top1_probability_mean")
                ),
            }
        )

    agg = pd.DataFrame(rows)

    if agg.empty:
        if strict:
            raise RuntimeError(
                f"No attack summaries for model {model_dir.name!r} matched "
                f"radius={radius_tag}, epsilon={epsilon}, m={m}."
            )
        return agg, issues

    agg = agg.sort_values(
        by="dataset_name", key=lambda s: s.map(dataset_sort_key)
    ).reset_index(drop=True)

    return agg, issues


def symmetric_yerr(
    df: pd.DataFrame,
    mean_col: str,
    low_col: str,
    high_col: str,
) -> np.ndarray | None:
    if (
        mean_col not in df.columns
        or low_col not in df.columns
        or high_col not in df.columns
    ):
        return None

    mean = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=float)

    lower = np.maximum(mean - low, 0.0)
    upper = np.maximum(high - mean, 0.0)

    if not np.isfinite(lower).any() and not np.isfinite(upper).any():
        return None

    return np.vstack([lower, upper])


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
    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(df) + 3.5), 4.8))

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

    if baseline_col is not None and baseline_col in df.columns:
        ax.plot(
            x,
            pd.to_numeric(df[baseline_col], errors="coerce"),
            color=BASELINE_COLOR,
            linestyle="--",
            marker=".",
            linewidth=1.6,
            label=baseline_label or "Baseline",
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


def make_attack_vs_baseline_figure(df: pd.DataFrame, out_stem: str | Path) -> None:
    if df.empty:
        return

    x = np.arange(len(df), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(df) + 3.5), 4.8))

    ax.bar(
        x - width,
        df["exact_id_ball"],
        width=width,
        color=BALL_COLOR,
        edgecolor="#222222",
        linewidth=0.5,
        label="Ball-DP",
        yerr=symmetric_yerr(
            df,
            "exact_id_ball",
            "exact_id_ball_ci_low",
            "exact_id_ball_ci_high",
        ),
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    ax.bar(
        x,
        df["exact_id_standard"],
        width=width,
        color=STANDARD_COLOR,
        edgecolor="#222222",
        linewidth=0.5,
        label="Standard DP",
        yerr=symmetric_yerr(
            df,
            "exact_id_standard",
            "exact_id_standard_ci_low",
            "exact_id_standard_ci_high",
        ),
        capsize=3,
        error_kw={"elinewidth": 0.9},
    )
    ax.bar(
        x + width,
        df["oblivious_baseline"],
        width=width,
        color=BASELINE_COLOR,
        edgecolor="#222222",
        linewidth=0.5,
        label="Uniform-prior baseline 1/m",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel("Empirical exact-ID")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Convex finite-prior exact identification")
    ax.legend(ncol=3)
    savefig_stem(fig, out_stem)


def make_advantage_figure(df: pd.DataFrame, out_stem: str | Path) -> None:
    grouped_bar_figure(
        df,
        left_mean="advantage_ball",
        right_mean="advantage_standard",
        left_label="Ball-DP",
        right_label="Standard DP",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label="Attack advantage over 1/m",
        title="Exact-ID advantage over the uniform-prior baseline",
        out_stem=out_stem,
    )


def build_publication_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Empirical exact-ID (Ball)": df["exact_id_ball"],
            "Empirical exact-ID (Std)": df["exact_id_standard"],
            "Uniform prior baseline": df["oblivious_baseline"],
            "Instance bound (Ball)": df["bound_direct_instance_ball"],
            "Instance bound (Std)": df["bound_direct_instance_standard"],
            "Direct release bound (Ball)": df["bound_direct_ball"],
            "Direct release bound (Std)": df["bound_direct_standard"],
            "Advantage (Ball)": df["advantage_ball"],
            "Advantage (Std)": df["advantage_standard"],
            "Trials (Ball)": df["n_trials_ball"],
            "Trials (Std)": df["n_trials_standard"],
        }
    )


def build_utility_bound_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Accuracy (Ball)": df["accuracy_ball"],
            "Accuracy (Std)": df["accuracy_standard"],
            "$\\sigma$ (Ball)": df["sigma_ball"],
            "$\\sigma$ (Std)": df["sigma_standard"],
            "Direct bound (Ball)": df["bound_direct_ball"],
            "Direct bound (Std)": df["bound_direct_standard"],
            "Std same-noise bound": df["bound_direct_standard_same_noise"],
            "RDP bound (Ball)": df["bound_rdp_ball"],
            "RDP bound (Std)": df["bound_rdp_standard"],
        }
    )


def build_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Support source": df["support_source_mode"],
            "Support selection": df["support_selection"],
            "Ridge sens mode": df["ridge_sensitivity_mode"],
            "Instance bound (Ball)": df["bound_direct_instance_ball"],
            "Instance bound (Std)": df["bound_direct_instance_standard"],
            "Median model SNR (Ball)": df["model_pairwise_snr_ball"],
            "Median model SNR (Std)": df["model_pairwise_snr_standard"],
            "NN model SNR (Ball)": df["model_nn_snr_ball"],
            "NN model SNR (Std)": df["model_nn_snr_standard"],
            "Feature pairwise dist (Ball)": df["feature_pairwise_distance_ball"],
            "Ridge count dilution (Ball)": df["ridge_count_dilution_ball"],
            "Ridge tau (Ball)": df["ridge_inverse_tau_ball"],
            "Posterior eff candidates (Ball)": df[
                "posterior_effective_candidates_ball"
            ],
            "Posterior eff candidates (Std)": df[
                "posterior_effective_candidates_standard"
            ],
            "Posterior true prob (Ball)": df["posterior_true_probability_ball"],
            "Posterior true prob (Std)": df["posterior_true_probability_standard"],
            "Pred mode share (Ball)": df["predicted_source_mode_share_ball"],
            "Pred mode share (Std)": df["predicted_source_mode_share_standard"],
            "Pred anchor rate (Ball)": df["predicted_anchor_rate_ball"],
            "Pred anchor rate (Std)": df["predicted_anchor_rate_standard"],
            "Support contains anchor (Ball)": df["support_contains_anchor_rate_ball"],
            "Support contains anchor (Std)": df[
                "support_contains_anchor_rate_standard"
            ],
        }
    )


def write_publication_outputs(
    *,
    agg_df: pd.DataFrame,
    out_dir: Path,
    radius_tag: str,
    epsilon: float,
    m: int,
) -> list[str]:
    files: list[str] = []

    save_dataframe(agg_df, out_dir / "aggregate_summary.csv")
    files.append(str(out_dir / "aggregate_summary.csv"))

    save_table(build_publication_table(agg_df), out_dir / "aggregate_publication_table")
    save_table(
        build_utility_bound_table(agg_df), out_dir / "aggregate_utility_bound_table"
    )
    save_table(build_diagnostic_table(agg_df), out_dir / "aggregate_diagnostic_table")

    files.extend(
        [
            str(out_dir / "aggregate_publication_table.csv"),
            str(out_dir / "aggregate_publication_table.tex"),
            str(out_dir / "aggregate_utility_bound_table.csv"),
            str(out_dir / "aggregate_utility_bound_table.tex"),
            str(out_dir / "aggregate_diagnostic_table.csv"),
            str(out_dir / "aggregate_diagnostic_table.tex"),
        ]
    )

    title_suffix = f"radius={radius_tag}, $\\varepsilon$={epsilon:g}, m={m}"

    make_attack_vs_baseline_figure(agg_df, out_dir / "fig_agg_attack_vs_baseline")
    make_advantage_figure(agg_df, out_dir / "fig_agg_attack_advantage_bar")
    grouped_bar_figure(
        agg_df,
        left_mean="bound_direct_instance_ball",
        right_mean="bound_direct_instance_standard",
        left_label="Ball instance bound",
        right_label="Standard instance bound",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label=r"Instance direct bound $\gamma$",
        title=f"Convex attack instance direct bound\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_instance_bound_grouped_bar",
        right_hatch="//",
    )
    grouped_bar_figure(
        agg_df,
        left_mean="bound_direct_ball",
        right_mean="bound_direct_standard",
        left_label="Ball release bound",
        right_label="Standard release bound",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label=r"Release direct bound $\gamma$",
        title=f"Convex release direct bound\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_release_bound_grouped_bar",
        right_hatch="//",
    )
    grouped_bar_figure(
        agg_df,
        left_mean="accuracy_ball",
        right_mean="accuracy_standard",
        left_label="Ball-DP",
        right_label="Standard DP",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label="Accuracy",
        title=f"Convex attack release accuracy\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_release_accuracy_grouped_bar",
    )
    grouped_bar_figure(
        agg_df,
        left_mean="sigma_ball",
        right_mean="sigma_standard",
        left_label="Ball-DP",
        right_label="Standard DP",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label=r"Noise scale $\sigma$",
        title=f"Convex attack release noise scale\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_release_sigma_grouped_bar",
        yscale=choose_log_yscale(agg_df, ["sigma_ball", "sigma_standard"]),
    )
    grouped_bar_figure(
        agg_df,
        left_mean="model_pairwise_snr_ball",
        right_mean="model_pairwise_snr_standard",
        left_label="Ball-DP",
        right_label="Standard DP",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label="Median model-space SNR",
        title=f"Convex attack model-space SNR\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_model_snr_grouped_bar",
        yscale=choose_log_yscale(
            agg_df, ["model_pairwise_snr_ball", "model_pairwise_snr_standard"]
        ),
    )
    grouped_bar_figure(
        agg_df,
        left_mean="posterior_effective_candidates_ball",
        right_mean="posterior_effective_candidates_standard",
        left_label="Ball-DP",
        right_label="Standard DP",
        left_color=BALL_COLOR,
        right_color=STANDARD_COLOR,
        y_label="Posterior effective candidates",
        title=f"Posterior effective candidate count\n{title_suffix}",
        out_stem=out_dir / "fig_agg_attack_posterior_effective_candidates",
    )

    files.extend(
        [
            str((out_dir / "fig_agg_attack_vs_baseline").with_suffix(".pdf")),
            str((out_dir / "fig_agg_attack_advantage_bar").with_suffix(".pdf")),
            str(
                (out_dir / "fig_agg_attack_instance_bound_grouped_bar").with_suffix(
                    ".pdf"
                )
            ),
            str(
                (out_dir / "fig_agg_attack_release_bound_grouped_bar").with_suffix(
                    ".pdf"
                )
            ),
            str(
                (out_dir / "fig_agg_attack_release_accuracy_grouped_bar").with_suffix(
                    ".pdf"
                )
            ),
            str(
                (out_dir / "fig_agg_attack_release_sigma_grouped_bar").with_suffix(
                    ".pdf"
                )
            ),
            str((out_dir / "fig_agg_attack_model_snr_grouped_bar").with_suffix(".pdf")),
            str(
                (out_dir / "fig_agg_attack_posterior_effective_candidates").with_suffix(
                    ".pdf"
                )
            ),
        ]
    )

    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset fixed-support finite-prior convex attack results "
            "into model-specific publication tables and figures."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--model", nargs="*", type=str, default=None)
    parser.add_argument("--radius", type=str, default=DEFAULT_RADIUS_TAG)
    parser.add_argument("--epsilon", type=float, default=float(DEFAULT_FIXED_EPSILON))
    parser.add_argument("--m", type=int, default=int(DEFAULT_FIXED_M))
    parser.add_argument("--support-source", type=str, default="public_only")
    parser.add_argument(
        "--support-selection",
        type=str,
        default=str(DEFAULT_SUPPORT_SELECTION),
        help="Use 'any' to disable this filter.",
    )
    parser.add_argument(
        "--ridge-sensitivity-mode",
        type=str,
        default=str(DEFAULT_RIDGE_SENSITIVITY_MODE),
        help="Use 'any' to disable this filter.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    results_root = Path(args.results_root)
    attack_root = results_root / "attack"

    if not attack_root.exists():
        raise FileNotFoundError(f"Attack results root not found: {attack_root}")

    support_source_filter = normalize_optional_filter(args.support_source)
    support_selection_filter = normalize_optional_filter(args.support_selection)
    ridge_sensitivity_filter = normalize_optional_filter(args.ridge_sensitivity_mode)

    summary_manifest: dict[str, Any] = {
        "results_root": str(results_root),
        "radius": str(args.radius),
        "epsilon": float(args.epsilon),
        "m": int(args.m),
        "support_source": support_source_filter,
        "support_selection": support_selection_filter,
        "ridge_sensitivity_mode": ridge_sensitivity_filter,
        "models": {},
    }

    models = discover_models(results_root, args.model)

    if not models:
        raise RuntimeError(f"No attack model directories found under {attack_root}")

    any_success = False

    for model_family in models:
        model_dir = attack_root / model_family

        if not model_dir.exists() or not model_dir.is_dir():
            if args.strict:
                raise FileNotFoundError(f"Missing attack model directory: {model_dir}")
            summary_manifest["models"][model_family] = {
                "status": "missing_model_dir",
                "issues": [
                    {"reason": "missing model directory", "path": str(model_dir)}
                ],
            }
            continue

        agg_df, issues = load_model_rows(
            model_dir,
            radius_tag=str(args.radius),
            epsilon=float(args.epsilon),
            m=int(args.m),
            support_source_mode=support_source_filter,
            support_selection=support_selection_filter,
            ridge_sensitivity_mode=ridge_sensitivity_filter,
            strict=bool(args.strict),
        )

        raw_trials, raw_releases, raw_issues = load_raw_dataset_frames(
            model_dir,
            strict=False,
        )
        issues.extend(raw_issues)

        if agg_df.empty:
            summary_manifest["models"][model_family] = {
                "included_datasets": [],
                "issues": issues,
                "status": "empty",
            }
            continue

        any_success = True

        out_dir = ensure_dir(model_dir / "_aggregate")

        files = write_publication_outputs(
            agg_df=agg_df,
            out_dir=out_dir,
            radius_tag=str(args.radius),
            epsilon=float(args.epsilon),
            m=int(args.m),
        )

        if not raw_trials.empty:
            save_dataframe(raw_trials, out_dir / "aggregate_attack_trial_rows.csv")
            files.append(str(out_dir / "aggregate_attack_trial_rows.csv"))

        if not raw_releases.empty:
            save_dataframe(raw_releases, out_dir / "aggregate_release_rows.csv")
            files.append(str(out_dir / "aggregate_release_rows.csv"))

        summary_manifest["models"][model_family] = {
            "status": "ok",
            "included_datasets": agg_df["dataset_name"].tolist(),
            "num_datasets": int(len(agg_df)),
            "num_summary_rows": int(len(agg_df)),
            "num_trial_rows": int(len(raw_trials)),
            "num_release_rows": int(len(raw_releases)),
            "issues": issues,
            "output_dir": str(out_dir),
            "files": files,
        }

    manifest_path = attack_root / "aggregate_attack_manifest.json"
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, sort_keys=True))

    if not any_success:
        raise RuntimeError(
            "No convex attack aggregate outputs were produced. "
            f"See diagnostic manifest: {manifest_path}"
        )

    print(json.dumps(summary_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
