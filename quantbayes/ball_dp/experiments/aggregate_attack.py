#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BALL_COLOR = "#1f77b4"
STANDARD_COLOR = "#d62728"
BASELINE_COLOR = "#6b7280"

DEFAULT_RADIUS_TAG = "q80"
DEFAULT_EPSILON = 4.0
DEFAULT_M = 8
DEFAULT_SUPPORT_SELECTION = None
DEFAULT_RIDGE_SENSITIVITY_MODE = None

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


def dataset_sort_key(name: str) -> tuple[int, str]:
    return (
        DATASET_ORDER_INDEX.get(str(name), len(PAPER_DATASET_ORDER)),
        str(name).lower(),
    )


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


def close_enough(a: pd.Series, target: float, tol: float = 1e-9) -> pd.Series:
    return np.isclose(a.astype(float).to_numpy(), float(target), atol=tol, rtol=0.0)


def discover_models(
    results_root: Path, requested_models: Optional[Iterable[str]]
) -> list[str]:
    attack_root = results_root / "attack"
    if requested_models:
        models = [str(m) for m in requested_models]
    else:
        models = [p.name for p in sorted(attack_root.iterdir()) if p.is_dir()]
    return models


def load_model_rows(
    model_dir: Path,
    *,
    radius_tag: str,
    epsilon: float,
    m: int,
    support_source_mode: Optional[str],
    support_selection: Optional[str],
    ridge_sensitivity_mode: Optional[str],
    strict: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for dataset_dir in sorted(
        p for p in model_dir.iterdir() if p.is_dir() and p.name != "_aggregate"
    ):
        summary_path = dataset_dir / "attack_summary.csv"
        if not summary_path.exists():
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": "missing attack_summary.csv",
                }
            )
            continue

        summary = pd.read_csv(summary_path)
        mask = (
            (summary["candidate_radius_tag"] == str(radius_tag))
            & close_enough(summary["epsilon"], float(epsilon))
            & (summary["m"].astype(int) == int(m))
        )
        if support_source_mode is not None and "support_source_mode" in summary.columns:
            mask = mask & (summary["support_source_mode"] == str(support_source_mode))
        if support_selection is not None and "support_selection" in summary.columns:
            mask = mask & (summary["support_selection"] == str(support_selection))
        if (
            ridge_sensitivity_mode is not None
            and "ridge_sensitivity_mode" in summary.columns
        ):
            mask = mask & (
                summary["ridge_sensitivity_mode"] == str(ridge_sensitivity_mode)
            )
        sub = summary[mask].copy()
        if sub.empty:
            issues.append(
                {
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
                }
            )
            continue

        ball = sub[sub["mechanism"] == "ball"]
        standard = sub[sub["mechanism"] == "standard"]
        if ball.empty or standard.empty:
            issues.append(
                {
                    "dataset_dir": dataset_dir.name,
                    "reason": "missing mechanism rows",
                    "has_ball": not ball.empty,
                    "has_standard": not standard.empty,
                }
            )
            continue

        ball_row = ball.iloc[0]
        std_row = standard.iloc[0]
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
                "exact_id_ball": float(ball_row["exact_id_mean"]),
                "exact_id_ball_ci_low": float(ball_row["exact_id_ci_low"]),
                "exact_id_ball_ci_high": float(ball_row["exact_id_ci_high"]),
                "exact_id_standard": float(std_row["exact_id_mean"]),
                "exact_id_standard_ci_low": float(std_row["exact_id_ci_low"]),
                "exact_id_standard_ci_high": float(std_row["exact_id_ci_high"]),
                "oblivious_baseline": float(ball_row["oblivious_kappa"]),
                "advantage_ball": float(ball_row["attack_advantage"]),
                "advantage_standard": float(std_row["attack_advantage"]),
                "n_trials_ball": int(ball_row["n_trials"]),
                "n_trials_standard": int(std_row["n_trials"]),
                "num_support_anchors_ball": int(
                    ball_row.get("num_support_anchors", np.nan)
                ),
                "num_support_anchors_standard": int(
                    std_row.get("num_support_anchors", np.nan)
                ),
                "support_draws_ball": int(ball_row.get("support_draws", np.nan)),
                "support_draws_standard": int(std_row.get("support_draws", np.nan)),
                "release_seeds_ball": int(ball_row["release_seeds"]),
                "release_seeds_standard": int(std_row["release_seeds"]),
                "predicted_source_mode_share_ball": float(
                    ball_row.get("predicted_source_mode_share", np.nan)
                ),
                "predicted_source_mode_share_standard": float(
                    std_row.get("predicted_source_mode_share", np.nan)
                ),
                "predicted_anchor_rate_ball": float(
                    ball_row.get("predicted_anchor_rate", np.nan)
                ),
                "predicted_anchor_rate_standard": float(
                    std_row.get("predicted_anchor_rate", np.nan)
                ),
                "support_contains_anchor_rate_ball": float(
                    ball_row.get("support_contains_anchor_rate", np.nan)
                ),
                "support_contains_anchor_rate_standard": float(
                    std_row.get("support_contains_anchor_rate", np.nan)
                ),
                "sigma_ball": float(ball_row.get("release_sigma_mean", np.nan)),
                "sigma_standard": float(std_row.get("release_sigma_mean", np.nan)),
                "accuracy_ball": float(ball_row.get("release_accuracy_mean", np.nan)),
                "accuracy_standard": float(
                    std_row.get("release_accuracy_mean", np.nan)
                ),
                "bound_direct_ball": float(ball_row.get("bound_direct_mean", np.nan)),
                "bound_direct_standard": float(
                    std_row.get("bound_direct_mean", np.nan)
                ),
                "bound_direct_standard_same_noise": float(
                    ball_row.get("bound_direct_standard_same_noise_mean", np.nan)
                ),
                "bound_direct_instance_ball": float(
                    ball_row.get("bound_direct_instance_finite_opt_mean", np.nan)
                ),
                "bound_direct_instance_standard": float(
                    std_row.get("bound_direct_instance_finite_opt_mean", np.nan)
                ),
                "model_pairwise_snr_ball": float(
                    ball_row.get("model_pairwise_snr_median_mean", np.nan)
                ),
                "model_pairwise_snr_standard": float(
                    std_row.get("model_pairwise_snr_median_mean", np.nan)
                ),
                "ridge_count_dilution_ball": float(
                    ball_row.get("ridge_count_dilution_mean", np.nan)
                ),
                "ridge_count_dilution_standard": float(
                    std_row.get("ridge_count_dilution_mean", np.nan)
                ),
                "ridge_inverse_tau_ball": float(
                    ball_row.get("ridge_inverse_noise_tau_mean", np.nan)
                ),
                "ridge_inverse_tau_standard": float(
                    std_row.get("ridge_inverse_noise_tau_mean", np.nan)
                ),
                "posterior_effective_candidates_ball": float(
                    ball_row.get("posterior_effective_candidates_mean", np.nan)
                ),
                "posterior_effective_candidates_standard": float(
                    std_row.get("posterior_effective_candidates_mean", np.nan)
                ),
            }
        )

    agg = pd.DataFrame(rows)
    if agg.empty:
        if strict:
            raise RuntimeError(
                f"No attack summaries for model {model_dir.name!r} matched radius={radius_tag}, epsilon={epsilon}, m={m}."
            )
        return agg, issues

    agg = agg.sort_values(
        by="dataset_name", key=lambda s: s.map(dataset_sort_key)
    ).reset_index(drop=True)
    return agg, issues


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
        label="Ball-DP",
        yerr=np.vstack(
            [
                df["exact_id_ball"] - df["exact_id_ball_ci_low"],
                df["exact_id_ball_ci_high"] - df["exact_id_ball"],
            ]
        ),
        capsize=3,
    )
    ax.bar(
        x,
        df["exact_id_standard"],
        width=width,
        color=STANDARD_COLOR,
        label="Standard DP",
        yerr=np.vstack(
            [
                df["exact_id_standard"] - df["exact_id_standard_ci_low"],
                df["exact_id_standard_ci_high"] - df["exact_id_standard"],
            ]
        ),
        capsize=3,
    )
    ax.bar(
        x + width,
        df["oblivious_baseline"],
        width=width,
        color=BASELINE_COLOR,
        label="Uniform-prior baseline 1/m",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel("Empirical exact-ID")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Empirical exact identification vs uniform-prior baseline")
    ax.legend(ncol=3)
    savefig_stem(fig, out_stem)


def make_advantage_figure(df: pd.DataFrame, out_stem: str | Path) -> None:
    if df.empty:
        return
    x = np.arange(len(df), dtype=float)
    width = 0.33

    fig, ax = plt.subplots(figsize=(max(8.2, 0.9 * len(df) + 3.5), 4.8))
    ax.bar(
        x - width / 2.0,
        df["advantage_ball"],
        width=width,
        color=BALL_COLOR,
        label="Ball-DP",
    )
    ax.bar(
        x + width / 2.0,
        df["advantage_standard"],
        width=width,
        color=STANDARD_COLOR,
        label="Standard DP",
    )
    ax.axhline(0.0, color=BASELINE_COLOR, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel("Attack advantage over 1/m")
    ax.set_title("Exact-ID advantage over the uniform-prior baseline")
    ax.legend(ncol=2)
    savefig_stem(fig, out_stem)


def build_publication_table(df: pd.DataFrame) -> pd.DataFrame:
    pub = pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Empirical exact-ID (Ball)": df["exact_id_ball"],
            "Empirical exact-ID (Std)": df["exact_id_standard"],
            "Uniform prior baseline": df["oblivious_baseline"],
            "Instance bound (Ball)": df["bound_direct_instance_ball"],
            "Instance bound (Std)": df["bound_direct_instance_standard"],
            "Advantage (Ball)": df["advantage_ball"],
            "Advantage (Std)": df["advantage_standard"],
            "Trials (Ball)": df["n_trials_ball"],
            "Trials (Std)": df["n_trials_standard"],
        }
    )
    return pub


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
            "Ridge count dilution (Ball)": df["ridge_count_dilution_ball"],
            "Ridge tau (Ball)": df["ridge_inverse_tau_ball"],
            "Posterior eff candidates (Ball)": df[
                "posterior_effective_candidates_ball"
            ],
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


def main() -> None:
    configure_matplotlib()

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset fixed-support finite-prior attack results into model-specific publication tables and figures."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--model", nargs="*", type=str, default=None)
    parser.add_argument("--radius", type=str, default=DEFAULT_RADIUS_TAG)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--support-source", type=str, default="public_only")
    parser.add_argument(
        "--support-selection", type=str, default=DEFAULT_SUPPORT_SELECTION
    )
    parser.add_argument(
        "--ridge-sensitivity-mode", type=str, default=DEFAULT_RIDGE_SENSITIVITY_MODE
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    attack_root = results_root / "attack"
    if not attack_root.exists():
        raise FileNotFoundError(f"Attack results root not found: {attack_root}")

    summary_manifest: dict[str, Any] = {
        "results_root": str(results_root),
        "radius": str(args.radius),
        "epsilon": float(args.epsilon),
        "m": int(args.m),
        "support_source": str(args.support_source),
        "support_selection": (
            None if args.support_selection is None else str(args.support_selection)
        ),
        "ridge_sensitivity_mode": (
            None
            if args.ridge_sensitivity_mode is None
            else str(args.ridge_sensitivity_mode)
        ),
        "models": {},
    }

    models = discover_models(results_root, args.model)
    if not models:
        raise RuntimeError(f"No attack model directories found under {attack_root}")

    for model_family in models:
        model_dir = attack_root / model_family
        if not model_dir.exists() or not model_dir.is_dir():
            if args.strict:
                raise FileNotFoundError(f"Missing attack model directory: {model_dir}")
            continue

        agg_df, issues = load_model_rows(
            model_dir,
            radius_tag=str(args.radius),
            epsilon=float(args.epsilon),
            m=int(args.m),
            support_source_mode=str(args.support_source),
            support_selection=(
                None if args.support_selection is None else str(args.support_selection)
            ),
            ridge_sensitivity_mode=(
                None
                if args.ridge_sensitivity_mode is None
                else str(args.ridge_sensitivity_mode)
            ),
            strict=bool(args.strict),
        )
        if agg_df.empty:
            summary_manifest["models"][model_family] = {
                "included_datasets": [],
                "issues": issues,
                "status": "empty",
            }
            continue

        out_dir = ensure_dir(model_dir / "_aggregate")
        save_dataframe(agg_df, out_dir / "aggregate_summary.csv")
        save_table(
            build_publication_table(agg_df), out_dir / "aggregate_publication_table"
        )
        save_table(
            build_diagnostic_table(agg_df), out_dir / "aggregate_diagnostic_table"
        )

        make_attack_vs_baseline_figure(agg_df, out_dir / "fig_agg_attack_vs_baseline")
        make_advantage_figure(agg_df, out_dir / "fig_agg_attack_advantage_bar")

        summary_manifest["models"][model_family] = {
            "status": "ok",
            "included_datasets": agg_df["dataset_name"].tolist(),
            "num_datasets": int(len(agg_df)),
            "issues": issues,
            "output_dir": str(out_dir),
            "files": [
                str(out_dir / "aggregate_summary.csv"),
                str(out_dir / "aggregate_publication_table.csv"),
                str(out_dir / "aggregate_publication_table.tex"),
                str(out_dir / "aggregate_diagnostic_table.csv"),
                str(out_dir / "aggregate_diagnostic_table.tex"),
                str((out_dir / "fig_agg_attack_vs_baseline").with_suffix(".pdf")),
                str((out_dir / "fig_agg_attack_advantage_bar").with_suffix(".pdf")),
            ],
        }

    manifest_path = attack_root / "aggregate_attack_manifest.json"
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, sort_keys=True))
    print(json.dumps(summary_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
