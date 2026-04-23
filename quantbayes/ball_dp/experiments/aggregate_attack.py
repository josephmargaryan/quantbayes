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
DEFAULT_EPSILON = 1.0
DEFAULT_M = 16

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
                "num_targets": int(ball_row["num_targets"]),
                "candidate_draws": int(ball_row["candidate_draws"]),
                "release_seeds": int(ball_row["release_seeds"]),
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
        label="Best oblivious guess",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset_name"], rotation=28, ha="right")
    ax.set_ylabel("Empirical exact-ID")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Empirical exact identification vs oblivious baseline")
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
    ax.set_title("Exact-ID advantage over the best oblivious guess")
    ax.legend(ncol=2)
    savefig_stem(fig, out_stem)


def build_publication_table(df: pd.DataFrame) -> pd.DataFrame:
    pub = pd.DataFrame(
        {
            "Dataset": df["dataset_name"],
            "Empirical exact-ID (Ball)": df["exact_id_ball"],
            "Empirical exact-ID (Std)": df["exact_id_standard"],
            "Oblivious baseline": df["oblivious_baseline"],
            "Advantage (Ball)": df["advantage_ball"],
            "Advantage (Std)": df["advantage_standard"],
            "Trials (Ball)": df["n_trials_ball"],
            "Trials (Std)": df["n_trials_standard"],
        }
    )
    return pub


def main() -> None:
    configure_matplotlib()

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset convex attack results into model-specific publication tables and figures."
        )
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--model", nargs="*", type=str, default=None)
    parser.add_argument("--radius", type=str, default=DEFAULT_RADIUS_TAG)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
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
                str((out_dir / "fig_agg_attack_vs_baseline").with_suffix(".pdf")),
                str((out_dir / "fig_agg_attack_advantage_bar").with_suffix(".pdf")),
            ],
        }

    manifest_path = attack_root / "aggregate_attack_manifest.json"
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, sort_keys=True))
    print(json.dumps(summary_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
