#!/usr/bin/env python3
"""Thesis-scale aggregated convex finite-prior Ball-DP experiment.

This script is intentionally aligned with the public APIs used by
``quantbayes/ball_dp/examples/convex_demo.ipynb`` and the more comprehensive
``experiments/convex/run_attack_experiment.py``:

* the finite-prior support construction uses the shared canonical helpers;
* the "standard" comparator is implemented as the same convex Gaussian-output
  release with radius ``2B`` (or an explicitly supplied standard radius);
* empirical attacks are exact finite-prior Gaussian-output MAP attacks;
* outputs are written as CSV/JSON files and figures for thesis use.

Example
-------
python quantbayes/ball_dp/experiments/convex/run_thesis_experiment.py \
  --out-dir runs/convex_thesis_eps4 \
  --num-supports 8 \
  --support-draws 2 \
  --target-policy all \
  --epsilon 4.0

Quick smoke test:
python quantbayes/ball_dp/experiments/convex/run_thesis_experiment.py \
  --out-dir runs/convex_smoke \
  --num-supports 1 \
  --support-draws 1 \
  --targets-per-support 2 \
  --max-trials 2 \
  --skip-curves
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path.cwd()
if (REPO_ROOT / "quantbayes").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantbayes.ball_dp import ball_rero, fit_convex, make_finite_identification_prior
from quantbayes.ball_dp.api import (
    attack_convex_finite_prior_trial,
    diagnose_convex_ball_output_finite_prior,
)
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
)

MODEL_ORDER = ["Ball-DP", "Std-DP"]
MODEL_COLORS = {"Ball-DP": "tab:blue", "Std-DP": "tab:orange"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "figure.figsize": (7.6, 4.8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.5,
            "legend.frameon": False,
            "legend.fontsize": 9.8,
            "font.size": 10.5,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def savefig_stem(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def first_epsilon(ledger: Any) -> float:
    certs = getattr(ledger, "dp_certificates", None)
    if certs:
        return float(certs[0].epsilon)
    cert = getattr(ledger, "dp_certificate", None)
    if cert is not None:
        return float(cert.epsilon)
    return float("nan")


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = float(k) / float(n)
    denom = 1.0 + z * z / float(n)
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(center - half), float(center + half)


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(arr))
    if arr.size == 1:
        return mu, mu, mu
    half = float(z * np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mu, mu - half, mu + half


def make_synthetic_embeddings(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=int(args.n_samples),
        n_features=int(args.n_features),
        n_informative=int(args.n_features),
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=float(args.class_sep),
        flip_y=float(args.label_noise),
        random_state=int(args.seed),
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_public, y_train, y_public = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_public = scaler.transform(X_public).astype(np.float32)

    max_norm = max(
        float(np.linalg.norm(X_train, axis=1).max()),
        float(np.linalg.norm(X_public, axis=1).max()),
        1e-12,
    )
    scale = 0.95 * float(args.embedding_bound) / max_norm
    return (
        (scale * X_train).astype(np.float32),
        y_train.astype(np.int32),
        (scale * X_public).astype(np.float32),
        y_public.astype(np.int32),
    )


def same_label_public_neighbor_counts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    counts = np.zeros(len(X_train), dtype=np.int32)
    for label in sorted(np.unique(y_train).tolist()):
        train_idx = np.where(y_train == int(label))[0]
        X_pub_label = np.asarray(X_public[y_public == int(label)], dtype=np.float32)
        for idx in train_idx:
            d = np.linalg.norm(X_pub_label - X_train[idx], axis=1)
            counts[idx] = int(np.sum(d <= float(radius) + 1e-8))
    return counts


def fit_release(
    *,
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    args: argparse.Namespace,
    mechanism: str,
    seed: int,
) -> Any:
    standard_radius = (
        2.0 * float(args.embedding_bound)
        if args.standard_radius is None
        else float(args.standard_radius)
    )
    if mechanism == "ball":
        privacy = "ball_dp"
        radius = float(args.radius)
        epsilon = float(args.epsilon)
        delta = float(args.delta)
    elif mechanism == "standard":
        privacy = "ball_dp"
        radius = float(standard_radius)
        epsilon = float(args.epsilon)
        delta = float(args.delta)
    elif mechanism == "erm":
        privacy = "noiseless"
        radius = float(args.radius)
        epsilon = None
        delta = None
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    return fit_convex(
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        X_eval=np.asarray(X_eval, dtype=np.float32),
        y_eval=np.asarray(y_eval, dtype=np.int32),
        model_family=str(args.model_family),
        privacy=privacy,
        radius=float(radius),
        standard_radius=float(standard_radius),
        ridge_sensitivity_mode=str(args.ridge_sensitivity_mode),
        lam=float(args.lam),
        epsilon=epsilon,
        delta=delta,
        embedding_bound=float(args.embedding_bound),
        num_classes=int(len(np.unique(np.concatenate([y, y_eval])))),
        orders=tuple(float(v) for v in args.orders),
        max_iter=int(args.max_iter),
        solver=str(args.solver),
        seed=int(seed),
    )


def release_row(
    release: Any, *, model: str, mechanism: str, trial_id: int | None
) -> dict[str, Any]:
    return {
        "trial_id": -1 if trial_id is None else int(trial_id),
        "model": str(model),
        "mechanism": str(mechanism),
        "accuracy": float(release.utility_metrics.get("accuracy", np.nan)),
        "sigma": float(getattr(release.privacy.ball, "sigma", np.nan)),
        "release_radius": float(getattr(release.privacy.ball, "radius", np.nan)),
        "delta_ball": safe_float(getattr(release.sensitivity, "delta_ball", np.nan)),
        "delta_standard": safe_float(getattr(release.sensitivity, "delta_std", np.nan)),
        "epsilon_ball_view": first_epsilon(getattr(release.privacy, "ball", None)),
        "epsilon_standard_view": first_epsilon(
            getattr(release.privacy, "standard", None)
        ),
    }


def build_trials(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    public_source = CandidateSource("public", X_public, y_public)
    banks = find_feasible_replacement_banks(
        X_train=X_train,
        y_train=y_train,
        candidate_sources=[public_source],
        radius=float(args.radius),
        min_support_size=int(args.prior_size),
        num_banks=int(args.num_supports),
        seed=int(args.seed),
        anchor_selection="large_bank",
        strict=True,
    )

    support_rows: list[dict[str, Any]] = []
    trial_entries: list[dict[str, Any]] = []

    for bank_id, bank in enumerate(banks):
        for draw_index in range(int(args.support_draws)):
            support = select_support_from_bank(
                bank,
                m=int(args.prior_size),
                selection=str(args.support_selection),
                seed=int(args.seed),
                draw_index=int(draw_index),
            )
            positions = target_positions_for_support(
                support,
                policy=str(args.target_policy),
                num_targets=(
                    None
                    if args.target_policy == "all"
                    else int(args.targets_per_support)
                ),
                seed=int(args.seed + 17 * draw_index + 7919 * bank_id),
            )
            support_id = len({row["support_id"] for row in support_rows})
            for pos in range(int(support.m)):
                support_rows.append(
                    {
                        "support_id": int(support_id),
                        "bank_id": int(bank_id),
                        "draw_index": int(draw_index),
                        "support_position": int(pos),
                        "source_id": str(support.source_ids[pos]),
                        "label": int(support.y[pos]),
                        "distance_to_center": float(support.distances_to_center[pos]),
                        "prior_weight": float(support.weights[pos]),
                        "is_target_pool": int(pos in set(positions)),
                        "center_source_id": str(support.center_source_id),
                        "support_hash": str(support.support_hash),
                        "bank_size": int(bank.X.shape[0]),
                    }
                )

            for target_pos in positions:
                trial = make_replacement_trial(
                    X_train=X_train,
                    y_train=y_train,
                    support=support,
                    target_support_position=int(target_pos),
                )
                trial_entries.append(
                    {
                        "trial_id": int(len(trial_entries)),
                        "bank_id": int(bank_id),
                        "draw_index": int(draw_index),
                        "support_id": int(support_id),
                        "support": support,
                        "trial": trial,
                        "target_position": int(target_pos),
                        "target_source_id": str(trial.target_source_id),
                        "center_source_id": str(support.center_source_id),
                        "support_hash": str(support.support_hash),
                        "bank_size": int(bank.X.shape[0]),
                    }
                )

    if args.max_trials is not None:
        trial_entries = trial_entries[: int(args.max_trials)]

    trials_df = pd.DataFrame(
        [
            {
                "trial_id": e["trial_id"],
                "support_id": e["support_id"],
                "bank_id": e["bank_id"],
                "draw_index": e["draw_index"],
                "center_source_id": e["center_source_id"],
                "target_position": e["target_position"],
                "target_source_id": e["target_source_id"],
                "support_hash": e["support_hash"],
                "bank_size": e["bank_size"],
                "oblivious_kappa": float(e["support"].oblivious_kappa),
            }
            for e in trial_entries
        ]
    )
    return trial_entries, pd.DataFrame(support_rows), trials_df


def summarize_outputs(
    utility_df: pd.DataFrame, attack_df: pd.DataFrame, bound_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    utility_summary_rows = []
    for model, grp in utility_df.groupby("model"):
        mu, lo, hi = mean_ci(grp["accuracy"])
        utility_summary_rows.append(
            {
                "model": model,
                "n_releases": int(len(grp)),
                "mean_accuracy": mu,
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                "mean_sigma": float(np.nanmean(grp["sigma"])),
                "mean_epsilon_ball_view": float(np.nanmean(grp["epsilon_ball_view"])),
                "mean_epsilon_standard_view": float(
                    np.nanmean(grp["epsilon_standard_view"])
                ),
                "mean_delta_ball": float(np.nanmean(grp["delta_ball"])),
                "mean_delta_standard": float(np.nanmean(grp["delta_standard"])),
            }
        )
    utility_summary = pd.DataFrame(utility_summary_rows)

    for missing_col in ["instance_bound_finite_opt", "instance_bound_center_max"]:
        if missing_col not in bound_df.columns:
            bound_df[missing_col] = np.nan

    attack_ok = attack_df[attack_df["source_exact_id"].notna()].copy()
    attack_summary_rows = []
    for model, grp in attack_ok.groupby("model"):
        n = int(len(grp))
        k = int(np.sum(np.asarray(grp["source_exact_id"], dtype=float)))
        lo, hi = wilson_interval(k, n)
        attack_summary_rows.append(
            {
                "model": model,
                "n_trials": n,
                "empirical_exact_id": float(k / max(n, 1)),
                "exact_id_ci_low": lo,
                "exact_id_ci_high": hi,
                "mean_prior_rank": float(np.nanmean(grp["prior_rank"])),
                "mean_posterior_true": float(
                    np.nanmean(grp["posterior_true_probability"])
                ),
                "mean_posterior_top1": float(
                    np.nanmean(grp["posterior_top1_probability"])
                ),
                "chance_kappa": float(np.nanmean(grp["baseline_kappa"])),
                "mean_accuracy": float(np.nanmean(grp["accuracy"])),
            }
        )
    attack_summary = pd.DataFrame(attack_summary_rows)

    bound_summary = bound_df.groupby(["model", "bound_mode"], as_index=False).agg(
        n_trials=("kappa", "size"),
        mean_kappa=("kappa", "mean"),
        mean_gamma_ball=("gamma_ball", "mean"),
        mean_gamma_standard=("gamma_standard", "mean"),
        mean_instance_finite_opt=("instance_bound_finite_opt", "mean"),
        mean_instance_center_max=("instance_bound_center_max", "mean"),
    )

    attack_with_bounds = attack_summary.merge(
        bound_summary[bound_summary["bound_mode"] == "gaussian_direct"][
            [
                "model",
                "mean_gamma_ball",
                "mean_gamma_standard",
                "mean_instance_finite_opt",
                "mean_instance_center_max",
            ]
        ],
        on="model",
        how="left",
    )
    return utility_summary, attack_summary, bound_summary, attack_with_bounds


def make_figures(
    out_dir: Path,
    utility_summary: pd.DataFrame,
    attack_with_bounds: pd.DataFrame,
    posterior_df: pd.DataFrame,
) -> None:
    fig_dir = ensure_dir(out_dir / "figures")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)

    # Utility.
    ax = axes[0]
    sub = (
        utility_summary[utility_summary["model"].isin(MODEL_ORDER)]
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(how="all")
    )
    x = np.arange(len(sub))
    y = sub["mean_accuracy"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - sub["accuracy_ci_low"].to_numpy(dtype=float),
            sub["accuracy_ci_high"].to_numpy(dtype=float) - y,
        ]
    )
    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        color=[MODEL_COLORS[m] for m in sub.index],
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"public accuracy")
    ax.set_title(r"Convex noisy-ERM utility")

    # Attack and bounds.
    ax = axes[1]
    sub = (
        attack_with_bounds[attack_with_bounds["model"].isin(MODEL_ORDER)]
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(how="all")
    )
    x = np.arange(len(sub))
    y = sub["empirical_exact_id"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - sub["exact_id_ci_low"].to_numpy(dtype=float),
            sub["exact_id_ci_high"].to_numpy(dtype=float) - y,
        ]
    )
    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        alpha=0.82,
        color=[MODEL_COLORS[m] for m in sub.index],
        label=r"empirical exact-ID",
    )
    if "mean_instance_finite_opt" in sub:
        ax.scatter(
            x,
            sub["mean_instance_finite_opt"],
            s=80,
            marker="D",
            color="black",
            label=r"finite Gaussian bound",
        )
    if "mean_gamma_ball" in sub:
        ax.scatter(
            x,
            sub["mean_gamma_ball"],
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.8,
            label=r"generic Ball bound",
        )
    chance = float(np.nanmean(sub["chance_kappa"])) if len(sub) else float("nan")
    if np.isfinite(chance):
        ax.axhline(
            chance,
            linestyle="--",
            color="black",
            linewidth=1.2,
            label=rf"chance $\kappa={chance:.3f}$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"exact-ID probability")
    ax.set_title(r"Finite-prior attacks vs. bounds")
    ax.legend(loc="upper right")

    # Privacy/utility scatter.
    ax = axes[2]
    for model in MODEL_ORDER:
        row = attack_with_bounds[attack_with_bounds["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        ax.scatter(
            row["mean_instance_finite_opt"],
            row["mean_accuracy"],
            s=90,
            color=MODEL_COLORS[model],
            label=rf"{model}: finite",
        )
        if np.isfinite(row.get("mean_gamma_ball", np.nan)):
            ax.scatter(
                row["mean_gamma_ball"],
                row["mean_accuracy"],
                s=90,
                marker="o",
                facecolors="none",
                edgecolors=MODEL_COLORS[model],
                linewidths=1.8,
                label=rf"{model}: generic",
            )
        ax.text(
            float(row["mean_instance_finite_opt"]) + 0.01,
            float(row["mean_accuracy"]) + 0.005,
            model,
            fontsize=9,
        )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"finite-prior upper bound $\Gamma$")
    ax.set_ylabel(r"mean public accuracy")
    ax.set_title(r"Utility vs. finite-prior privacy bound")
    ax.legend(loc="lower right")

    savefig_stem(fig, fig_dir / "convex_summary")
    plt.close(fig)

    if not posterior_df.empty:
        example_trial = posterior_df["trial_id"].iloc[0]
        fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
        for model, sub in posterior_df[
            posterior_df["trial_id"] == example_trial
        ].groupby("model"):
            sub = sub.sort_values("candidate_position")
            ax.plot(
                sub["candidate_position"],
                sub["posterior_probability"],
                marker="o",
                label=model,
                color=MODEL_COLORS.get(model, None),
            )
        ax.axhline(
            float(posterior_df["baseline_kappa"].iloc[0]),
            linestyle="--",
            color="black",
            label=r"uniform prior $1/m$",
        )
        true_pos = int(
            posterior_df[posterior_df["trial_id"] == example_trial][
                "target_position"
            ].iloc[0]
        )
        ax.axvline(true_pos, linestyle=":", color="black", label=r"true target")
        ax.set_xlabel(r"support candidate index")
        ax.set_ylabel(r"posterior probability")
        ax.set_title(r"Example convex finite-prior posterior")
        ax.legend(loc="upper right")
        savefig_stem(fig, fig_dir / "convex_example_posterior")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-samples", type=int, default=1600)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.40)
    parser.add_argument("--class-sep", type=float, default=2.8)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--embedding-bound", type=float, default=5.0)

    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--standard-radius", type=float, default=None)
    parser.add_argument("--prior-size", type=int, default=6)
    parser.add_argument("--num-supports", type=int, default=8)
    parser.add_argument("--support-draws", type=int, default=2)
    parser.add_argument(
        "--support-selection",
        type=str,
        default="farthest",
        choices=["random", "nearest", "farthest"],
    )
    parser.add_argument(
        "--target-policy", type=str, default="all", choices=["all", "sample"]
    )
    parser.add_argument("--targets-per-support", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=None)

    parser.add_argument("--model-family", type=str, default="ridge_prototype")
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver", type=str, default="lbfgs_fullbatch")
    parser.add_argument("--ridge-sensitivity-mode", type=str, default="global")
    parser.add_argument(
        "--orders", type=float, nargs="+", default=list(range(2, 65)) + [80, 96, 128]
    )

    parser.add_argument(
        "--skip-curves",
        action="store_true",
        help="Reserved for symmetry with other runners; finite bounds are still computed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    write_json(out_dir / "config.json", vars(args))

    X_train, y_train, X_public, y_public = make_synthetic_embeddings(args)
    num_classes = int(len(np.unique(np.concatenate([y_train, y_public]))))
    feature_dim = int(X_train.shape[1])
    standard_radius = (
        2.0 * float(args.embedding_bound)
        if args.standard_radius is None
        else float(args.standard_radius)
    )

    data_summary = pd.DataFrame(
        [
            {
                "split": "private_train",
                "n": len(X_train),
                "feature_dim": feature_dim,
                "max_l2_norm": float(np.linalg.norm(X_train, axis=1).max()),
                "num_classes": num_classes,
            },
            {
                "split": "public_eval",
                "n": len(X_public),
                "feature_dim": feature_dim,
                "max_l2_norm": float(np.linalg.norm(X_public, axis=1).max()),
                "num_classes": num_classes,
            },
        ]
    )
    data_summary.to_csv(out_dir / "data_summary.csv", index=False)

    neighbor_counts = same_label_public_neighbor_counts(
        X_train, y_train, X_public, y_public, radius=float(args.radius)
    )
    feasibility = pd.DataFrame(
        [
            {
                "radius": float(args.radius),
                "prior_size": int(args.prior_size),
                "anchors_with_at_least_m_candidates": int(
                    np.sum(neighbor_counts >= int(args.prior_size))
                ),
                "max_candidate_count": int(np.max(neighbor_counts)),
                "median_candidate_count": float(np.median(neighbor_counts)),
                "mean_candidate_count": float(np.mean(neighbor_counts)),
            }
        ]
    )
    feasibility.to_csv(out_dir / "support_feasibility.csv", index=False)

    trial_entries, support_df, trials_df = build_trials(
        X_train, y_train, X_public, y_public, args
    )
    support_df.to_csv(out_dir / "supports.csv", index=False)
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    if not trial_entries:
        raise RuntimeError("No finite-prior trials were created.")

    utility_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []

    # Full-dataset utility reference.
    for mechanism, model in [
        ("erm", "ERM"),
        ("ball", "Ball-DP"),
        ("standard", "Std-DP"),
    ]:
        print(f"Fitting full-data utility release: {model}")
        release = fit_release(
            X=X_train,
            y=y_train,
            X_eval=X_public,
            y_eval=y_public,
            args=args,
            mechanism=mechanism,
            seed=int(args.seed + {"erm": 11, "ball": 13, "standard": 17}[mechanism]),
        )
        utility_rows.append(
            {
                **release_row(release, model=model, mechanism=mechanism, trial_id=None),
                "scope": "full_data",
            }
        )

    for entry in trial_entries:
        trial_id = int(entry["trial_id"])
        trial = entry["trial"]
        support = entry["support"]
        finite_prior = make_finite_identification_prior(
            support.X.reshape(support.m, -1),
            weights=support.weights,
        )
        print(
            f"trial {trial_id + 1:03d}/{len(trial_entries):03d} | target={entry['target_source_id']}"
        )

        for mech_id, (mechanism, model) in enumerate(
            [("ball", "Ball-DP"), ("standard", "Std-DP")]
        ):
            release_seed = int(args.seed + 100_000 * trial_id + 1_000 * mech_id)
            release = fit_release(
                X=trial.X_full,
                y=trial.y_full,
                X_eval=X_public,
                y_eval=y_public,
                args=args,
                mechanism=mechanism,
                seed=release_seed,
            )
            utility_rows.append(
                {
                    **release_row(
                        release, model=model, mechanism=mechanism, trial_id=trial_id
                    ),
                    "scope": "replacement_trial",
                }
            )

            for mode in ["gaussian_direct", "rdp", "dp"]:
                try:
                    report = ball_rero(
                        release, prior=finite_prior, eta_grid=(0.5,), mode=mode
                    )
                    point = report.points[0]
                    bound_row = {
                        "trial_id": trial_id,
                        "model": model,
                        "mechanism": mechanism,
                        "bound_mode": mode,
                        "kappa": float(point.kappa),
                        "gamma_ball": float(point.gamma_ball),
                        "gamma_standard": (
                            np.nan
                            if point.gamma_standard is None
                            else float(point.gamma_standard)
                        ),
                    }
                except Exception as exc:
                    bound_row = {
                        "trial_id": trial_id,
                        "model": model,
                        "mechanism": mechanism,
                        "bound_mode": mode,
                        "kappa": float(support.oblivious_kappa),
                        "gamma_ball": np.nan,
                        "gamma_standard": np.nan,
                        "error": str(exc),
                    }
                bound_rows.append(bound_row)

            attack = attack_convex_finite_prior_trial(
                release,
                trial,
                known_label=int(trial.support.center_y),
                eta_grid=(0.5,),
            )
            diagnostics = diagnose_convex_ball_output_finite_prior(
                release,
                trial.X_full,
                trial.y_full,
                target_index=int(trial.target_index),
                X_candidates=trial.support.X,
                y_candidates=trial.support.y,
                prior_weights=trial.support.weights,
                known_label=int(trial.support.center_y),
                center_features=trial.support.center_x,
                center_label=int(trial.support.center_y),
            )
            # Attach instance diagnostics to the gaussian_direct row.
            for row in reversed(bound_rows):
                if (
                    row.get("trial_id") == trial_id
                    and row.get("model") == model
                    and row.get("bound_mode") == "gaussian_direct"
                ):
                    row.update(
                        {
                            k: v
                            for k, v in diagnostics.items()
                            if isinstance(v, (int, float, np.integer, np.floating))
                        }
                    )
                    break

            attack_rows.append(
                {
                    "trial_id": trial_id,
                    "model": model,
                    "mechanism": mechanism,
                    "status": attack.status,
                    "accuracy": float(release.utility_metrics.get("accuracy", np.nan)),
                    "epsilon_ball_view": first_epsilon(release.privacy.ball),
                    "epsilon_standard_view": first_epsilon(release.privacy.standard),
                    "sigma": float(getattr(release.privacy.ball, "sigma", np.nan)),
                    "baseline_kappa": float(support.oblivious_kappa),
                    "source_exact_id": float(
                        attack.metrics.get(
                            "source_exact_identification_success", np.nan
                        )
                    ),
                    "feature_exact_id": float(
                        attack.metrics.get("exact_identification_success", np.nan)
                    ),
                    "prior_rank": float(attack.metrics.get("prior_rank", np.nan)),
                    "posterior_true_probability": float(
                        attack.metrics.get("posterior_true_probability", np.nan)
                    ),
                    "posterior_top1_probability": float(
                        attack.metrics.get("posterior_top1_probability", np.nan)
                    ),
                    "posterior_effective_candidates": float(
                        attack.metrics.get("posterior_effective_candidates", np.nan)
                    ),
                    "predicted_prior_index": attack.diagnostics.get(
                        "predicted_prior_index"
                    ),
                    "true_prior_index": attack.diagnostics.get("true_prior_index"),
                    "predicted_source_id": attack.diagnostics.get(
                        "predicted_source_id"
                    ),
                    "target_source_id": attack.diagnostics.get("target_source_id"),
                    "target_position": int(entry["target_position"]),
                    "support_hash": str(entry["support_hash"]),
                }
            )

            probs = attack.diagnostics.get("candidate_posterior_probabilities")
            if probs is not None:
                probs_arr = np.asarray(probs, dtype=float).reshape(-1)
                for idx, prob in enumerate(probs_arr):
                    posterior_rows.append(
                        {
                            "trial_id": trial_id,
                            "model": model,
                            "candidate_position": int(idx),
                            "posterior_probability": float(prob),
                            "is_target": int(idx == int(trial.target_support_position)),
                            "target_position": int(trial.target_support_position),
                            "baseline_kappa": float(support.oblivious_kappa),
                        }
                    )

    utility_df = pd.DataFrame(utility_rows)
    bound_df = pd.DataFrame(bound_rows)
    attack_df = pd.DataFrame(attack_rows)
    posterior_df = pd.DataFrame(posterior_rows)

    utility_df.to_csv(out_dir / "utility.csv", index=False)
    bound_df.to_csv(out_dir / "finite_bounds.csv", index=False)
    attack_df.to_csv(out_dir / "attacks.csv", index=False)
    posterior_df.to_csv(out_dir / "posterior.csv", index=False)

    utility_summary, attack_summary, bound_summary, attack_with_bounds = (
        summarize_outputs(
            utility_df[utility_df["scope"] == "replacement_trial"],
            attack_df,
            bound_df,
        )
    )
    utility_summary.to_csv(out_dir / "utility_summary.csv", index=False)
    attack_summary.to_csv(out_dir / "attack_summary.csv", index=False)
    bound_summary.to_csv(out_dir / "finite_bound_summary.csv", index=False)
    attack_with_bounds.to_csv(out_dir / "attack_with_bounds.csv", index=False)

    make_figures(out_dir, utility_summary, attack_with_bounds, posterior_df)

    # Geometry quick-look figure for the first support.
    first_support = trial_entries[0]["support"]
    pca = PCA(n_components=2, random_state=int(args.seed))
    all_points = np.concatenate(
        [X_public, first_support.center_x[None, :], first_support.X], axis=0
    )
    Z = pca.fit_transform(all_points)
    Z_public = Z[: len(X_public)]
    z_center = Z[len(X_public)]
    Z_support = Z[len(X_public) + 1 :]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.scatter(
        Z_public[:, 0], Z_public[:, 1], s=12, alpha=0.18, label=r"public candidates"
    )
    ax.scatter(
        [z_center[0]],
        [z_center[1]],
        s=190,
        marker="*",
        edgecolor="black",
        label=r"center $u$",
    )
    ax.scatter(
        Z_support[:, 0], Z_support[:, 1], s=90, edgecolor="black", label=r"support $S$"
    )
    ax.set_xlabel(r"principal component $1$")
    ax.set_ylabel(r"principal component $2$")
    ax.set_title(r"Representative convex finite-prior support")
    ax.legend(loc="best")
    savefig_stem(fig, fig_dir / "convex_support_geometry")
    plt.close(fig)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
