# ball_svd_rank_sweep_public_warmstart.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import jax.random as jr

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from quantbayes.ball_dp.api import (
    fit_ball_sgd,
    calibrate_ball_sgd_noise_multiplier,
    extract_privacy_epsilon,
    evaluate_release_classifier,
)
from quantbayes.ball_dp.nonconvex.models.ball_net import (
    make_ball_tanh_net,
    check_input_bound,
)
from quantbayes.ball_dp.nonconvex.models.ball_net_svd import (
    certified_tanh_svd_mlp_lz,
    full_hidden_rank,
    make_ball_svd_adam,
    make_ball_svd_projector,
    make_ball_svd_tanh_net,
    make_ball_svd_tanh_net_from_dense,
    svd_frobenius_energy_fraction,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUT_DIR = Path("artifacts/ball_svd_rank_sweep_public_warmstart")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
RANKS: list[int | None] = [4, 8, 16, 32, 64, None]  # None -> full rank

# Data / model
N_SAMPLES = 3000
D_IN = 64
H = 64
B = 1.0
A = 1.0
LAMBDA = 1.0
RADIUS = 1.0
CLIP_NORM = 1.0

# Public dense warm-start training
PUBLIC_DENSE_LR = 3e-2
PUBLIC_DENSE_STEPS = 250
PUBLIC_DENSE_BATCH = 128
PUBLIC_DENSE_EVAL_EVERY = 25
PUBLIC_DENSE_EVAL_BATCH = 512

# Private training
PRIVATE_LR = 3e-2
PRIVATE_STEPS = 250
PRIVATE_BATCH = 64
PRIVATE_EVAL_EVERY = 25
PRIVATE_EVAL_BATCH = 512

# Privacy
DELTA = 1e-5
EPSILON_TARGET = 3.0
ORDERS = (2, 3, 4, 5, 8, 16, 32, 64, 128)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def make_dataset(seed: int = 0):
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=D_IN,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.25,
        flip_y=0.03,
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Public preprocessing to enforce ||x||_2 <= B.
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(X_norms, 1e-12)

    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray, *, seed: int = 0):
    # 60% private train, 20% public pool, 20% test
    X_private, X_tmp, y_private, y_tmp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )

    X_public_pool, X_test, y_public_pool, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp,
    )

    # Split public pool into:
    #   50% warmstart-train, 25% warmstart-val, 25% release-eval
    X_pub_train, X_pub_tmp, y_pub_train, y_pub_tmp = train_test_split(
        X_public_pool,
        y_public_pool,
        test_size=0.5,
        random_state=seed,
        stratify=y_public_pool,
    )

    X_pub_val, X_pub_eval, y_pub_val, y_pub_eval = train_test_split(
        X_pub_tmp,
        y_pub_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_pub_tmp,
    )

    return {
        "X_private": X_private,
        "y_private": y_private,
        "X_pub_train": X_pub_train,
        "y_pub_train": y_pub_train,
        "X_pub_val": X_pub_val,
        "y_pub_val": y_pub_val,
        "X_pub_eval": X_pub_eval,
        "y_pub_eval": y_pub_eval,
        "X_test": X_test,
        "y_test": y_test,
    }


# -----------------------------------------------------------------------------
# Public dense warm-start
# -----------------------------------------------------------------------------


def train_public_dense_teacher(
    *,
    X_pub_train: np.ndarray,
    y_pub_train: np.ndarray,
    X_pub_val: np.ndarray,
    y_pub_val: np.ndarray,
    hidden_dim: int,
    seed: int,
):
    """Train a dense tanh model on public-only data.

    This is intentionally noiseless and unclipped:
      - privacy='noiseless'
      - clip_norm=inf
      - noise_multiplier=0
    """
    model = make_ball_tanh_net(
        d_in=X_pub_train.shape[1],
        hidden_dim=hidden_dim,
        key=jr.PRNGKey(seed),
        init_project=False,
    )
    optimizer = optax.adam(PUBLIC_DENSE_LR)

    release = fit_ball_sgd(
        model=model,
        optimizer=optimizer,
        X_train=X_pub_train,
        y_train=y_pub_train,
        X_eval=X_pub_val,
        y_eval=y_pub_val,
        radius=RADIUS,
        lz=None,
        privacy="noiseless",
        delta=None,
        epsilon=None,
        num_steps=PUBLIC_DENSE_STEPS,
        batch_size=PUBLIC_DENSE_BATCH,
        clip_norm=10e6,
        noise_multiplier=0.0,
        orders=ORDERS,
        loss_name="binary_logistic",
        checkpoint_selection="best_public_eval_accuracy",
        eval_every=PUBLIC_DENSE_EVAL_EVERY,
        eval_batch_size=PUBLIC_DENSE_EVAL_BATCH,
        seed=seed,
        key=jr.PRNGKey(seed),
        param_projector=None,
    )
    return release.payload, release.utility_metrics


# -----------------------------------------------------------------------------
# Private SVD runs
# -----------------------------------------------------------------------------


def make_private_svd_model(
    *,
    d_in: int,
    hidden_dim: int,
    rank: int | None,
    seed: int,
    init_mode: str,
    public_dense_model: Any | None,
):
    if init_mode == "random_basis":
        return make_ball_svd_tanh_net(
            d_in=d_in,
            hidden_dim=hidden_dim,
            rank=rank,
            key=jr.PRNGKey(seed),
            init_project=True,
            Lambda=LAMBDA,
            A=A,
        )

    if init_mode == "public_warmstart":
        if public_dense_model is None:
            raise ValueError(
                "public_dense_model must be provided for public_warmstart."
            )
        return make_ball_svd_tanh_net_from_dense(
            public_dense_model,
            rank=rank,
            init_project=True,
            Lambda=LAMBDA,
            A=A,
        )

    raise ValueError(f"Unknown init_mode={init_mode!r}.")


def fit_private_release(
    *,
    model,
    privacy: str,
    noise_multiplier: float,
    epsilon: float,
    X_private: np.ndarray,
    y_private: np.ndarray,
    X_pub_eval: np.ndarray,
    y_pub_eval: np.ndarray,
    lz: float,
    seed: int,
):
    optimizer = make_ball_svd_adam(model, learning_rate=PRIVATE_LR)
    projector = make_ball_svd_projector(Lambda=LAMBDA, A=A)

    return fit_ball_sgd(
        model=model,
        optimizer=optimizer,
        X_train=X_private,
        y_train=y_private,
        X_eval=X_pub_eval,
        y_eval=y_pub_eval,
        radius=RADIUS,
        lz=(lz if privacy.startswith("ball") else None),
        privacy=privacy,
        epsilon=epsilon,
        delta=DELTA,
        num_steps=PRIVATE_STEPS,
        batch_size=PRIVATE_BATCH,
        clip_norm=CLIP_NORM,
        noise_multiplier=noise_multiplier,
        orders=ORDERS,
        loss_name="binary_logistic",
        checkpoint_selection="best_public_eval_accuracy",
        eval_every=PRIVATE_EVAL_EVERY,
        eval_batch_size=PRIVATE_EVAL_BATCH,
        warn_if_ball_equals_standard=True,
        seed=seed,
        key=jr.PRNGKey(seed),
        param_projector=projector,
    )


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def rank_to_label(rank: int | None, *, d_in: int, hidden_dim: int) -> str:
    if rank is None:
        return "full"
    return str(int(rank))


def effective_rank(rank: int | None, *, d_in: int, hidden_dim: int) -> int:
    return full_hidden_rank(d_in, hidden_dim) if rank is None else int(rank)


def count_trainable_svd_params(rank_eff: int, hidden_dim: int) -> int:
    # trainable: s (rank_eff), hidden bias b (hidden_dim), output head a (hidden_dim)
    return int(rank_eff + hidden_dim + hidden_dim)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(
            ["init_mode", "privacy", "rank_label", "rank_effective"], as_index=False
        )
        .agg(
            n_runs=("test_accuracy", "size"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_public_eval_accuracy=("public_eval_accuracy", "mean"),
            std_public_eval_accuracy=("public_eval_accuracy", "std"),
            mean_noise_multiplier=("noise_multiplier", "mean"),
            mean_epsilon=("epsilon", "mean"),
            mean_energy_fraction=("svd_energy_fraction", "mean"),
        )
        .sort_values(["init_mode", "privacy", "rank_effective"])
        .reset_index(drop=True)
    )
    return out


def make_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["seed", "init_mode", "rank_label", "rank_effective"],
        columns="privacy",
        values="test_accuracy",
    ).reset_index()

    pivot["ball_minus_standard_test_acc"] = pivot["ball_dp"] - pivot["standard_dp"]

    gap = (
        pivot.groupby(["init_mode", "rank_label", "rank_effective"], as_index=False)
        .agg(
            mean_ball_minus_standard_test_acc=("ball_minus_standard_test_acc", "mean"),
            std_ball_minus_standard_test_acc=("ball_minus_standard_test_acc", "std"),
        )
        .sort_values(["init_mode", "rank_effective"])
        .reset_index(drop=True)
    )
    return gap


def make_warmstart_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["seed", "privacy", "rank_label", "rank_effective"],
        columns="init_mode",
        values="test_accuracy",
    ).reset_index()

    pivot["public_warmstart_minus_random_basis"] = (
        pivot["public_warmstart"] - pivot["random_basis"]
    )

    gain = (
        pivot.groupby(["privacy", "rank_label", "rank_effective"], as_index=False)
        .agg(
            mean_public_warmstart_gain=("public_warmstart_minus_random_basis", "mean"),
            std_public_warmstart_gain=("public_warmstart_minus_random_basis", "std"),
        )
        .sort_values(["privacy", "rank_effective"])
        .reset_index(drop=True)
    )
    return gain


def plot_accuracy_vs_rank(summary: pd.DataFrame, *, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (init_mode, privacy), g in summary.groupby(["init_mode", "privacy"]):
        g = g.sort_values("rank_effective")
        ax.errorbar(
            g["rank_effective"],
            g["mean_test_accuracy"],
            yerr=g["std_test_accuracy"].fillna(0.0),
            marker="o",
            capsize=4,
            label=f"{privacy} | {init_mode}",
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ball_minus_standard(gap_df: pd.DataFrame, *, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for init_mode, g in gap_df.groupby("init_mode"):
        g = g.sort_values("rank_effective")
        ax.errorbar(
            g["rank_effective"],
            g["mean_ball_minus_standard_test_acc"],
            yerr=g["std_ball_minus_standard_test_acc"].fillna(0.0),
            marker="o",
            capsize=4,
            label=init_mode,
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Ball - Standard test accuracy")
    ax.set_title("Ball advantage vs rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_warmstart_gain(gain_df: pd.DataFrame, *, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for privacy, g in gain_df.groupby("privacy"):
        g = g.sort_values("rank_effective")
        ax.errorbar(
            g["rank_effective"],
            g["mean_public_warmstart_gain"],
            yerr=g["std_public_warmstart_gain"].fillna(0.0),
            marker="o",
            capsize=4,
            label=privacy,
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Public warm-start - random basis")
    ax.set_title("Warm-start gain vs rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    X, y = make_dataset(seed=0)
    check_input_bound(X, B=B)

    split = split_dataset(X, y, seed=0)

    X_private = split["X_private"]
    y_private = split["y_private"]
    X_pub_train = split["X_pub_train"]
    y_pub_train = split["y_pub_train"]
    X_pub_val = split["X_pub_val"]
    y_pub_val = split["y_pub_val"]
    X_pub_eval = split["X_pub_eval"]
    y_pub_eval = split["y_pub_eval"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    lz = certified_tanh_svd_mlp_lz(A=A, Lambda=LAMBDA, B=B, H=H)

    # Rank-independent calibration for this theorem.
    nm_ball = float(
        calibrate_ball_sgd_noise_multiplier(
            dataset_size=len(X_private),
            radius=RADIUS,
            lz=lz,
            num_steps=PRIVATE_STEPS,
            batch_size=PRIVATE_BATCH,
            clip_norm=CLIP_NORM,
            target_epsilon=EPSILON_TARGET,
            delta=DELTA,
            privacy="ball_rdp",
            orders=ORDERS,
            lower=1e-3,
            upper=0.25,
            max_upper=128.0,
            num_bisection_steps=10,
        )["noise_multiplier"]
    )
    nm_std = float(
        calibrate_ball_sgd_noise_multiplier(
            dataset_size=len(X_private),
            radius=RADIUS,
            lz=None,
            num_steps=PRIVATE_STEPS,
            batch_size=PRIVATE_BATCH,
            clip_norm=CLIP_NORM,
            target_epsilon=EPSILON_TARGET,
            delta=DELTA,
            privacy="standard_rdp",
            orders=ORDERS,
            lower=1e-3,
            upper=0.25,
            max_upper=128.0,
            num_bisection_steps=10,
        )["noise_multiplier"]
    )

    meta = {
        "H": H,
        "A": A,
        "Lambda": LAMBDA,
        "B": B,
        "radius": RADIUS,
        "clip_norm": CLIP_NORM,
        "epsilon_target": EPSILON_TARGET,
        "delta": DELTA,
        "lz": lz,
        "noise_multiplier_ball": nm_ball,
        "noise_multiplier_standard": nm_std,
        "seeds": SEEDS,
        "ranks": ["full" if r is None else int(r) for r in RANKS],
    }
    (OUT_DIR / "config.json").write_text(json.dumps(meta, indent=2))

    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        print("=" * 80)
        print(f"Seed {seed}")
        print("=" * 80)

        public_dense_model, public_dense_metrics = train_public_dense_teacher(
            X_pub_train=X_pub_train,
            y_pub_train=y_pub_train,
            X_pub_val=X_pub_val,
            y_pub_val=y_pub_val,
            hidden_dim=H,
            seed=seed,
        )

        hidden_weight = public_dense_model.hidden.weight

        for rank in RANKS:
            rank_eff = effective_rank(rank, d_in=D_IN, hidden_dim=H)
            rank_lbl = rank_to_label(rank, d_in=D_IN, hidden_dim=H)
            energy_frac = svd_frobenius_energy_fraction(hidden_weight, rank)

            for init_mode in ("random_basis", "public_warmstart"):
                model_ball = make_private_svd_model(
                    d_in=D_IN,
                    hidden_dim=H,
                    rank=rank,
                    seed=seed,
                    init_mode=init_mode,
                    public_dense_model=public_dense_model,
                )
                model_std = make_private_svd_model(
                    d_in=D_IN,
                    hidden_dim=H,
                    rank=rank,
                    seed=seed,
                    init_mode=init_mode,
                    public_dense_model=public_dense_model,
                )

                release_ball = fit_private_release(
                    model=model_ball,
                    privacy="ball_dp",
                    noise_multiplier=nm_ball,
                    epsilon=EPSILON_TARGET,
                    X_private=X_private,
                    y_private=y_private,
                    X_pub_eval=X_pub_eval,
                    y_pub_eval=y_pub_eval,
                    lz=lz,
                    seed=seed,
                )
                release_std = fit_private_release(
                    model=model_std,
                    privacy="standard_dp",
                    noise_multiplier=nm_std,
                    epsilon=EPSILON_TARGET,
                    X_private=X_private,
                    y_private=y_private,
                    X_pub_eval=X_pub_eval,
                    y_pub_eval=y_pub_eval,
                    lz=lz,
                    seed=seed,
                )

                eval_ball = evaluate_release_classifier(
                    release_ball,
                    X_test,
                    y_test,
                    key=jr.PRNGKey(10_000 + seed),
                    batch_size=PRIVATE_EVAL_BATCH,
                )
                eval_std = evaluate_release_classifier(
                    release_std,
                    X_test,
                    y_test,
                    key=jr.PRNGKey(20_000 + seed),
                    batch_size=PRIVATE_EVAL_BATCH,
                )

                rows.append(
                    {
                        "seed": seed,
                        "rank_label": rank_lbl,
                        "rank_effective": rank_eff,
                        "init_mode": init_mode,
                        "privacy": "ball_dp",
                        "noise_multiplier": nm_ball,
                        "epsilon": extract_privacy_epsilon(
                            release_ball, accounting_view="ball"
                        ),
                        "public_eval_accuracy": release_ball.utility_metrics.get(
                            "public_eval_accuracy", float("nan")
                        ),
                        "test_accuracy": eval_ball["accuracy"],
                        "trainable_dim": count_trainable_svd_params(rank_eff, H),
                        "svd_energy_fraction": energy_frac,
                        "teacher_public_val_accuracy": public_dense_metrics.get(
                            "public_eval_accuracy", float("nan")
                        ),
                    }
                )

                rows.append(
                    {
                        "seed": seed,
                        "rank_label": rank_lbl,
                        "rank_effective": rank_eff,
                        "init_mode": init_mode,
                        "privacy": "standard_dp",
                        "noise_multiplier": nm_std,
                        "epsilon": extract_privacy_epsilon(
                            release_std, accounting_view="standard"
                        ),
                        "public_eval_accuracy": release_std.utility_metrics.get(
                            "public_eval_accuracy", float("nan")
                        ),
                        "test_accuracy": eval_std["accuracy"],
                        "trainable_dim": count_trainable_svd_params(rank_eff, H),
                        "svd_energy_fraction": energy_frac,
                        "teacher_public_val_accuracy": public_dense_metrics.get(
                            "public_eval_accuracy", float("nan")
                        ),
                    }
                )

                print(
                    f"seed={seed:2d} | rank={rank_lbl:>4s} | {init_mode:>16s} | "
                    f"ball={eval_ball['accuracy']:.4f} | std={eval_std['accuracy']:.4f}"
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["init_mode", "privacy", "rank_effective", "seed"]).reset_index(
        drop=True
    )
    df.to_csv(OUT_DIR / "raw_results.csv", index=False)

    summary = summarize_results(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    gap_df = make_gap_table(df)
    gap_df.to_csv(OUT_DIR / "ball_minus_standard_gap.csv", index=False)

    warmstart_gain_df = make_warmstart_gain_table(df)
    warmstart_gain_df.to_csv(OUT_DIR / "warmstart_gain.csv", index=False)

    plot_accuracy_vs_rank(summary, out_path=OUT_DIR / "test_accuracy_vs_rank.png")
    plot_ball_minus_standard(
        gap_df,
        out_path=OUT_DIR / "ball_minus_standard_gap_vs_rank.png",
    )
    plot_warmstart_gain(
        warmstart_gain_df,
        out_path=OUT_DIR / "warmstart_gain_vs_rank.png",
    )

    print()
    print("=" * 80)
    print("Saved:")
    print("=" * 80)
    print(OUT_DIR / "config.json")
    print(OUT_DIR / "raw_results.csv")
    print(OUT_DIR / "summary.csv")
    print(OUT_DIR / "ball_minus_standard_gap.csv")
    print(OUT_DIR / "warmstart_gain.csv")
    print(OUT_DIR / "test_accuracy_vs_rank.png")
    print(OUT_DIR / "ball_minus_standard_gap_vs_rank.png")
    print(OUT_DIR / "warmstart_gain_vs_rank.png")
    print()
    print(summary)
    print()
    print(gap_df)
    print()
    print(warmstart_gain_df)


if __name__ == "__main__":
    main()
