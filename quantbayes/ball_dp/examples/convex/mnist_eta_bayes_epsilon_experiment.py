#!/usr/bin/env python3
"""MNIST finite-prior eta-Bayes reconstruction experiment over epsilons.

Prerequisite source-level additions:
  1. quantbayes/ball_dp/attacks/finite_prior_eta.py exists.
  2. quantbayes/ball_dp/attacks/__init__.py exports the finite-prior eta helpers.

This script evaluates three finite-prior decoders:

  1. exact_map:
       Exact-ID MAP candidate, evaluated under eta-success.

  2. support_eta_bayes:
       Bayes-optimal eta decoder restricted to output one support point.

  3. arbitrary_eta_bayes:
       Bayes-optimal eta decoder allowed to output any embedding center.
       For m=10 this is computed exactly by subset enumeration:
           max_{A subset S : rad(A) <= eta} posterior(A).

The main thesis figure should usually use arbitrary_eta_bayes, because it
matches the finite-support arbitrary-center baseline kappa_S(eta).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# Optional repo-root path convenience
# ============================================================

THIS_FILE = Path(__file__).resolve()
for parent in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (parent / "quantbayes").exists() and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
        break


# ============================================================
# QuantBayes imports
# ============================================================

from quantbayes.ball_dp import fit_convex
from quantbayes.ball_dp.api import attack_convex_finite_prior_trial
from quantbayes.ball_dp.attacks.finite_prior_eta import (
    build_finite_support_eta_geometry,
    eta_grid_from_geometry,
    evaluate_eta_decoders,
    finite_support_kappa_rows,
    posterior_probabilities_from_attack_result,
)
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trials_for_support,
    select_support_from_bank,
)
from quantbayes.ball_dp.evaluation.rero import gaussian_direct_ball_rero_bound
from quantbayes.ball_dp.experiments.load_mnist_embeddings import (
    load_or_create_mnist_resnet18_embeddings,
)


# ============================================================
# Plot constants
# ============================================================

MECH_ORDER = ["q50", "q80", "q95", "standard"]

MECH_LABELS = {
    "q50": r"Ball $q50$",
    "q80": r"Ball $q80$",
    "q95": r"Ball $q95$",
    "standard": r"Standard",
}

MECH_COLORS = {
    "q50": "#0072B2",
    "q80": "#009E73",
    "q95": "#D55E00",
    "standard": "#3A3A3A",
}

MECH_LINESTYLES = {
    "q50": "-",
    "q80": "-",
    "q95": "-",
    "standard": "--",
}

DECODER_ORDER = ["exact_map", "support_eta_bayes", "arbitrary_eta_bayes"]

DECODER_LABELS = {
    "exact_map": r"exact-ID MAP under $\eta$",
    "support_eta_bayes": r"support $\eta$-Bayes",
    "arbitrary_eta_bayes": r"arbitrary-center $\eta$-Bayes",
}

DECODER_LINESTYLES = {
    "exact_map": "--",
    "support_eta_bayes": "-.",
    "arbitrary_eta_bayes": "-",
}


# ============================================================
# Generic utilities
# ============================================================

def configure_matplotlib(*, use_latex: bool = False) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "figure.constrained_layout.use": True,
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.75,
            "lines.linewidth": 2.20,
            "lines.markersize": 4.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": bool(use_latex),
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
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")

    phat = float(k) / float(n)
    denom = 1.0 + z * z / float(n)
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mu = float(np.mean(arr))

    if arr.size == 1:
        return mu, mu, mu

    half = float(z * np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mu, mu - half, mu + half


def first_epsilon(ledger: Any) -> float:
    certs = getattr(ledger, "dp_certificates", None)
    if certs:
        return float(certs[0].epsilon)

    cert = getattr(ledger, "dp_certificate", None)
    if cert is not None:
        return float(cert.epsilon)

    return float("nan")


# ============================================================
# MNIST loading
# ============================================================

def _device_get(x: Any) -> Any:
    try:
        import jax

        return jax.device_get(x)
    except Exception:
        return x


def stratified_subsample_indices(
    y: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    weights = counts / counts.sum()

    raw = weights * int(n_total)
    per_label = np.floor(raw).astype(int)

    remainder = int(n_total) - int(per_label.sum())
    if remainder > 0:
        order = np.argsort(-(raw - per_label))
        for j in order[:remainder]:
            per_label[j] += 1

    chosen: list[np.ndarray] = []
    for label, k in zip(labels, per_label):
        idx = np.where(y == label)[0]
        k = min(int(k), len(idx))
        if k > 0:
            chosen.append(rng.choice(idx, size=k, replace=False))

    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out.astype(int)


def load_mnist_embeddings(
    *,
    max_private_train: int | None,
    max_public: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    print("Loading MNIST ResNet18 embeddings...")

    X_train, y_train, X_public, y_public = load_or_create_mnist_resnet18_embeddings()

    X_train = np.asarray(_device_get(X_train), dtype=np.float32)
    y_train = np.asarray(_device_get(y_train), dtype=np.int32).reshape(-1)
    X_public = np.asarray(_device_get(X_public), dtype=np.float32)
    y_public = np.asarray(_device_get(y_public), dtype=np.int32).reshape(-1)

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32, copy=False)
    X_public = X_public.reshape(X_public.shape[0], -1).astype(np.float32, copy=False)

    label_values = np.unique(np.concatenate([y_train, y_public]))
    label_map = {int(v): i for i, v in enumerate(label_values.tolist())}

    y_train = np.asarray([label_map[int(v)] for v in y_train], dtype=np.int32)
    y_public = np.asarray([label_map[int(v)] for v in y_public], dtype=np.int32)

    rng = np.random.default_rng(int(seed))

    if max_private_train is not None and int(max_private_train) < len(X_train):
        idx = stratified_subsample_indices(y_train, int(max_private_train), rng)
        X_train = X_train[idx]
        y_train = y_train[idx]

    if max_public is not None and int(max_public) < len(X_public):
        idx = stratified_subsample_indices(y_public, int(max_public), rng)
        X_public = X_public[idx]
        y_public = y_public[idx]

    meta = {
        "num_classes": int(len(np.unique(y_train))),
        "feature_dim": int(X_train.shape[1]),
        "private_train_n": int(len(X_train)),
        "public_n": int(len(X_public)),
        "label_values_original": label_values.astype(int).tolist(),
        "max_train_norm": float(np.linalg.norm(X_train, axis=1).max()),
        "max_public_norm": float(np.linalg.norm(X_public, axis=1).max()),
    }

    return X_train, y_train, X_public, y_public, meta


# ============================================================
# Radius estimation
# ============================================================

def sample_same_label_pairwise_distances(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_sampled_pairs: int,
    seed: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    rng = np.random.default_rng(int(seed))
    labels = np.unique(y)

    label_indices = [np.where(y == int(label))[0] for label in labels]
    pair_counts = np.asarray(
        [len(idx) * (len(idx) - 1) // 2 for idx in label_indices],
        dtype=float,
    )

    total_pairs = float(pair_counts.sum())
    if total_pairs <= 0:
        raise ValueError("No same-label pairs exist.")

    distances: list[np.ndarray] = []

    for j, idx in enumerate(label_indices):
        n = len(idx)
        if n < 2:
            continue

        s = max(1, int(round(pair_counts[j] / total_pairs * int(max_sampled_pairs))))

        a = rng.integers(0, n, size=s)
        b = rng.integers(0, n - 1, size=s)
        b = b + (b >= a)

        d = np.linalg.norm(X[idx[a]] - X[idx[b]], axis=1)
        distances.append(d.astype(np.float64))

    out = np.concatenate(distances)
    out = out[np.isfinite(out)]
    if out.size == 0:
        raise ValueError("Radius sampling produced no finite distances.")

    return out


def sample_global_pairwise_distances(
    X: np.ndarray,
    *,
    max_sampled_pairs: int,
    seed: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = len(X)

    if n < 2:
        return np.array([0.0], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    s = int(max_sampled_pairs)

    a = rng.integers(0, n, size=s)
    b = rng.integers(0, n - 1, size=s)
    b = b + (b >= a)

    d = np.linalg.norm(X[a] - X[b], axis=1).astype(np.float64)
    return d[np.isfinite(d)]


def compute_radius_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    quantiles: Sequence[float],
    max_sampled_pairs: int,
    standard_radius: float | None,
    standard_radius_source: Literal["2B", "sampled_empirical_diameter"],
    embedding_bound: float,
    seed: int,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    same_label_d = sample_same_label_pairwise_distances(
        X_train,
        y_train,
        max_sampled_pairs=int(max_sampled_pairs),
        seed=int(seed),
    )

    radius_values: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for q in quantiles:
        tag = f"q{int(round(100.0 * float(q)))}"
        value = float(np.quantile(same_label_d, float(q)))
        radius_values[tag] = value
        rows.append(
            {
                "radius_tag": tag,
                "quantile": float(q),
                "radius": value,
                "source": "sampled_same_label_pairwise_train_distances",
                "n_distances": int(len(same_label_d)),
            }
        )

    if standard_radius is not None:
        std = float(standard_radius)
        std_source = "user_supplied"
        std_n = 0
    elif standard_radius_source == "2B":
        std = 2.0 * float(embedding_bound)
        std_source = "2_times_embedding_bound"
        std_n = 0
    else:
        global_d = sample_global_pairwise_distances(
            X_train,
            max_sampled_pairs=int(max_sampled_pairs),
            seed=int(seed) + 1729,
        )
        std = float(np.max(global_d))
        std_source = "sampled_global_train_pairwise_max"
        std_n = int(len(global_d))

    rows.append(
        {
            "radius_tag": "standard",
            "quantile": np.nan,
            "radius": std,
            "source": std_source,
            "n_distances": std_n,
        }
    )

    return radius_values, std, pd.DataFrame(rows)


# ============================================================
# Fast support-bank prefilter
# ============================================================

def select_prefilter_positions(
    *,
    d2_inside: np.ndarray,
    support_selection: str,
    max_prefilter_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(len(d2_inside))
    cap = int(max_prefilter_candidates)

    if n <= cap:
        return np.arange(n, dtype=int)

    if support_selection == "farthest":
        return np.argsort(-d2_inside)[:cap]

    if support_selection == "nearest":
        return np.argsort(d2_inside)[:cap]

    return rng.choice(n, size=cap, replace=False).astype(int)


def find_fast_anchor_and_public_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
    *,
    radius: float,
    m: int,
    support_selection: str,
    seed: int,
    max_anchor_trials: int,
    max_prefilter_candidates: int,
    fast_anchor_policy: Literal["first_feasible", "largest_sampled"],
) -> tuple[int, np.ndarray, pd.DataFrame]:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_public = np.asarray(X_public, dtype=np.float32)
    y_public = np.asarray(y_public, dtype=np.int32)

    rng = np.random.default_rng(int(seed))
    anchor_order = rng.permutation(len(X_train))
    anchor_order = anchor_order[: min(int(max_anchor_trials), len(anchor_order))]

    public_by_label = {
        int(label): np.where(y_public == int(label))[0]
        for label in np.unique(y_public)
    }

    r2 = float(radius) ** 2

    best_anchor: int | None = None
    best_public_idx: np.ndarray | None = None
    best_count = -1
    checked = 0

    rows = []

    for checked, anchor_idx in enumerate(anchor_order.tolist(), start=1):
        label = int(y_train[int(anchor_idx)])
        pub_idx = public_by_label.get(label)

        if pub_idx is None or len(pub_idx) < int(m):
            continue

        diffs = X_public[pub_idx] - X_train[int(anchor_idx)][None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        inside_local = np.where(d2 <= r2 + 1e-8)[0]
        count = int(len(inside_local))

        if count > best_count:
            keep_local = select_prefilter_positions(
                d2_inside=d2[inside_local],
                support_selection=str(support_selection),
                max_prefilter_candidates=int(max_prefilter_candidates),
                rng=rng,
            )
            best_count = count
            best_anchor = int(anchor_idx)
            best_public_idx = pub_idx[inside_local[keep_local]]

        if checked % 500 == 0:
            print(
                f"  checked {checked} anchors; "
                f"best same-label public bank size so far = {best_count}"
            )

        if count >= int(m) and fast_anchor_policy == "first_feasible":
            keep_local = select_prefilter_positions(
                d2_inside=d2[inside_local],
                support_selection=str(support_selection),
                max_prefilter_candidates=int(max_prefilter_candidates),
                rng=rng,
            )
            selected_public = pub_idx[inside_local[keep_local]]
            rows.append(
                {
                    "checked_anchors": int(checked),
                    "selected_anchor_index": int(anchor_idx),
                    "selected_anchor_label": int(label),
                    "full_bank_size_inside_radius": int(count),
                    "prefiltered_public_candidates": int(len(selected_public)),
                    "fast_anchor_policy": str(fast_anchor_policy),
                }
            )
            return int(anchor_idx), selected_public.astype(int), pd.DataFrame(rows)

    if best_anchor is None or best_public_idx is None or best_count < int(m):
        raise RuntimeError(
            f"Could not find a feasible support bank within {checked} checked anchors. "
            f"Best bank size was {best_count}, but m={int(m)}."
        )

    rows.append(
        {
            "checked_anchors": int(checked),
            "selected_anchor_index": int(best_anchor),
            "selected_anchor_label": int(y_train[int(best_anchor)]),
            "full_bank_size_inside_radius": int(best_count),
            "prefiltered_public_candidates": int(len(best_public_idx)),
            "fast_anchor_policy": str(fast_anchor_policy),
        }
    )

    return int(best_anchor), best_public_idx.astype(int), pd.DataFrame(rows)


def build_fixed_support_and_trials(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
    *,
    support_radius: float,
    m: int,
    support_selection: str,
    target_policy: str,
    targets_per_support: int,
    seed: int,
    fast_anchor_policy: Literal["first_feasible", "largest_sampled"],
    fast_anchor_trials: int,
    max_prefilter_candidates: int,
) -> tuple[Any, list[Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Fast prefilter: finding explicit anchor and small public candidate source...")

    anchor_idx, public_idx, prefilter_df = find_fast_anchor_and_public_subset(
        X_train,
        y_train,
        X_public,
        y_public,
        radius=float(support_radius),
        m=int(m),
        support_selection=str(support_selection),
        seed=int(seed),
        max_anchor_trials=int(fast_anchor_trials),
        max_prefilter_candidates=int(max_prefilter_candidates),
        fast_anchor_policy=fast_anchor_policy,
    )

    print(
        f"Selected anchor_index={anchor_idx}, "
        f"label={int(y_train[anchor_idx])}, "
        f"prefiltered_public_candidates={len(public_idx)}"
    )

    public_source = CandidateSource(
        "public",
        np.asarray(X_public[public_idx], dtype=np.float32),
        np.asarray(y_public[public_idx], dtype=np.int32),
        indices=public_idx,
    )

    banks = find_feasible_replacement_banks(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int32),
        candidate_sources=[public_source],
        radius=float(support_radius),
        min_support_size=int(m),
        num_banks=1,
        seed=int(seed),
        max_search=1,
        explicit_anchor_indices=[int(anchor_idx)],
        anchor_selection="random",
        strict=True,
    )

    bank = banks[0]

    support = select_support_from_bank(
        bank,
        m=int(m),
        selection=str(support_selection),
        seed=int(seed),
        draw_index=0,
    )

    trials = make_replacement_trials_for_support(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int32),
        support=support,
        target_policy=str(target_policy),
        num_targets=None if target_policy == "all" else int(targets_per_support),
        seed=int(seed),
    )

    trial_rows = []
    for trial_id, trial in enumerate(trials):
        trial_rows.append(
            {
                "trial_id": int(trial_id),
                "target_support_position": int(trial.target_support_position),
                "target_source_id": str(trial.target_source_id),
                "support_hash": str(support.support_hash),
                "center_source_id": str(support.center_source_id),
                "center_index": int(support.center_index),
                "center_y": int(support.center_y),
                "support_m": int(support.m),
                "oblivious_kappa": float(support.oblivious_kappa),
            }
        )

    support_rows = []
    for pos in range(int(support.m)):
        support_rows.append(
            {
                "support_position": int(pos),
                "source_id": str(support.source_ids[pos]),
                "label": int(support.y[pos]),
                "distance_to_center": float(support.distances_to_center[pos]),
                "prior_weight": float(support.weights[pos]),
                "support_hash": str(support.support_hash),
                "center_source_id": str(support.center_source_id),
                "center_index": int(support.center_index),
                "center_y": int(support.center_y),
                "radius": float(support.radius),
                "selection": str(support.selection),
            }
        )

    return (
        support,
        trials,
        pd.DataFrame(trial_rows),
        pd.DataFrame(support_rows),
        prefilter_df,
    )


# ============================================================
# Release fitting and attack loop
# ============================================================

def fit_release_for_policy(
    trial: Any,
    *,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
    epsilon: float,
    release_radius: float,
    standard_radius: float,
    embedding_bound: float,
    model_family: str,
    ridge_sensitivity_mode: str,
    lam: float,
    delta: float,
    orders: Sequence[float],
    max_iter: int,
    solver: str,
    seed: int,
) -> Any:
    return fit_convex(
        np.asarray(trial.X_full, dtype=np.float32),
        np.asarray(trial.y_full, dtype=np.int32),
        X_eval=np.asarray(X_eval, dtype=np.float32),
        y_eval=np.asarray(y_eval, dtype=np.int32),
        model_family=str(model_family),
        privacy="ball_dp",
        radius=float(release_radius),
        standard_radius=float(standard_radius),
        ridge_sensitivity_mode=str(ridge_sensitivity_mode),
        lam=float(lam),
        epsilon=float(epsilon),
        delta=float(delta),
        embedding_bound=float(embedding_bound),
        num_classes=int(num_classes),
        orders=tuple(float(v) for v in orders),
        max_iter=int(max_iter),
        solver=str(solver),
        seed=int(seed),
    )


def ridge_center_signal_diagnostics(
    *,
    trial: Any,
    sigma: float,
    lam: float,
) -> dict[str, float]:
    support = trial.support

    n_total = int(len(trial.y_full))
    n_label_minus = int(np.sum(np.asarray(trial.D_minus_y) == int(support.center_y)))

    A_y = 2.0 * (float(n_label_minus) + 1.0) + float(lam) * float(n_total)

    center_sensitivities = (
        2.0
        * np.linalg.norm(
            np.asarray(support.X, dtype=np.float64)
            - np.asarray(support.center_x, dtype=np.float64).reshape(1, -1),
            axis=1,
        )
        / A_y
    )

    if sigma > 0.0:
        center_d = center_sensitivities / float(sigma)
    else:
        center_d = np.full_like(center_sensitivities, np.nan, dtype=float)

    return {
        "ridge_A_y": float(A_y),
        "center_effective_sensitivity": float(np.max(center_sensitivities)),
        "center_dmax": float(np.nanmax(center_d)),
        "center_dmean": float(np.nanmean(center_d)),
    }


def run_epsilon_sweep(
    *,
    support: Any,
    trials: list[Any],
    eta_geometry: Any,
    eta_grid: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    radius_values: dict[str, float],
    standard_radius: float,
    epsilon_grid: Sequence[float],
    release_seeds: Sequence[int],
    embedding_bound: float,
    model_family: str,
    ridge_sensitivity_mode: str,
    lam: float,
    delta: float,
    orders: Sequence[float],
    max_iter: int,
    solver: str,
    num_classes: int,
    base_seed: int,
    decoder_modes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    radius_policy_values = {
        "q50": float(radius_values["q50"]),
        "q80": float(radius_values["q80"]),
        "q95": float(radius_values["q95"]),
        "standard": float(standard_radius),
    }

    release_rows: list[dict[str, Any]] = []
    exact_map_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    total_jobs = (
        len(trials)
        * len(MECH_ORDER)
        * len(tuple(epsilon_grid))
        * len(tuple(release_seeds))
    )
    job = 0

    for trial_id, trial in enumerate(trials):
        target_pos = int(trial.target_support_position)

        for eps_id, epsilon in enumerate(epsilon_grid):
            for mech_id, radius_policy in enumerate(MECH_ORDER):
                release_radius = float(radius_policy_values[radius_policy])

                for release_seed_user in release_seeds:
                    job += 1

                    release_seed = int(
                        base_seed
                        + 10_000_000 * eps_id
                        + 1_000_000 * mech_id
                        + 10_000 * trial_id
                        + int(release_seed_user)
                    )

                    print(
                        f"[{job:04d}/{total_jobs:04d}] "
                        f"trial={trial_id} target={target_pos} "
                        f"eps={float(epsilon):g} policy={radius_policy} "
                        f"seed={int(release_seed_user)}"
                    )

                    release = fit_release_for_policy(
                        trial,
                        X_eval=X_eval,
                        y_eval=y_eval,
                        num_classes=int(num_classes),
                        epsilon=float(epsilon),
                        release_radius=release_radius,
                        standard_radius=float(standard_radius),
                        embedding_bound=float(embedding_bound),
                        model_family=str(model_family),
                        ridge_sensitivity_mode=str(ridge_sensitivity_mode),
                        lam=float(lam),
                        delta=float(delta),
                        orders=orders,
                        max_iter=int(max_iter),
                        solver=str(solver),
                        seed=release_seed,
                    )

                    sigma = float(release.privacy.ball.sigma)
                    sensitivity = float(release.sensitivity.delta_ball)
                    accuracy = float(release.utility_metrics.get("accuracy", np.nan))

                    center_diag = ridge_center_signal_diagnostics(
                        trial=trial,
                        sigma=sigma,
                        lam=float(lam),
                    )

                    common = {
                        "trial_id": int(trial_id),
                        "target_support_position": int(target_pos),
                        "epsilon": float(epsilon),
                        "radius_policy": str(radius_policy),
                        "release_seed": int(release_seed_user),
                        "release_radius": float(release_radius),
                        "support_hash": str(support.support_hash),
                        "support_m": int(support.m),
                    }

                    release_rows.append(
                        {
                            **common,
                            "sigma": sigma,
                            "sensitivity": sensitivity,
                            "delta_over_sigma": float(sensitivity / sigma),
                            "accuracy": accuracy,
                            "epsilon_ball_view": first_epsilon(release.privacy.ball),
                            "epsilon_standard_view": first_epsilon(release.privacy.standard),
                            "oblivious_kappa": float(support.oblivious_kappa),
                            **center_diag,
                        }
                    )

                    attack = attack_convex_finite_prior_trial(
                        release,
                        trial,
                        known_label=int(support.center_y),
                        eta_grid=(0.0,),
                    )

                    posterior = posterior_probabilities_from_attack_result(attack)
                    true_idx = int(trial.target_support_position)
                    pred_idx = int(attack.diagnostics["predicted_prior_index"])

                    pred_dist = float(
                        eta_geometry.pairwise_distances[true_idx, pred_idx]
                    )

                    exact_map_rows.append(
                        {
                            **common,
                            "true_prior_index": int(true_idx),
                            "predicted_prior_index": int(pred_idx),
                            "prediction_distance": pred_dist,
                            "exact_id": float(pred_idx == true_idx),
                            "source_exact_id": float(
                                attack.metrics.get(
                                    "source_exact_identification_success",
                                    np.nan,
                                )
                            ),
                            "posterior_true_probability": float(posterior[true_idx]),
                            "posterior_top1_probability": float(np.max(posterior)),
                            "posterior_effective_candidates": float(
                                np.exp(
                                    -np.sum(
                                        posterior
                                        * np.log(np.maximum(posterior, 1e-300))
                                    )
                                )
                            ),
                            "target_source_id": str(
                                attack.diagnostics.get(
                                    "target_source_id",
                                    trial.target_source_id,
                                )
                            ),
                            "predicted_source_id": str(
                                attack.diagnostics.get("predicted_source_id")
                            ),
                        }
                    )

                    for i, prob in enumerate(posterior.tolist()):
                        posterior_rows.append(
                            {
                                **common,
                                "candidate_position": int(i),
                                "posterior_probability": float(prob),
                                "is_target": int(i == true_idx),
                                "is_exact_map": int(i == pred_idx),
                            }
                        )

                    eta_decisions = evaluate_eta_decoders(
                        eta_geometry,
                        posterior,
                        eta_grid,
                        true_index=true_idx,
                        modes=tuple(decoder_modes),
                    )

                    for row in eta_decisions:
                        decision_rows.append(
                            {
                                **common,
                                "decoder_mode": str(row["decoder_mode"]),
                                "eta": float(row["eta"]),
                                "posterior_success_probability": float(
                                    row["posterior_success_probability"]
                                ),
                                "empirical_success": float(row["empirical_success"]),
                                "predicted_prior_index": (
                                    np.nan
                                    if row["predicted_prior_index"] is None
                                    else int(row["predicted_prior_index"])
                                ),
                                "predicted_subset_mask": int(row["predicted_subset_mask"]),
                                "predicted_subset_size": int(row["predicted_subset_size"]),
                                "true_index": int(true_idx),
                            }
                        )

    return (
        pd.DataFrame(release_rows),
        pd.DataFrame(exact_map_rows),
        pd.DataFrame(posterior_rows),
        pd.DataFrame(decision_rows),
    )


# ============================================================
# Aggregation
# ============================================================

def per_release_gamma_summary(
    *,
    kappa: float,
    sensitivities: np.ndarray,
    sigmas: np.ndarray,
) -> dict[str, float]:
    vals = []

    for sens, sigma in zip(sensitivities, sigmas):
        if np.isfinite(sens) and np.isfinite(sigma) and sigma > 0.0:
            vals.append(
                gaussian_direct_ball_rero_bound(
                    kappa=float(kappa),
                    sensitivity=float(sens),
                    sigma=float(sigma),
                )
            )

    if not vals:
        return {"mean": float("nan"), "max": float("nan")}

    arr = np.asarray(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "max": float(np.max(arr))}


def aggregate_eta_curves(
    decisions_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    kappa_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = decisions_df.merge(
        releases_df[
            [
                "trial_id",
                "target_support_position",
                "epsilon",
                "radius_policy",
                "release_seed",
                "sigma",
                "sensitivity",
                "center_effective_sensitivity",
                "delta_over_sigma",
                "accuracy",
                "center_dmax",
            ]
        ],
        on=[
            "trial_id",
            "target_support_position",
            "epsilon",
            "radius_policy",
            "release_seed",
        ],
        how="left",
    )

    kappa_lookup = kappa_df.set_index("eta").to_dict(orient="index")
    rows = []

    for (decoder_mode, radius_policy, epsilon, eta), frame in merged.groupby(
        ["decoder_mode", "radius_policy", "epsilon", "eta"]
    ):
        successes = frame["empirical_success"].to_numpy(dtype=float)
        successes = successes[np.isfinite(successes)]

        n = int(len(successes))
        k = int(np.sum(successes))
        empirical = float(k / max(n, 1))
        lo, hi = wilson_interval(k, n)

        sigmas = frame["sigma"].to_numpy(dtype=float)
        sensitivities = frame["sensitivity"].to_numpy(dtype=float)
        center_sensitivities = frame["center_effective_sensitivity"].to_numpy(dtype=float)

        kappa_row = kappa_lookup[float(eta)]

        if decoder_mode == "arbitrary_eta_bayes":
            kappa = float(kappa_row["kappa_exact_arbitrary_center"])
        else:
            kappa = float(kappa_row["kappa_support_center"])

        gamma_release = per_release_gamma_summary(
            kappa=kappa,
            sensitivities=sensitivities,
            sigmas=sigmas,
        )
        gamma_center = per_release_gamma_summary(
            kappa=kappa,
            sensitivities=center_sensitivities,
            sigmas=sigmas,
        )

        rows.append(
            {
                "decoder_mode": str(decoder_mode),
                "radius_policy": str(radius_policy),
                "epsilon": float(epsilon),
                "eta": float(eta),
                "k_success": int(k),
                "n_trials": int(n),
                "empirical_success": empirical,
                "success_ci_low": lo,
                "success_ci_high": hi,
                "mean_posterior_success_probability": float(
                    np.nanmean(
                        frame["posterior_success_probability"].to_numpy(dtype=float)
                    )
                ),
                "kappa_matching_decoder": kappa,
                "kappa_support_center": float(kappa_row["kappa_support_center"]),
                "kappa_exact_arbitrary_center": float(
                    kappa_row["kappa_exact_arbitrary_center"]
                ),
                "kappa_upper_2eta": float(kappa_row["kappa_upper_2eta"]),
                "gamma_release_mean": gamma_release["mean"],
                "gamma_release_max": gamma_release["max"],
                "gamma_centermax_mean": gamma_center["mean"],
                "gamma_centermax_max": gamma_center["max"],
                "mean_sigma": float(np.nanmean(sigmas)),
                "mean_sensitivity": float(np.nanmean(sensitivities)),
                "mean_center_effective_sensitivity": float(
                    np.nanmean(center_sensitivities)
                ),
                "mean_delta_over_sigma": float(
                    np.nanmean(frame["delta_over_sigma"].to_numpy(dtype=float))
                ),
                "mean_accuracy": float(
                    np.nanmean(frame["accuracy"].to_numpy(dtype=float))
                ),
                "mean_center_dmax": float(
                    np.nanmean(frame["center_dmax"].to_numpy(dtype=float))
                ),
            }
        )

    out = pd.DataFrame(rows)
    out["radius_policy"] = pd.Categorical(
        out["radius_policy"],
        categories=MECH_ORDER,
        ordered=True,
    )
    out["decoder_mode"] = pd.Categorical(
        out["decoder_mode"],
        categories=DECODER_ORDER,
        ordered=True,
    )
    return out.sort_values(
        ["decoder_mode", "epsilon", "radius_policy", "eta"]
    ).reset_index(drop=True)


def selected_eta_summary(
    curves_df: pd.DataFrame,
    *,
    normalized_tolerances: Sequence[float],
    support_diameter: float,
) -> pd.DataFrame:
    rows = []

    for tau in normalized_tolerances:
        eta = float(tau) * float(support_diameter)

        sub = curves_df.copy()
        sub["eta_distance"] = np.abs(sub["eta"] - eta)

        nearest_eta = (
            sub.groupby(["decoder_mode", "radius_policy", "epsilon"], observed=True)[
                "eta_distance"
            ]
            .idxmin()
            .dropna()
            .astype(int)
        )

        chosen = sub.loc[nearest_eta].copy()
        chosen["requested_normalized_eta"] = float(tau)
        chosen["requested_eta"] = float(eta)
        rows.append(chosen)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(
        ["requested_normalized_eta", "decoder_mode", "epsilon", "radius_policy"]
    ).reset_index(drop=True)


def release_summary_by_epsilon(releases_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (radius_policy, epsilon), frame in releases_df.groupby(["radius_policy", "epsilon"]):
        mu, lo, hi = mean_ci(frame["accuracy"].to_numpy(dtype=float))

        rows.append(
            {
                "radius_policy": str(radius_policy),
                "epsilon": float(epsilon),
                "n_releases": int(len(frame)),
                "mean_accuracy": mu,
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                "mean_sigma": float(np.nanmean(frame["sigma"].to_numpy(dtype=float))),
                "mean_sensitivity": float(
                    np.nanmean(frame["sensitivity"].to_numpy(dtype=float))
                ),
                "mean_delta_over_sigma": float(
                    np.nanmean(frame["delta_over_sigma"].to_numpy(dtype=float))
                ),
                "mean_center_dmax": float(
                    np.nanmean(frame["center_dmax"].to_numpy(dtype=float))
                ),
            }
        )

    out = pd.DataFrame(rows)
    out["radius_policy"] = pd.Categorical(
        out["radius_policy"],
        categories=MECH_ORDER,
        ordered=True,
    )
    return out.sort_values(["epsilon", "radius_policy"]).reset_index(drop=True)


# ============================================================
# Figures
# ============================================================

def plot_arbitrary_eta_bayes_by_epsilon(
    curves_df: pd.DataFrame,
    kappa_df: pd.DataFrame,
    *,
    support_diameter: float,
    out_stem: Path,
) -> None:
    curves = curves_df[curves_df["decoder_mode"].astype(str) == "arbitrary_eta_bayes"]
    eps_values = sorted(curves["epsilon"].unique().tolist())

    cmap = plt.get_cmap("viridis")
    eps_colors = {
        eps: cmap(i / max(len(eps_values) - 1, 1))
        for i, eps in enumerate(eps_values)
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, mech in zip(axes, MECH_ORDER):
        for eps in eps_values:
            sub = curves[
                (curves["radius_policy"].astype(str) == mech)
                & (curves["epsilon"] == eps)
            ].sort_values("eta")

            if sub.empty:
                continue

            x = sub["eta"].to_numpy(dtype=float) / float(support_diameter)
            y = sub["empirical_success"].to_numpy(dtype=float)
            lo = sub["success_ci_low"].to_numpy(dtype=float)
            hi = sub["success_ci_high"].to_numpy(dtype=float)

            ax.step(
                x,
                y,
                where="post",
                color=eps_colors[eps],
                linewidth=2.1,
            )
            ax.fill_between(
                x,
                lo,
                hi,
                step="post",
                color=eps_colors[eps],
                alpha=0.12,
                linewidth=0,
            )

        ax.step(
            kappa_df["eta"].to_numpy(dtype=float) / float(support_diameter),
            kappa_df["kappa_exact_arbitrary_center"].to_numpy(dtype=float),
            where="post",
            color="black",
            linestyle=":",
            linewidth=1.6,
            alpha=0.75,
        )

        ax.set_title(MECH_LABELS[mech])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r"$\eta/\operatorname{diam}(S)$")
        ax.set_ylabel(r"$\Pr[\|Z-\widehat Z_\eta\|_2\leq\eta]$")

    eps_handles = [
        Line2D(
            [0],
            [0],
            color=eps_colors[eps],
            linewidth=2.3,
            label=rf"$\varepsilon={eps:g}$",
        )
        for eps in eps_values
    ]
    kappa_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=r"$\kappa_S(\eta)$",
    )

    fig.legend(
        handles=eps_handles + [kappa_handle],
        loc="lower center",
        ncol=len(eps_values) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    savefig_stem(fig, out_stem)
    plt.close(fig)


def plot_decoder_comparison_for_epsilon(
    curves_df: pd.DataFrame,
    *,
    support_diameter: float,
    epsilon: float,
    out_stem: Path,
) -> None:
    eps = float(epsilon)

    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, mech in zip(axes, MECH_ORDER):
        for decoder in DECODER_ORDER:
            sub = curves_df[
                (curves_df["epsilon"] == eps)
                & (curves_df["radius_policy"].astype(str) == mech)
                & (curves_df["decoder_mode"].astype(str) == decoder)
            ].sort_values("eta")

            if sub.empty:
                continue

            x = sub["eta"].to_numpy(dtype=float) / float(support_diameter)
            y = sub["empirical_success"].to_numpy(dtype=float)

            ax.step(
                x,
                y,
                where="post",
                color=MECH_COLORS[mech],
                linestyle=DECODER_LINESTYLES[decoder],
                linewidth=2.15,
            )

        ax.set_title(MECH_LABELS[mech])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r"$\eta/\operatorname{diam}(S)$")
        ax.set_ylabel(r"success probability")

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DECODER_LINESTYLES[d],
            linewidth=2.2,
            label=DECODER_LABELS[d],
        )
        for d in DECODER_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    savefig_stem(fig, out_stem)
    plt.close(fig)


def plot_lift_over_exact_map(
    curves_df: pd.DataFrame,
    *,
    support_diameter: float,
    out_stem: Path,
    decoder: str = "arbitrary_eta_bayes",
) -> None:
    base = curves_df[curves_df["decoder_mode"].astype(str) == "exact_map"]
    bayes = curves_df[curves_df["decoder_mode"].astype(str) == str(decoder)]

    merged = bayes.merge(
        base[
            [
                "radius_policy",
                "epsilon",
                "eta",
                "empirical_success",
            ]
        ],
        on=["radius_policy", "epsilon", "eta"],
        how="inner",
        suffixes=("_bayes", "_map"),
    )
    merged["lift"] = merged["empirical_success_bayes"] - merged["empirical_success_map"]

    eps_values = np.asarray(sorted(merged["epsilon"].unique()), dtype=float)
    eta_values = np.asarray(sorted(merged["eta"].unique()), dtype=float)
    eta_norm = eta_values / float(support_diameter)

    lift_min = float(np.nanmin(merged["lift"]))
    lift_max = float(np.nanmax(merged["lift"]))
    if lift_min < 0.0 < lift_max:
        norm = TwoSlopeNorm(vmin=lift_min, vcenter=0.0, vmax=lift_max)
        cmap = "coolwarm"
    else:
        norm = Normalize(vmin=min(0.0, lift_min), vmax=max(1e-12, lift_max))
        cmap = "magma"

    fig, axes = plt.subplots(
        1,
        len(MECH_ORDER),
        figsize=(3.0 * len(MECH_ORDER), 3.25),
        sharey=True,
    )

    last_im = None

    for ax, mech in zip(np.asarray(axes).reshape(-1), MECH_ORDER):
        sub = merged[merged["radius_policy"].astype(str) == mech]

        pivot = (
            sub.pivot_table(
                index="epsilon",
                columns="eta",
                values="lift",
                aggfunc="mean",
            )
            .reindex(index=eps_values, columns=eta_values)
            .to_numpy(dtype=float)
        )

        last_im = ax.imshow(
            pivot,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                float(eta_norm.min()),
                float(eta_norm.max()),
                -0.5,
                len(eps_values) - 0.5,
            ],
            cmap=cmap,
            norm=norm,
        )

        ax.set_title(MECH_LABELS[mech])
        ax.set_xlabel(r"$\eta/\operatorname{diam}(S)$")
        ax.set_yticks(np.arange(len(eps_values)))
        ax.set_yticklabels([f"{e:g}" for e in eps_values])

    axes_arr = np.asarray(axes).reshape(-1)
    axes_arr[0].set_ylabel(r"$\varepsilon$")

    cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), shrink=0.92, pad=0.015)
    cbar.set_label(r"success lift over exact-ID MAP diagnostic")

    savefig_stem(fig, out_stem)
    plt.close(fig)


def plot_success_vs_epsilon_at_tolerances(
    curves_df: pd.DataFrame,
    *,
    support_diameter: float,
    out_stem: Path,
    tolerances_to_plot: Sequence[float],
    decoder_mode: str = "arbitrary_eta_bayes",
) -> None:
    curves = curves_df[curves_df["decoder_mode"].astype(str) == str(decoder_mode)]

    rows = []
    for tau in tolerances_to_plot:
        eta = float(tau) * float(support_diameter)
        temp = curves.copy()
        temp["eta_distance"] = np.abs(temp["eta"] - eta)

        idx = (
            temp.groupby(["radius_policy", "epsilon"], observed=True)["eta_distance"]
            .idxmin()
            .dropna()
            .astype(int)
        )
        chosen = temp.loc[idx].copy()
        chosen["tau"] = float(tau)
        rows.append(chosen)

    selected = pd.concat(rows, axis=0, ignore_index=True)

    tolerances = [float(t) for t in tolerances_to_plot]
    n = len(tolerances)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.45 * ncols, 2.85 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, tau in zip(axes_arr, tolerances):
        sub_tau = selected[np.isclose(selected["tau"], tau)]

        for mech in MECH_ORDER:
            sub = sub_tau[sub_tau["radius_policy"].astype(str) == mech].sort_values("epsilon")
            if sub.empty:
                continue

            x = sub["epsilon"].to_numpy(dtype=float)
            y = sub["empirical_success"].to_numpy(dtype=float)
            lo = sub["success_ci_low"].to_numpy(dtype=float)
            hi = sub["success_ci_high"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                color=MECH_COLORS[mech],
                linestyle=MECH_LINESTYLES[mech],
                label=MECH_LABELS[mech],
            )
            ax.fill_between(x, lo, hi, color=MECH_COLORS[mech], alpha=0.10, linewidth=0)

        if np.all(np.asarray(sorted(curves["epsilon"].unique()), dtype=float) > 0.0):
            ax.set_xscale("log", base=2)

        ax.set_title(rf"$\eta/\operatorname{{diam}}(S)={tau:g}$")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"success probability")
        ax.set_ylim(0.0, 1.02)

    for ax in axes_arr[n:]:
        ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            color=MECH_COLORS[m],
            linestyle=MECH_LINESTYLES[m],
            marker="o",
            linewidth=2.0,
            label=MECH_LABELS[m],
        )
        for m in MECH_ORDER
    ]

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    savefig_stem(fig, out_stem)
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--epsilon-grid",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--release-seeds",
        type=int,
        nargs="+",
        default=list(range(5)),
    )

    parser.add_argument("--m-support", type=int, default=10)
    parser.add_argument(
        "--support-selection",
        type=str,
        default="farthest",
        choices=["random", "nearest", "farthest"],
    )
    parser.add_argument(
        "--target-policy",
        type=str,
        default="all",
        choices=["all", "sample"],
    )
    parser.add_argument("--targets-per-support", type=int, default=3)

    parser.add_argument(
        "--fast-anchor-policy",
        type=str,
        default="first_feasible",
        choices=["first_feasible", "largest_sampled"],
    )
    parser.add_argument("--fast-anchor-trials", type=int, default=5000)
    parser.add_argument("--max-prefilter-candidates", type=int, default=512)

    parser.add_argument(
        "--radius-quantiles",
        type=float,
        nargs="+",
        default=[0.50, 0.80, 0.95],
    )
    parser.add_argument("--radius-sample-pairs", type=int, default=250_000)

    parser.add_argument("--standard-radius", type=float, default=None)
    parser.add_argument(
        "--standard-radius-source",
        type=str,
        default="sampled_empirical_diameter",
        choices=["2B", "sampled_empirical_diameter"],
    )
    parser.add_argument("--embedding-bound", type=float, default=None)

    parser.add_argument("--model-family", type=str, default="ridge_prototype")
    parser.add_argument("--ridge-sensitivity-mode", type=str, default="count_aware")
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument(
        "--orders",
        type=float,
        nargs="+",
        default=list(range(2, 65)) + [80, 96, 128],
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver", type=str, default="lbfgs_fullbatch")

    parser.add_argument("--eta-dense-points", type=int, default=201)
    parser.add_argument("--include-eta-midpoints", action="store_true")
    parser.add_argument(
        "--decoder-modes",
        type=str,
        nargs="+",
        default=["exact_map", "support_eta_bayes", "arbitrary_eta_bayes"],
        choices=["exact_map", "support_eta_bayes", "arbitrary_eta_bayes"],
    )
    parser.add_argument(
        "--selected-normalized-eta",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument(
        "--plot-normalized-eta",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.25, 0.50],
    )
    parser.add_argument(
        "--comparison-epsilon",
        type=float,
        default=None,
        help="If omitted, uses max epsilon in the grid.",
    )

    parser.add_argument("--max-private-train", type=int, default=None)
    parser.add_argument("--max-public", type=int, default=None)
    parser.add_argument("--use-latex", action="store_true")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    configure_matplotlib(use_latex=bool(args.use_latex))

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    write_json(out_dir / "config.json", vars(args))

    X_train, y_train, X_public, y_public, data_meta = load_mnist_embeddings(
        max_private_train=args.max_private_train,
        max_public=args.max_public,
        seed=int(args.seed),
    )

    embedding_bound = (
        float(args.embedding_bound)
        if args.embedding_bound is not None
        else max(float(data_meta["max_train_norm"]), float(data_meta["max_public_norm"]))
    )
    data_meta["embedding_bound_used"] = float(embedding_bound)
    write_json(out_dir / "data_metadata.json", data_meta)

    radius_values, standard_radius, radius_df = compute_radius_values(
        X_train,
        y_train,
        quantiles=args.radius_quantiles,
        max_sampled_pairs=int(args.radius_sample_pairs),
        standard_radius=args.standard_radius,
        standard_radius_source=args.standard_radius_source,
        embedding_bound=float(embedding_bound),
        seed=int(args.seed),
    )

    required = {"q50", "q80", "q95"}
    missing = sorted(required.difference(radius_values))
    if missing:
        raise ValueError(
            f"Missing radius tags {missing}. Use --radius-quantiles 0.50 0.80 0.95."
        )

    radius_df.to_csv(out_dir / "radius_values.csv", index=False)
    write_json(
        out_dir / "radius_values.json",
        {
            "radius_values": radius_values,
            "standard_radius": float(standard_radius),
            "standard_radius_source": str(args.standard_radius_source),
            "support_radius_tag": "q50",
            "support_radius": float(radius_values["q50"]),
        },
    )

    print("\nData metadata")
    print(pd.Series(data_meta).to_string())

    print("\nRadius policies")
    print(radius_df.to_string(index=False))

    print("\nBuilding fixed support inside q50...")
    support, trials, trial_index_df, support_df, prefilter_df = build_fixed_support_and_trials(
        X_train,
        y_train,
        X_public,
        y_public,
        support_radius=float(radius_values["q50"]),
        m=int(args.m_support),
        support_selection=str(args.support_selection),
        target_policy=str(args.target_policy),
        targets_per_support=int(args.targets_per_support),
        seed=int(args.seed),
        fast_anchor_policy=args.fast_anchor_policy,
        fast_anchor_trials=int(args.fast_anchor_trials),
        max_prefilter_candidates=int(args.max_prefilter_candidates),
    )

    support_df.to_csv(out_dir / "support.csv", index=False)
    trial_index_df.to_csv(out_dir / "trial_index.csv", index=False)
    prefilter_df.to_csv(out_dir / "support_prefilter.csv", index=False)

    print("\nPrecomputing eta geometry...")
    eta_geometry = build_finite_support_eta_geometry(
        support.X,
        support.y,
        support.weights,
        max_m_for_exact_subsets=12,
    )

    eta_grid = eta_grid_from_geometry(
        eta_geometry,
        dense_points=int(args.eta_dense_points),
        include_midpoints=bool(args.include_eta_midpoints),
    )

    kappa_df = pd.DataFrame(finite_support_kappa_rows(eta_geometry, eta_grid))
    kappa_df["eta_normalized"] = kappa_df["eta"] / float(eta_geometry.diameter)
    kappa_df.to_csv(out_dir / "kappa_curves.csv", index=False)

    support_meta = {
        "support_hash": str(support.support_hash),
        "support_m": int(support.m),
        "support_diameter": float(eta_geometry.diameter),
        "support_radius": float(support.radius),
        "center_source_id": str(support.center_source_id),
        "center_index": int(support.center_index),
        "center_y": int(support.center_y),
        "oblivious_kappa": float(support.oblivious_kappa),
        "num_trials": int(len(trials)),
        "target_policy": str(args.target_policy),
        "support_selection": str(args.support_selection),
        "eta_grid_size": int(len(eta_grid)),
        "eta_decoder_modes": list(args.decoder_modes),
    }
    write_json(out_dir / "support_metadata.json", support_meta)

    print("\nFixed support metadata")
    print(pd.Series(support_meta).to_string())

    print("\nRunning epsilon sweep with eta-decoders...")
    releases_df, exact_map_df, posterior_df, decisions_df = run_epsilon_sweep(
        support=support,
        trials=trials,
        eta_geometry=eta_geometry,
        eta_grid=eta_grid,
        X_eval=X_public,
        y_eval=y_public,
        radius_values=radius_values,
        standard_radius=float(standard_radius),
        epsilon_grid=[float(e) for e in args.epsilon_grid],
        release_seeds=[int(s) for s in args.release_seeds],
        embedding_bound=float(embedding_bound),
        model_family=str(args.model_family),
        ridge_sensitivity_mode=str(args.ridge_sensitivity_mode),
        lam=float(args.lam),
        delta=float(args.delta),
        orders=[float(o) for o in args.orders],
        max_iter=int(args.max_iter),
        solver=str(args.solver),
        num_classes=int(data_meta["num_classes"]),
        base_seed=int(args.seed),
        decoder_modes=[str(v) for v in args.decoder_modes],
    )

    releases_df.to_csv(out_dir / "mnist_eta_bayes_releases.csv", index=False)
    exact_map_df.to_csv(out_dir / "mnist_eta_bayes_exact_map_attacks.csv", index=False)
    posterior_df.to_csv(out_dir / "mnist_eta_bayes_posteriors.csv", index=False)
    decisions_df.to_csv(out_dir / "mnist_eta_bayes_decisions.csv", index=False)

    print("\nAggregating eta-Bayes curves...")
    curves_df = aggregate_eta_curves(decisions_df, releases_df, kappa_df)
    curves_df["eta_normalized"] = curves_df["eta"] / float(eta_geometry.diameter)
    curves_df.to_csv(out_dir / "mnist_eta_bayes_curves.csv", index=False)

    selected_df = selected_eta_summary(
        curves_df,
        normalized_tolerances=[float(v) for v in args.selected_normalized_eta],
        support_diameter=float(eta_geometry.diameter),
    )
    selected_df.to_csv(out_dir / "mnist_eta_bayes_selected_tolerances.csv", index=False)

    release_summary_df = release_summary_by_epsilon(releases_df)
    release_summary_df.to_csv(out_dir / "mnist_eta_bayes_release_summary.csv", index=False)

    print("\nSelected tolerance summary")
    display_cols = [
        "requested_normalized_eta",
        "decoder_mode",
        "radius_policy",
        "epsilon",
        "empirical_success",
        "success_ci_low",
        "success_ci_high",
        "n_trials",
        "mean_posterior_success_probability",
        "mean_sigma",
        "mean_accuracy",
    ]
    print(selected_df[display_cols].to_string(index=False))

    comparison_epsilon = (
        float(args.comparison_epsilon)
        if args.comparison_epsilon is not None
        else float(max(args.epsilon_grid))
    )

    print("\nCreating figures...")
    plot_arbitrary_eta_bayes_by_epsilon(
        curves_df,
        kappa_df,
        support_diameter=float(eta_geometry.diameter),
        out_stem=fig_dir / "eta_bayes_arbitrary_by_epsilon_with_wilson_bands",
    )

    plot_decoder_comparison_for_epsilon(
        curves_df,
        support_diameter=float(eta_geometry.diameter),
        epsilon=float(comparison_epsilon),
        out_stem=fig_dir / f"eta_decoder_comparison_eps_{comparison_epsilon:g}",
    )

    plot_lift_over_exact_map(
        curves_df,
        support_diameter=float(eta_geometry.diameter),
        out_stem=fig_dir / "eta_bayes_lift_over_exact_map_heatmap",
        decoder="arbitrary_eta_bayes",
    )

    plot_success_vs_epsilon_at_tolerances(
        curves_df,
        support_diameter=float(eta_geometry.diameter),
        out_stem=fig_dir / "eta_bayes_success_vs_epsilon_selected_tolerances",
        tolerances_to_plot=[float(v) for v in args.plot_normalized_eta],
        decoder_mode="arbitrary_eta_bayes",
    )

    print("\nSaved outputs to:")
    print(f"  {out_dir}")

    print("\nMain figures:")
    print(f"  {fig_dir / 'eta_bayes_arbitrary_by_epsilon_with_wilson_bands'}.[pdf|png]")
    print(f"  {fig_dir / f'eta_decoder_comparison_eps_{comparison_epsilon:g}'}.[pdf|png]")
    print(f"  {fig_dir / 'eta_bayes_lift_over_exact_map_heatmap'}.[pdf|png]")
    print(f"  {fig_dir / 'eta_bayes_success_vs_epsilon_selected_tolerances'}.[pdf|png]")


if __name__ == "__main__":
    main()
