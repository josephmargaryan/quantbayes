#!/usr/bin/env python3
"""Thesis-scale decentralized Ball-DP observer experiment.

This runner is the decentralized counterpart of the convex/nonconvex thesis
scripts.  It uses the shared finite-prior support layer, then evaluates a
decentralized linear-Gaussian observer view induced by deterministic gossip.

The key comparison is mechanism-specific:

* Ball-DP uses local sensitivity ``min(L_z r, 2C)`` and therefore needs less
  local Gaussian noise at the same local public-transcript epsilon.
* Standard DP uses sensitivity ``2C``.
* Empirical exact-ID is averaged over supports, targets, observer modes, and
  noise draws.

Example
-------
python quantbayes/ball_dp/experiments/decentralized/run_thesis_experiment.py \
  --out-dir runs/decentralized_thesis_eps4 \
  --num-supports 8 \
  --support-draws 2 \
  --target-policy all \
  --target-epsilon 4.0

Quick smoke test:
python quantbayes/ball_dp/experiments/decentralized/run_thesis_experiment.py \
  --out-dir runs/decentralized_smoke \
  --max-trials 2 \
  --noise-draws 1
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
from matplotlib.lines import Line2D
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path.cwd()
if (REPO_ROOT / "quantbayes").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantbayes.ball_dp.accountants.rdp import RdpCurve, rdp_to_dp
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    enrich_attack_result_with_trial,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
    trial_true_record,
)
from quantbayes.ball_dp.decentralized.accounting import (
    LocalNodeSGDSchedule,
    account_linear_gaussian_observer,
    account_public_transcript_node_local_rdp,
)
from quantbayes.ball_dp.decentralized.attacks import (
    make_linear_gaussian_mean_fn,
    run_linear_gaussian_finite_prior_attack,
)
from quantbayes.ball_dp.decentralized.gossip import (
    constant_mixing_matrices,
    gossip_observer_noise_covariance,
    gossip_transfer_matrix,
    graph_distances,
    make_graph_adjacency,
    metropolis_mixing_matrix,
    selector_matrix,
)
from quantbayes.ball_dp.decentralized.prototypes import (
    partition_indices_iid,
    run_noisy_prototype_gossip,
)
from quantbayes.ball_dp.decentralized.rero import (
    direct_gaussian_rero_success_bound_from_sensitivity_sq,
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


def first_epsilon(view: Any) -> float:
    certs = getattr(view, "dp_certificates", None)
    if certs:
        return float(certs[0].epsilon)
    cert = getattr(view, "dp_certificate", None)
    if cert is not None:
        return float(cert.epsilon)
    return float("nan")


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
        n_classes=int(args.num_classes),
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
    X_train, y_train, X_public, y_public, *, radius: float
) -> np.ndarray:
    counts = np.zeros(len(X_train), dtype=np.int32)
    for label in sorted(np.unique(y_train).tolist()):
        train_idx = np.where(y_train == int(label))[0]
        X_pub_label = np.asarray(X_public[y_public == int(label)], dtype=np.float32)
        for idx in train_idx:
            d = np.linalg.norm(X_pub_label - X_train[idx], axis=1)
            counts[idx] = int(np.sum(d <= float(radius) + 1e-8))
    return counts


def clip_l2_vector(x: np.ndarray, clip_norm: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if norm <= float(clip_norm) or norm <= 1e-12:
        return x.astype(np.float32, copy=False)
    return (float(clip_norm) / norm * x).astype(np.float32, copy=False)


def make_sensitive_blocks_fn(*, rounds: int, clip_norm: float, decay: float = 1.0):
    def sensitive_blocks(x: np.ndarray, label: int) -> np.ndarray:
        del label
        x_clip = clip_l2_vector(x, clip_norm=float(clip_norm))
        return np.stack([float(decay**t) * x_clip for t in range(int(rounds))], axis=0)

    return sensitive_blocks


def block_sensitivities(
    mechanism: str,
    *,
    radius: float,
    rounds: int,
    clip_norm: float,
    lz: float,
    decay: float = 1.0,
) -> tuple[float, ...]:
    if mechanism == "ball":
        base = min(float(lz) * float(radius), 2.0 * float(clip_norm))
    elif mechanism == "standard":
        base = 2.0 * float(clip_norm)
    else:
        raise ValueError("mechanism must be 'ball' or 'standard'")
    return tuple(float(base * (decay**t)) for t in range(int(rounds)))


def make_local_schedule(
    args: argparse.Namespace, *, noise_multiplier: float, local_dataset_size: int
) -> LocalNodeSGDSchedule:
    return LocalNodeSGDSchedule.from_noise_multipliers(
        dataset_size=int(local_dataset_size),
        batch_sizes=(int(args.local_batch_size),) * int(args.rounds),
        clip_norms=(float(args.clip_norm),) * int(args.rounds),
        noise_multipliers=float(noise_multiplier),
        radius=float(args.radius),
        lz=float(args.feature_lipschitz),
        orders=tuple(float(v) for v in args.orders),
        dp_delta=float(args.dp_delta),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
    )


def local_public_transcript_epsilon(
    args: argparse.Namespace,
    *,
    mechanism: str,
    noise_multiplier: float,
    local_dataset_size: int,
) -> float:
    schedule = make_local_schedule(
        args,
        noise_multiplier=float(noise_multiplier),
        local_dataset_size=int(local_dataset_size),
    )
    bridge = account_public_transcript_node_local_rdp(
        schedule, attacked_node=int(args.attacked_node)
    )
    view = bridge.ledger.ball if mechanism == "ball" else bridge.ledger.standard
    return first_epsilon(view)


def calibrate_local_noise_multiplier(
    args: argparse.Namespace, *, mechanism: str, local_dataset_size: int
) -> float:
    target = float(args.target_epsilon)
    lo = float(args.noise_lower)
    hi = float(args.noise_upper)
    eps_hi = local_public_transcript_epsilon(
        args,
        mechanism=mechanism,
        noise_multiplier=hi,
        local_dataset_size=local_dataset_size,
    )
    while eps_hi > target:
        lo = hi
        hi *= 2.0
        if hi > float(args.max_noise_upper):
            raise RuntimeError(
                f"Failed to bracket local noise multiplier for mechanism={mechanism}. "
                f"last upper={hi:.6g}, epsilon={eps_hi:.6g}, target={target:.6g}"
            )
        eps_hi = local_public_transcript_epsilon(
            args,
            mechanism=mechanism,
            noise_multiplier=hi,
            local_dataset_size=local_dataset_size,
        )
    for _ in range(int(args.noise_bisection_steps)):
        mid = 0.5 * (lo + hi)
        eps_mid = local_public_transcript_epsilon(
            args,
            mechanism=mechanism,
            noise_multiplier=mid,
            local_dataset_size=local_dataset_size,
        )
        if eps_mid <= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def build_trials(X_train, y_train, X_public, y_public, args: argparse.Namespace):
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


def observer_nodes_from_mode(
    mode: str, *, attacked_node: int, distances: np.ndarray
) -> tuple[int, ...]:
    key = str(mode).lower().replace("-", "_")
    m = int(distances.shape[0])
    j = int(attacked_node)
    if key == "self":
        return (j,)
    if key in {"all", "global"}:
        return tuple(range(m))
    if key in {"all_except_self", "others"}:
        return tuple(i for i in range(m) if i != j)
    if key == "nearest":
        d = np.asarray(distances[j], dtype=float).copy()
        d[j] = np.inf
        return (int(np.argmin(d)),)
    if key == "farthest":
        d = np.asarray(distances[j], dtype=float).copy()
        d[j] = -np.inf
        d[~np.isfinite(d)] = -np.inf
        return (int(np.argmax(d)),)
    if key.startswith("node"):
        return (int(key.replace("node", "")),)
    raise ValueError(f"unknown observer mode {mode!r}")


def run_attack_trial(
    *,
    trial_entry: dict[str, Any],
    mechanism: str,
    model: str,
    observer_mode: str,
    noise_std: float,
    noise_draw: int,
    args: argparse.Namespace,
    WS: list[np.ndarray],
    distances: np.ndarray,
    feature_dim: int,
) -> dict[str, Any]:
    trial = trial_entry["trial"]
    support = trial_entry["support"]
    obs = observer_nodes_from_mode(
        observer_mode, attacked_node=int(args.attacked_node), distances=distances
    )
    S_A = selector_matrix(obs, num_nodes=int(args.num_nodes))
    H = gossip_transfer_matrix(
        WS, observer_selector=S_A, attacked_node=int(args.attacked_node)
    )
    K = gossip_observer_noise_covariance(
        WS,
        observer_selector=S_A,
        state_noise_stds=float(noise_std),
        observation_noise_std=0.0,
        jitter=float(args.jitter),
    )
    mean_fn = make_linear_gaussian_mean_fn(
        transfer_matrix=H,
        sensitive_blocks_fn=make_sensitive_blocks_fn(
            rounds=int(args.rounds),
            clip_norm=float(args.clip_norm),
            decay=float(args.sensitivity_decay),
        ),
    )
    truth = trial_true_record(trial)
    mean = np.asarray(mean_fn(truth.features, int(truth.label)), dtype=np.float64)
    rng = np.random.default_rng(
        int(
            args.seed
            + 1_000_003 * int(trial_entry["trial_id"])
            + 10_007 * noise_draw
            + (0 if mechanism == "ball" else 53)
        )
    )
    L = np.linalg.cholesky(K)
    noise = (L @ rng.normal(size=(K.shape[0], feature_dim))).reshape(-1)
    observed = mean + noise

    attack = run_linear_gaussian_finite_prior_attack(
        observed_view=observed,
        candidate_features=trial.support.X,
        candidate_labels=trial.support.y,
        mean_fn=mean_fn,
        covariance=K,
        prior_weights=trial.support.weights,
        true_record=truth,
        eta_grid=(0.25, 0.50, 1.00),
        covariance_mode="kron_eye",
    )
    attack = enrich_attack_result_with_trial(attack, trial)

    acct = account_linear_gaussian_observer(
        transfer_matrix=H,
        covariance=K,
        block_sensitivities=block_sensitivities(
            mechanism,
            radius=float(args.radius),
            rounds=int(args.rounds),
            clip_norm=float(args.clip_norm),
            lz=float(args.feature_lipschitz),
            decay=float(args.sensitivity_decay),
        ),
        parameter_dim=int(feature_dim),
        orders=tuple(float(v) for v in args.orders),
        radius=float(args.radius),
        dp_delta=float(args.dp_delta),
        attacked_node=int(args.attacked_node),
        observer=obs,
        covariance_mode="kron_eye",
        method="auto",
    )
    direct_bound = direct_gaussian_rero_success_bound_from_sensitivity_sq(
        kappa=float(support.oblivious_kappa),
        sensitivity_sq=float(acct.sensitivity_sq),
    )

    return {
        "trial_id": int(trial_entry["trial_id"]),
        "support_id": int(trial_entry["support_id"]),
        "model": model,
        "mechanism": mechanism,
        "observer_mode": observer_mode,
        "observer_nodes": json.dumps(tuple(int(v) for v in obs)),
        "noise_draw": int(noise_draw),
        "noise_std": float(noise_std),
        "status": attack.status,
        "baseline_kappa": float(support.oblivious_kappa),
        "source_exact_id": float(
            attack.metrics.get("source_exact_identification_success", np.nan)
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
        "predicted_prior_index": attack.diagnostics.get("predicted_prior_index"),
        "true_prior_index": attack.diagnostics.get("true_prior_index"),
        "predicted_source_id": attack.diagnostics.get("predicted_source_id"),
        "target_source_id": attack.diagnostics.get("target_source_id"),
        "target_position": int(trial_entry["target_position"]),
        "support_hash": str(trial_entry["support_hash"]),
        "sensitivity_sq": float(acct.sensitivity_sq),
        "transferred_sensitivity": math.sqrt(max(0.0, float(acct.sensitivity_sq))),
        "direct_exact_id_bound": float(direct_bound),
        "observer_dp_epsilon": first_epsilon(acct),
    }


def gaussian_mechanism_dp_epsilon(
    *, sensitivity: float, noise_std: float, delta: float, orders: Sequence[float]
) -> float:
    eps_rdp = tuple(
        0.5 * float(a) * (float(sensitivity) / float(noise_std)) ** 2 for a in orders
    )
    curve = RdpCurve(
        orders=tuple(float(a) for a in orders),
        epsilons=eps_rdp,
        source="single_gaussian_mechanism_rdp",
        radius=None,
    )
    return float(rdp_to_dp(curve, delta=float(delta), source="rdp_to_dp").epsilon)


def calibrate_gaussian_noise(
    *, sensitivity: float, target_epsilon: float, delta: float, orders: Sequence[float]
) -> float:
    lo, hi = 1e-6, max(1.0, float(sensitivity))
    while gaussian_mechanism_dp_epsilon(
        sensitivity=sensitivity, noise_std=hi, delta=delta, orders=orders
    ) > float(target_epsilon):
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("failed to bracket Gaussian noise calibration")
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        eps = gaussian_mechanism_dp_epsilon(
            sensitivity=sensitivity, noise_std=mid, delta=delta, orders=orders
        )
        if eps > float(target_epsilon):
            lo = mid
        else:
            hi = mid
    return float(hi)


def run_prototype_utility(
    X_train, y_train, X_public, y_public, W, args: argparse.Namespace
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    num_classes = int(len(np.unique(np.concatenate([y_train, y_public]))))
    shards = partition_indices_iid(
        len(X_train), int(args.num_nodes), seed=int(args.seed)
    )

    baseline = run_noisy_prototype_gossip(
        X_train=X_train,
        y_train=y_train,
        X_test=X_public,
        y_test=y_public,
        W=W,
        rounds=int(args.utility_rounds),
        num_classes=num_classes,
        clip_norm=float(args.clip_norm),
        noise_std=0.0,
        seed=int(args.seed),
        shards=shards,
    )
    rows.append(
        {
            "model": "Noiseless",
            "mechanism": "noiseless",
            "epsilon": np.inf,
            "noise_std": 0.0,
            "sensitivity": 0.0,
            "accuracy_mean": float(baseline.accuracy_mean),
            "accuracy_min": float(baseline.accuracy_min),
            "consensus_disagreement": float(baseline.consensus_disagreement),
        }
    )

    for mechanism, model in [("ball", "Ball-DP"), ("standard", "Std-DP")]:
        sensitivity = (
            min(
                float(args.feature_lipschitz) * float(args.radius),
                2.0 * float(args.clip_norm),
            )
            if mechanism == "ball"
            else 2.0 * float(args.clip_norm)
        )
        for eps in args.utility_eps_grid:
            sigma = calibrate_gaussian_noise(
                sensitivity=float(sensitivity),
                target_epsilon=float(eps),
                delta=float(args.dp_delta),
                orders=tuple(float(v) for v in args.orders),
            )
            result = run_noisy_prototype_gossip(
                X_train=X_train,
                y_train=y_train,
                X_test=X_public,
                y_test=y_public,
                W=W,
                rounds=int(args.utility_rounds),
                num_classes=num_classes,
                clip_norm=float(args.clip_norm),
                noise_std=float(sigma),
                seed=int(
                    args.seed
                    + 1000
                    + int(100 * float(eps))
                    + (0 if mechanism == "ball" else 17)
                ),
                shards=shards,
            )
            rows.append(
                {
                    "model": model,
                    "mechanism": mechanism,
                    "epsilon": float(eps),
                    "noise_std": float(sigma),
                    "sensitivity": float(sensitivity),
                    "accuracy_mean": float(result.accuracy_mean),
                    "accuracy_min": float(result.accuracy_min),
                    "consensus_disagreement": float(result.consensus_disagreement),
                }
            )
    return pd.DataFrame(rows)


def summarize_attacks(attack_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = attack_df[attack_df["source_exact_id"].notna()].copy()
    for (model, mechanism, observer_mode), grp in ok.groupby(
        ["model", "mechanism", "observer_mode"]
    ):
        n = int(len(grp))
        k = int(np.sum(np.asarray(grp["source_exact_id"], dtype=float)))
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "model": model,
                "mechanism": mechanism,
                "observer_mode": observer_mode,
                "n_trials": n,
                "empirical_exact_id": float(k / max(n, 1)),
                "exact_id_ci_low": lo,
                "exact_id_ci_high": hi,
                "mean_prior_rank": float(np.nanmean(grp["prior_rank"])),
                "mean_noise_std": float(np.nanmean(grp["noise_std"])),
                "chance_kappa": float(np.nanmean(grp["baseline_kappa"])),
                "mean_transferred_sensitivity": float(
                    np.nanmean(grp["transferred_sensitivity"])
                ),
                "mean_direct_exact_id_bound": float(
                    np.nanmean(grp["direct_exact_id_bound"])
                ),
                "mean_observer_dp_epsilon": float(
                    np.nanmean(grp["observer_dp_epsilon"])
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    out_dir: Path,
    attack_summary: pd.DataFrame,
    utility_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    fig_dir = ensure_dir(out_dir / "figures")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5), constrained_layout=True)

    # Empirical exact-ID by observer mode.
    ax = axes[0]
    observer_modes = list(dict.fromkeys(attack_summary["observer_mode"].tolist()))
    width = 0.38
    x = np.arange(len(observer_modes))
    for offset, model in [(-width / 2, "Ball-DP"), (width / 2, "Std-DP")]:
        sub = (
            attack_summary[attack_summary["model"] == model]
            .set_index("observer_mode")
            .reindex(observer_modes)
        )
        y = sub["empirical_exact_id"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["exact_id_ci_low"].to_numpy(dtype=float),
                sub["exact_id_ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=yerr,
            capsize=4,
            color=MODEL_COLORS[model],
            alpha=0.86,
            label=model,
        )
        ax.scatter(
            x + offset,
            sub["mean_direct_exact_id_bound"],
            s=55,
            marker="D",
            color="black",
            zorder=4,
        )
    chance = (
        float(np.nanmean(attack_summary["chance_kappa"]))
        if len(attack_summary)
        else float("nan")
    )
    if np.isfinite(chance):
        ax.axhline(
            chance,
            linestyle="--",
            color="black",
            linewidth=1.2,
            label=rf"chance $\kappa={chance:.3f}$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(observer_modes, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"empirical exact-ID success")
    ax.set_title(r"Decentralized finite-prior attacks")
    ax.legend(loc="upper right")

    # Utility/privacy.
    ax = axes[1]
    for model in ["Ball-DP", "Std-DP"]:
        sub = utility_df[utility_df["model"] == model].sort_values("epsilon")
        if sub.empty:
            continue
        ax.plot(
            sub["epsilon"],
            sub["accuracy_mean"],
            marker="o",
            linewidth=2.0,
            color=MODEL_COLORS[model],
            label=model,
        )
    base = utility_df[utility_df["model"] == "Noiseless"]
    if not base.empty:
        ax.axhline(
            float(base["accuracy_mean"].iloc[0]),
            linestyle="--",
            color="black",
            label=r"noiseless",
        )
    ax.set_xlabel(r"target $\varepsilon$")
    ax.set_ylabel(r"nearest-prototype accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(r"Prototype utility/privacy tradeoff")
    ax.legend(loc="lower right")

    # Sensitivity/graph heatmap.
    ax = axes[2]
    if not sensitivity_df.empty:
        pivot = sensitivity_df.pivot(
            index="observer_node",
            columns="attacked_node",
            values="ball_transferred_sensitivity",
        )
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
        ax.set_xlabel(r"attacked node $j$")
        ax.set_ylabel(r"observer node $a$")
        ax.set_title(r"Ball transferred sensitivity $c_{a\leftarrow j}$")
        ax.set_xticks(np.arange(pivot.shape[1]), [str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]), [str(i) for i in pivot.index])
        fig.colorbar(im, ax=ax, shrink=0.82)
    else:
        ax.set_axis_off()
    savefig_stem(fig, fig_dir / "decentralized_summary")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for model in ["Ball-DP", "Std-DP"]:
        sub = attack_summary[attack_summary["model"] == model]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_direct_exact_id_bound"],
            sub["empirical_exact_id"],
            s=80,
            color=MODEL_COLORS[model],
            label=model,
        )
        for row in sub.itertuples():
            ax.text(
                row.mean_direct_exact_id_bound + 0.01,
                row.empirical_exact_id + 0.005,
                str(row.observer_mode),
                fontsize=8.5,
            )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, label=r"$y=x$"
    )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"direct Gaussian bound $\Gamma$")
    ax.set_ylabel(r"empirical exact-ID success")
    ax.set_title(r"Empirical attack success vs. observer-specific bound")
    ax.legend(loc="lower right")
    savefig_stem(fig, fig_dir / "decentralized_attack_vs_bound")
    plt.close(fig)


def transferred_sensitivity_heatmap(
    args: argparse.Namespace,
    WS: list[np.ndarray],
    distances: np.ndarray,
    feature_dim: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for observer in range(int(args.num_nodes)):
        S_A = selector_matrix((observer,), num_nodes=int(args.num_nodes))
        K = gossip_observer_noise_covariance(
            WS,
            observer_selector=S_A,
            state_noise_stds=float(args.fixed_process_noise_std),
            observation_noise_std=0.0,
            jitter=float(args.jitter),
        )
        for attacked in range(int(args.num_nodes)):
            H = gossip_transfer_matrix(
                WS, observer_selector=S_A, attacked_node=int(attacked)
            )
            vals = {}
            for mechanism in ["ball", "standard"]:
                acct = account_linear_gaussian_observer(
                    transfer_matrix=H,
                    covariance=K,
                    block_sensitivities=block_sensitivities(
                        mechanism,
                        radius=float(args.radius),
                        rounds=int(args.rounds),
                        clip_norm=float(args.clip_norm),
                        lz=float(args.feature_lipschitz),
                        decay=float(args.sensitivity_decay),
                    ),
                    parameter_dim=int(feature_dim),
                    orders=tuple(float(v) for v in args.orders),
                    radius=float(args.radius),
                    dp_delta=None,
                    attacked_node=int(attacked),
                    observer=(int(observer),),
                    covariance_mode="kron_eye",
                    method="auto",
                )
                vals[f"{mechanism}_transferred_sensitivity"] = math.sqrt(
                    max(0.0, float(acct.sensitivity_sq))
                )
            rows.append(
                {"observer_node": int(observer), "attacked_node": int(attacked), **vals}
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-samples", type=int, default=1600)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.40)
    parser.add_argument("--class-sep", type=float, default=2.8)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--embedding-bound", type=float, default=5.0)

    parser.add_argument("--radius", type=float, default=2.0)
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

    parser.add_argument("--graph", type=str, default="path")
    parser.add_argument("--num-nodes", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--utility-rounds", type=int, default=10)
    parser.add_argument("--attacked-node", type=int, default=0)
    parser.add_argument("--lazy", type=float, default=0.0)
    parser.add_argument("--erdos-p", type=float, default=0.35)
    parser.add_argument(
        "--observer-modes",
        type=str,
        nargs="+",
        default=["self", "nearest", "farthest", "all_except_self"],
    )

    parser.add_argument("--target-epsilon", type=float, default=4.0)
    parser.add_argument("--dp-delta", type=float, default=1e-6)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-lipschitz", type=float, default=1.0)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--fixed-process-noise-std", type=float, default=5.0)
    parser.add_argument("--noise-lower", type=float, default=1e-3)
    parser.add_argument("--noise-upper", type=float, default=0.25)
    parser.add_argument("--max-noise-upper", type=float, default=1024.0)
    parser.add_argument("--noise-bisection-steps", type=int, default=20)
    parser.add_argument("--noise-draws", type=int, default=3)
    parser.add_argument("--sensitivity-decay", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument(
        "--orders", type=float, nargs="+", default=[2, 3, 4, 5, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--utility-eps-grid", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    out_dir = ensure_dir(args.out_dir)
    write_json(out_dir / "config.json", vars(args))

    X_train, y_train, X_public, y_public = make_synthetic_embeddings(args)
    feature_dim = int(X_train.shape[1])
    num_classes = int(len(np.unique(np.concatenate([y_train, y_public]))))
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

    ADJ = make_graph_adjacency(
        str(args.graph),
        num_nodes=int(args.num_nodes),
        erdos_p=float(args.erdos_p),
        seed=int(args.seed),
    )
    W = metropolis_mixing_matrix(ADJ, lazy=float(args.lazy))
    WS = constant_mixing_matrices(W, num_rounds=int(args.rounds))
    DIST = graph_distances(ADJ)
    pd.DataFrame(ADJ).to_csv(out_dir / "graph_adjacency.csv", index=False)
    pd.DataFrame(W).to_csv(out_dir / "mixing_matrix.csv", index=False)

    local_dataset_size = int(math.ceil(len(X_train) / int(args.num_nodes)))
    noise_rows = []
    noise_by_mechanism = {}
    for mechanism, model in [("ball", "Ball-DP"), ("standard", "Std-DP")]:
        nm = calibrate_local_noise_multiplier(
            args, mechanism=mechanism, local_dataset_size=local_dataset_size
        )
        eps = local_public_transcript_epsilon(
            args,
            mechanism=mechanism,
            noise_multiplier=nm,
            local_dataset_size=local_dataset_size,
        )
        noise_std = float(nm * float(args.clip_norm))
        noise_by_mechanism[mechanism] = noise_std
        noise_rows.append(
            {
                "model": model,
                "mechanism": mechanism,
                "noise_multiplier": nm,
                "noise_std": noise_std,
                "local_public_transcript_epsilon": eps,
            }
        )
    calibration_df = pd.DataFrame(noise_rows)
    calibration_df.to_csv(out_dir / "calibration.csv", index=False)

    utility_df = run_prototype_utility(X_train, y_train, X_public, y_public, W, args)
    utility_df.to_csv(out_dir / "prototype_utility.csv", index=False)

    attack_rows: list[dict[str, Any]] = []
    for entry in trial_entries:
        trial_id = int(entry["trial_id"])
        print(
            f"trial {trial_id + 1:03d}/{len(trial_entries):03d} | target={entry['target_source_id']}"
        )
        for observer_mode in args.observer_modes:
            for noise_draw in range(int(args.noise_draws)):
                for mechanism, model in [("ball", "Ball-DP"), ("standard", "Std-DP")]:
                    attack_rows.append(
                        run_attack_trial(
                            trial_entry=entry,
                            mechanism=mechanism,
                            model=model,
                            observer_mode=str(observer_mode),
                            noise_std=float(noise_by_mechanism[mechanism]),
                            noise_draw=int(noise_draw),
                            args=args,
                            WS=WS,
                            distances=DIST,
                            feature_dim=feature_dim,
                        )
                    )
    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(out_dir / "attacks.csv", index=False)

    attack_summary = summarize_attacks(attack_df)
    attack_summary.to_csv(out_dir / "attack_summary.csv", index=False)

    sensitivity_df = transferred_sensitivity_heatmap(args, WS, DIST, feature_dim)
    sensitivity_df.to_csv(out_dir / "transferred_sensitivity_heatmap.csv", index=False)

    make_figures(out_dir, attack_summary, utility_df, sensitivity_df)

    # Representative support geometry.
    fig_dir = ensure_dir(out_dir / "figures")
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
    ax.set_title(r"Representative decentralized finite-prior support")
    ax.legend(loc="best")
    savefig_stem(fig, fig_dir / "decentralized_support_geometry")
    plt.close(fig)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
