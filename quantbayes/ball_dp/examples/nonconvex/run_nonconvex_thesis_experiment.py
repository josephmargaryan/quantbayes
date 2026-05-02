#!/usr/bin/env python3
"""Thesis-scale nonconvex Ball-DP finite-prior transcript experiment.

Run from the repository root, for example:

    python quantbayes/ball_dp/experiments/nonconvex/run_thesis_experiment.py \
        --out-dir runs/nonconvex_thesis_eps12 \
        --num-supports 8 \
        --support-draws 2 \
        --target-policy all \
        --target-epsilon 12.0

The script writes raw trial-level CSVs, aggregate summaries, a config JSON,
and publication-oriented figures into the chosen run directory.

This script intentionally does not depend on notebooks. It is the batch runner
matching the aggregated nonconvex finite-prior notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve()
for parent in [REPO_ROOT, *REPO_ROOT.parents]:
    if (parent / "quantbayes").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from quantbayes.ball_dp import (  # noqa: E402
    ball_rero,
    get_public_curve_history,
    make_finite_identification_prior,
    make_uniform_ball_prior,
)
from quantbayes.ball_dp.api import (  # noqa: E402
    attack_nonconvex_finite_prior_trial,
    calibrate_ball_sgd_noise_multiplier,
    evaluate_release_classifier,
    extract_privacy_epsilon,
    make_trace_metadata_from_release,
)
from quantbayes.ball_dp.attacks.ball_policy import (
    BallTraceMapAttackConfig,
)  # noqa: E402
from quantbayes.ball_dp.attacks.finite_prior_setup import (  # noqa: E402
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
)
from quantbayes.ball_dp.attacks.gradient_based import (  # noqa: E402
    DPSGDTraceRecorder,
    subtract_known_batch_gradients,
)
from quantbayes.ball_dp.theorem import (  # noqa: E402
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    fit_release,
    make_model,
)
from quantbayes.ball_dp.types import ArrayDataset  # noqa: E402

MODEL_ORDER = ["ERM", "Ball-DP", "Std-DP"]
MODEL_COLORS = {"ERM": "tab:gray", "Ball-DP": "tab:blue", "Std-DP": "tab:orange"}
MODE_LABELS = {
    "known_inclusion": r"known / revealed",
    "unknown_inclusion": r"unknown / hidden",
    "rdp": r"RDP",
}
BOUND_MODE_SPECS = [
    ("ball_sgd_hayes", "known_inclusion", MODE_LABELS["known_inclusion"]),
    ("ball_sgd_hidden", "unknown_inclusion", MODE_LABELS["unknown_inclusion"]),
    ("rdp", "rdp", MODE_LABELS["rdp"]),
]


@dataclass(frozen=True)
class ReleaseBundle:
    name: str
    mechanism: str
    release: Any
    noise_multiplier: float
    epsilon_primary: float
    epsilon_ball: float
    epsilon_standard: float
    utility_value: float
    recorder: Any


@dataclass(frozen=True)
class BuiltTrials:
    entries: list[dict[str, Any]]
    support_df: pd.DataFrame
    trial_df: pd.DataFrame
    feasibility_df: pd.DataFrame
    neighbor_counts: np.ndarray


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


def parse_float_tuple(text: str) -> tuple[float, ...]:
    vals = [v.strip() for v in str(text).split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one float")
    return tuple(float(v) for v in vals)


def parse_orders(text: str) -> tuple[int, ...]:
    raw = str(text).strip()
    if ":" in raw:
        pieces = [p.strip() for p in raw.split(":")]
        if len(pieces) not in {2, 3}:
            raise argparse.ArgumentTypeError("orders range must be start:stop[:step]")
        start = int(pieces[0])
        stop = int(pieces[1])
        step = int(pieces[2]) if len(pieces) == 3 else 1
        vals = tuple(range(start, stop, step))
    else:
        vals = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one RDP order")
    if any(v <= 1 for v in vals):
        raise argparse.ArgumentTypeError("all RDP orders must be > 1")
    return vals


def parse_mechanisms(text: str) -> tuple[str, ...]:
    allowed = {"erm", "ball", "standard"}
    vals = tuple(v.strip().lower() for v in str(text).split(",") if v.strip())
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unsupported mechanism(s): {bad}; allowed={sorted(allowed)}"
        )
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one mechanism")
    return vals


def ordered_models(models: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(models))
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(present, key=lambda m: rank.get(m, 999))


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default)
    )


def savefig(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
    radius: float,
) -> np.ndarray:
    counts = np.zeros(len(X_train), dtype=np.int32)
    for label in sorted(np.unique(y_train).tolist()):
        train_idx = np.where(y_train == label)[0]
        X_pub_label = np.asarray(X_public[y_public == label], dtype=np.float32)
        for idx in train_idx:
            d = np.linalg.norm(X_pub_label - X_train[idx], axis=1)
            counts[idx] = int(np.sum(d <= float(radius) + 1e-8))
    return counts


def build_trials(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
) -> BuiltTrials:
    neighbor_counts = same_label_public_neighbor_counts(
        X_train, y_train, X_public, y_public, radius=float(args.radius)
    )
    feasibility_df = pd.DataFrame(
        [
            {
                "radius": float(args.radius),
                "support_size_m": int(args.m),
                "anchors_with_at_least_m_candidates": int(
                    np.sum(neighbor_counts >= int(args.m))
                ),
                "max_candidate_count": int(np.max(neighbor_counts)),
                "median_candidate_count": float(np.median(neighbor_counts)),
                "mean_candidate_count": float(np.mean(neighbor_counts)),
            }
        ]
    )

    public_source = CandidateSource("public", X_public, y_public)
    banks = find_feasible_replacement_banks(
        X_train=X_train,
        y_train=y_train,
        candidate_sources=[public_source],
        radius=float(args.radius),
        min_support_size=int(args.m),
        num_banks=int(args.num_supports),
        seed=int(args.seed),
        anchor_selection=str(args.anchor_selection),
        strict=bool(args.strict_feasible_supports),
    )

    entries: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    support_id_counter = 0

    for bank_id, bank in enumerate(banks):
        for draw_index in range(int(args.support_draws)):
            support = select_support_from_bank(
                bank,
                m=int(args.m),
                selection=str(args.support_selection),
                seed=int(args.seed),
                draw_index=int(draw_index),
            )

            target_positions = target_positions_for_support(
                support,
                policy=str(args.target_policy),
                num_targets=(
                    None
                    if args.target_policy == "all"
                    else int(args.targets_per_support)
                ),
                seed=int(args.seed + draw_index),
            )

            support_id = support_id_counter
            support_id_counter += 1
            selected_targets = set(int(pos) for pos in target_positions)
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
                        "is_selected_target_pool": int(pos in selected_targets),
                        "center_source_id": str(support.center_source_id),
                        "support_hash": str(support.support_hash),
                        "bank_size": int(bank.X.shape[0]),
                    }
                )

            for target_pos in target_positions:
                trial = make_replacement_trial(
                    X_train=X_train,
                    y_train=y_train,
                    support=support,
                    target_support_position=int(target_pos),
                )
                entries.append(
                    {
                        "trial_id": int(len(entries)),
                        "bank_id": int(bank_id),
                        "draw_index": int(draw_index),
                        "support_id": int(support_id),
                        "support": support,
                        "trial": trial,
                        "target_position": int(target_pos),
                        "center_source_id": str(support.center_source_id),
                        "target_source_id": str(trial.target_source_id),
                        "support_hash": str(support.support_hash),
                        "bank_size": int(bank.X.shape[0]),
                    }
                )

    support_df = pd.DataFrame(support_rows)
    trial_df = pd.DataFrame(
        [
            {
                "trial_id": int(row["trial_id"]),
                "support_id": int(row["support_id"]),
                "bank_id": int(row["bank_id"]),
                "draw_index": int(row["draw_index"]),
                "center_source_id": str(row["center_source_id"]),
                "target_position": int(row["target_position"]),
                "target_source_id": str(row["target_source_id"]),
                "support_hash": str(row["support_hash"]),
                "bank_size": int(row["bank_size"]),
                "oblivious_kappa": float(row["support"].oblivious_kappa),
            }
            for row in entries
        ]
    )
    return BuiltTrials(entries, support_df, trial_df, feasibility_df, neighbor_counts)


def make_release_specs(
    args: argparse.Namespace, *, ball_noise: float | None, standard_noise: float | None
) -> list[tuple[str, str, str, float]]:
    specs: list[tuple[str, str, str, float]] = []
    for mech in args.mechanisms:
        if mech == "erm":
            specs.append(("ERM", "erm", "noiseless", 0.0))
        elif mech == "ball":
            assert ball_noise is not None
            specs.append(("Ball-DP", "ball", "ball_rdp", float(ball_noise)))
        elif mech == "standard":
            assert standard_noise is not None
            specs.append(("Std-DP", "standard", "standard_rdp", float(standard_noise)))
        else:
            raise ValueError(f"unknown mechanism {mech!r}")
    return specs


def evaluate_accuracy_safe(release: Any, X: np.ndarray, y: np.ndarray) -> float:
    try:
        out = evaluate_release_classifier(release, X, y, batch_size=1024)
        if "accuracy" in out:
            return float(out["accuracy"])
    except Exception:
        pass
    return float(release.utility_metrics.get("accuracy", np.nan))


def extract_epsilon_safe(release: Any, accounting_view: str) -> float:
    try:
        return float(extract_privacy_epsilon(release, accounting_view=accounting_view))
    except Exception:
        return float("inf")


def calibrate_noise_multiplier(
    args: argparse.Namespace, *, privacy: str, dataset_size: int, lz: float
) -> float:
    out = calibrate_ball_sgd_noise_multiplier(
        dataset_size=int(dataset_size),
        radius=float(args.radius),
        lz=float(lz),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        clip_norm=float(args.clip_norm),
        target_epsilon=float(args.target_epsilon),
        delta=float(args.delta),
        privacy=str(privacy),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        orders=tuple(args.orders),
        lower=float(args.calibration_lower),
        upper=float(args.calibration_upper),
        max_upper=float(args.calibration_max_upper),
        num_bisection_steps=int(args.calibration_bisection_steps),
    )
    return float(out["noise_multiplier"])


def train_recorded_release(
    args: argparse.Namespace,
    *,
    trial: Any,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    X_public: np.ndarray,
    y_public: np.ndarray,
    name: str,
    mechanism: str,
    privacy: str,
    noise_multiplier: float,
    seed: int,
) -> ReleaseBundle:
    recorder = DPSGDTraceRecorder(
        capture_every=int(args.capture_every),
        keep_models=True,
        keep_batch_indices=True,
    )

    model = make_model(
        spec, key=jr.PRNGKey(int(seed)), init_project=True, bounds=bounds
    )

    cfg = TrainConfig(
        radius=float(args.radius),
        privacy=str(privacy),
        delta=float(args.delta),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=float(args.clip_norm),
        noise_multiplier=float(noise_multiplier),
        learning_rate=float(args.learning_rate),
        checkpoint_selection=str(args.checkpoint_selection),
        eval_every=int(args.eval_every),
        eval_batch_size=1024,
        normalize_noisy_sum_by="batch_size",
        seed=int(seed),
    )

    release = fit_release(
        model,
        spec,
        bounds,
        np.asarray(trial.X_full, dtype=np.float32),
        np.asarray(trial.y_full, dtype=np.int32),
        X_eval=np.asarray(X_public, dtype=np.float32),
        y_eval=np.asarray(y_public, dtype=np.int32),
        train_cfg=cfg,
        orders=tuple(args.orders),
        trace_recorder=recorder,
        record_operator_norms=bool(args.record_operator_norms),
        operator_norms_every=max(10, int(args.num_steps) // 5),
    )

    eps_ball = extract_epsilon_safe(release, "ball")
    eps_standard = extract_epsilon_safe(release, "standard")
    eps_primary = (
        eps_ball
        if mechanism == "ball"
        else eps_standard if mechanism == "standard" else float("inf")
    )

    return ReleaseBundle(
        name=name,
        mechanism=mechanism,
        release=release,
        noise_multiplier=float(noise_multiplier),
        epsilon_primary=float(eps_primary),
        epsilon_ball=float(eps_ball),
        epsilon_standard=float(eps_standard),
        utility_value=evaluate_accuracy_safe(release, X_public, y_public),
        recorder=recorder,
    )


def residualized_trace_for_trial(bundle: ReleaseBundle, trial: Any) -> tuple[Any, Any]:
    metadata = make_trace_metadata_from_release(
        bundle.release,
        target_index=int(trial.target_index),
        extra={"mechanism": bundle.mechanism, "model_name": bundle.name},
    )

    reduction = (
        "sum"
        if str(
            bundle.release.training_config.get("normalize_noisy_sum_by", "batch_size")
        ).lower()
        == "none"
        else "mean"
    )

    trace = bundle.recorder.to_trace(
        state=bundle.release.extra.get("model_state", None),
        loss_name="binary_logistic",
        reduction=reduction,
        metadata=metadata,
    )

    residualized = subtract_known_batch_gradients(
        trace,
        ArrayDataset(trial.X_full, trial.y_full, name="attack_train"),
        target_index=int(trial.target_index),
        loss_name="binary_logistic",
        seed=0,
    )
    return trace, residualized


def eta_for_uniform_ball_kappa(radius: float, dimension: int, kappa: float) -> float:
    return float(radius * (float(kappa) ** (1.0 / float(dimension))))


def mean_ci(series: pd.Series, z: float = 1.96) -> tuple[float, float, float]:
    x = pd.Series(series).dropna().astype(float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mu = float(np.mean(x))
    if len(x) == 1:
        return (mu, mu, mu)
    half = float(z * np.std(x, ddof=1) / math.sqrt(len(x)))
    return (mu, mu - half, mu + half)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))
        / denom
    )
    return (center - radius, center + radius)


def run_experiment(
    args: argparse.Namespace,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_public: np.ndarray,
    y_public: np.ndarray,
    built: BuiltTrials,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    lz: float,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    if not built.entries:
        raise RuntimeError(
            "No replacement trials were created. Increase candidate pool, radius, or lower m."
        )

    dataset_size = int(len(built.entries[0]["trial"].X_full))
    ball_noise = None
    standard_noise = None
    if "ball" in args.mechanisms:
        ball_noise = calibrate_noise_multiplier(
            args, privacy="ball_rdp", dataset_size=dataset_size, lz=lz
        )
    if "standard" in args.mechanisms:
        standard_noise = calibrate_noise_multiplier(
            args, privacy="standard_rdp", dataset_size=dataset_size, lz=lz
        )
    release_specs = make_release_specs(
        args, ball_noise=ball_noise, standard_noise=standard_noise
    )

    calibration_df = pd.DataFrame(
        [
            {
                "model": name,
                "mechanism": mechanism,
                "privacy": privacy,
                "noise_multiplier": noise,
            }
            for name, mechanism, privacy, noise in release_specs
        ]
    )
    calibration_df.to_csv(out_dir / "calibration.csv", index=False)

    utility_rows: list[dict[str, Any]] = []
    finite_bound_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    representative_bundles: dict[str, ReleaseBundle] = {}

    attack_eta_grid = tuple(args.attack_eta_grid)
    entries = (
        built.entries[: args.max_trials]
        if args.max_trials is not None
        else built.entries
    )

    for entry in entries:
        trial_id = int(entry["trial_id"])
        trial = entry["trial"]
        support = entry["support"]
        finite_prior = make_finite_identification_prior(
            support.X.reshape(support.m, -1), weights=support.weights
        )

        print(
            f"trial {trial_id + 1:03d}/{len(entries):03d} | "
            f"center={entry['center_source_id']} | target={entry['target_source_id']}",
            flush=True,
        )

        for mech_idx, (name, mechanism, privacy, noise) in enumerate(release_specs):
            train_seed = int(args.seed + 100_000 * trial_id + 1_000 * mech_idx)
            print(f"  training {name} with seed={train_seed}", flush=True)
            bundle = train_recorded_release(
                args,
                trial=trial,
                spec=spec,
                bounds=bounds,
                X_public=X_public,
                y_public=y_public,
                name=name,
                mechanism=mechanism,
                privacy=privacy,
                noise_multiplier=float(noise),
                seed=train_seed,
            )

            if name not in representative_bundles:
                representative_bundles[name] = bundle

            raw_trace, residual_trace = residualized_trace_for_trial(bundle, trial)
            target_inclusions = sum(
                bool(
                    np.any(
                        np.asarray(step.batch_indices, dtype=np.int64)
                        == trial.target_index
                    )
                )
                for step in raw_trace.steps
            )

            utility_rows.append(
                {
                    "trial_id": trial_id,
                    "support_id": int(entry["support_id"]),
                    "bank_id": int(entry["bank_id"]),
                    "model": name,
                    "mechanism": mechanism,
                    "accuracy": bundle.utility_value,
                    "noise_multiplier": bundle.noise_multiplier,
                    "epsilon_primary": bundle.epsilon_primary,
                    "epsilon_ball": bundle.epsilon_ball,
                    "epsilon_standard": bundle.epsilon_standard,
                    "retained_steps": len(raw_trace.steps),
                    "target_inclusions": int(target_inclusions),
                    "delta_ball": getattr(
                        bundle.release.sensitivity, "delta_ball", np.nan
                    ),
                    "delta_standard": getattr(
                        bundle.release.sensitivity, "delta_std", np.nan
                    ),
                    "target_source_id": str(entry["target_source_id"]),
                    "support_hash": str(entry["support_hash"]),
                }
            )

            for hist_row in get_public_curve_history(bundle.release):
                if "public_eval_accuracy" in hist_row:
                    history_rows.append(
                        {
                            "trial_id": trial_id,
                            "model": name,
                            "step": int(hist_row["step"]),
                            "public_eval_accuracy": float(
                                hist_row["public_eval_accuracy"]
                            ),
                        }
                    )

            if mechanism != "erm":
                for raw_mode, mode_key, mode_label in BOUND_MODE_SPECS:
                    try:
                        report = ball_rero(
                            bundle.release,
                            prior=finite_prior,
                            eta_grid=(0.5,),
                            mode=raw_mode,
                        )
                        point = report.points[0]
                        finite_bound_rows.append(
                            {
                                "trial_id": trial_id,
                                "model": name,
                                "mechanism": mechanism,
                                "mode": mode_key,
                                "mode_label": mode_label,
                                "raw_mode": raw_mode,
                                "kappa": float(point.kappa),
                                "gamma_ball": float(point.gamma_ball),
                                "gamma_standard": (
                                    np.nan
                                    if point.gamma_standard is None
                                    else float(point.gamma_standard)
                                ),
                            }
                        )
                    except Exception as exc:
                        finite_bound_rows.append(
                            {
                                "trial_id": trial_id,
                                "model": name,
                                "mechanism": mechanism,
                                "mode": mode_key,
                                "mode_label": mode_label,
                                "raw_mode": raw_mode,
                                "kappa": float(support.oblivious_kappa),
                                "gamma_ball": np.nan,
                                "gamma_standard": np.nan,
                                "error": str(exc),
                            }
                        )

            for attack_mode in ["known_inclusion", "unknown_inclusion"]:
                cfg = BallTraceMapAttackConfig(
                    mode=attack_mode,
                    step_mode=(
                        "present_steps" if attack_mode == "known_inclusion" else "all"
                    ),
                    seed=int(train_seed),
                )
                try:
                    attack = attack_nonconvex_finite_prior_trial(
                        residual_trace,
                        trial,
                        cfg=cfg,
                        known_label=int(trial.support.center_y),
                        eta_grid=attack_eta_grid,
                    )
                    attack_rows.append(
                        {
                            "trial_id": trial_id,
                            "model": name,
                            "mechanism": mechanism,
                            "mode": attack_mode,
                            "mode_label": MODE_LABELS[attack_mode],
                            "status": attack.status,
                            "accuracy": bundle.utility_value,
                            "epsilon_primary": bundle.epsilon_primary,
                            "baseline_kappa": support.oblivious_kappa,
                            "source_exact_id": attack.metrics.get(
                                "source_exact_identification_success", np.nan
                            ),
                            "feature_exact_id": attack.metrics.get(
                                "exact_identification_success", np.nan
                            ),
                            "prior_rank": attack.metrics.get("prior_rank", np.nan),
                            "predicted_prior_index": attack.diagnostics.get(
                                "predicted_prior_index"
                            ),
                            "true_prior_index": attack.diagnostics.get(
                                "true_prior_index"
                            ),
                            "predicted_source_id": attack.diagnostics.get(
                                "predicted_source_id"
                            ),
                            "target_source_id": attack.diagnostics.get(
                                "target_source_id"
                            ),
                            "selected_step_count": attack.diagnostics.get(
                                "selected_step_count"
                            ),
                        }
                    )
                except Exception as exc:
                    attack_rows.append(
                        {
                            "trial_id": trial_id,
                            "model": name,
                            "mechanism": mechanism,
                            "mode": attack_mode,
                            "mode_label": MODE_LABELS[attack_mode],
                            "status": "error",
                            "accuracy": bundle.utility_value,
                            "epsilon_primary": bundle.epsilon_primary,
                            "baseline_kappa": support.oblivious_kappa,
                            "source_exact_id": np.nan,
                            "feature_exact_id": np.nan,
                            "prior_rank": np.nan,
                            "selected_step_count": np.nan,
                            "error": str(exc),
                        }
                    )

    utility_df = pd.DataFrame(utility_rows)
    finite_bound_df = pd.DataFrame(finite_bound_rows)
    attack_df = pd.DataFrame(attack_rows)
    history_df = pd.DataFrame(history_rows)

    curve_df = pd.DataFrame()
    if not args.skip_curves:
        curve_df = compute_representative_curves(
            args, representative_bundles, feature_dim=X_train.shape[1]
        )

    dfs = {
        "calibration": calibration_df,
        "utility": utility_df,
        "finite_bounds": finite_bound_df,
        "attacks": attack_df,
        "history": history_df,
        "curves": curve_df,
    }
    for name, df in dfs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return dfs


def compute_representative_curves(
    args: argparse.Namespace,
    representative_bundles: dict[str, ReleaseBundle],
    *,
    feature_dim: int,
) -> pd.DataFrame:
    kappa_grid = np.geomspace(
        float(args.curve_kappa_min), float(args.curve_kappa_max), int(args.curve_points)
    )
    uniform_prior = make_uniform_ball_prior(
        center=np.zeros(int(feature_dim), dtype=np.float32),
        radius=float(args.radius),
        dimension=int(feature_dim),
    )
    eta_grid = tuple(
        eta_for_uniform_ball_kappa(float(args.radius), int(feature_dim), float(k))
        for k in kappa_grid
    )

    rows: list[dict[str, Any]] = []
    for model_name in ["Ball-DP", "Std-DP"]:
        bundle = representative_bundles.get(model_name)
        if bundle is None:
            continue
        for raw_mode, mode_key, mode_label in BOUND_MODE_SPECS:
            try:
                report = ball_rero(
                    bundle.release,
                    prior=uniform_prior,
                    eta_grid=eta_grid,
                    mode=raw_mode,
                )
                for point in report.points:
                    rows.append(
                        {
                            "model": model_name,
                            "mode": mode_key,
                            "mode_label": mode_label,
                            "kappa": float(point.kappa),
                            "gamma_ball": float(point.gamma_ball),
                            "gamma_standard": (
                                np.nan
                                if point.gamma_standard is None
                                else float(point.gamma_standard)
                            ),
                        }
                    )
            except Exception as exc:
                print(
                    f"Skipping representative curve for {model_name} / {raw_mode}: {exc}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def aggregate_summaries(
    dfs: dict[str, pd.DataFrame], out_dir: Path
) -> dict[str, pd.DataFrame]:
    utility_df = dfs["utility"]
    finite_bound_df = dfs["finite_bounds"]
    attack_df = dfs["attacks"]

    utility_summary_rows: list[dict[str, Any]] = []
    for model, grp in utility_df.groupby("model"):
        mu, lo, hi = mean_ci(grp["accuracy"])
        utility_summary_rows.append(
            {
                "model": model,
                "n_trials": int(len(grp)),
                "mean_accuracy": mu,
                "accuracy_ci_low": lo,
                "accuracy_ci_high": hi,
                "mean_noise_multiplier": float(np.mean(grp["noise_multiplier"])),
                "mean_epsilon_primary": float(np.mean(grp["epsilon_primary"])),
                "mean_epsilon_ball": float(np.mean(grp["epsilon_ball"])),
                "mean_epsilon_standard": float(np.mean(grp["epsilon_standard"])),
                "mean_target_inclusions": float(np.mean(grp["target_inclusions"])),
                "mean_retained_steps": float(np.mean(grp["retained_steps"])),
                "mean_delta_ball": float(np.mean(grp["delta_ball"])),
                "mean_delta_standard": float(np.mean(grp["delta_standard"])),
            }
        )
    utility_summary = pd.DataFrame(utility_summary_rows)
    if not utility_summary.empty:
        rank = {m: i for i, m in enumerate(MODEL_ORDER)}
        utility_summary = utility_summary.sort_values(
            "model", key=lambda s: s.map(rank).fillna(999)
        )

    attack_ok = attack_df[
        attack_df["source_exact_id"].notna()
        & attack_df["status"].astype(str).str.startswith("ok")
    ].copy()
    attack_summary_rows: list[dict[str, Any]] = []
    for (model, mode), grp in attack_ok.groupby(["model", "mode"]):
        vals = np.asarray(grp["source_exact_id"], dtype=np.float64)
        k = int(np.sum(vals))
        n = int(len(vals))
        lo, hi = wilson_interval(k, n)
        attack_summary_rows.append(
            {
                "model": model,
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "n_trials": n,
                "empirical_exact_id": float(k / max(n, 1)),
                "exact_id_ci_low": lo,
                "exact_id_ci_high": hi,
                "mean_accuracy": float(np.mean(grp["accuracy"])),
                "mean_prior_rank": float(np.nanmean(grp["prior_rank"])),
                "mean_selected_step_count": float(
                    np.nanmean(grp["selected_step_count"])
                ),
                "chance_kappa": float(np.nanmean(grp["baseline_kappa"])),
            }
        )
    attack_summary = pd.DataFrame(attack_summary_rows)
    if not attack_summary.empty:
        model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
        mode_rank = {"known_inclusion": 0, "unknown_inclusion": 1}
        attack_summary = attack_summary.sort_values(
            by=["model", "mode"],
            key=lambda s: s.map(model_rank if s.name == "model" else mode_rank).fillna(
                999
            ),
        )

    if finite_bound_df.empty:
        finite_bound_summary = pd.DataFrame()
    else:
        finite_bound_summary = finite_bound_df.groupby(
            ["model", "mode", "mode_label"], as_index=False
        ).agg(
            n_trials=("kappa", "size"),
            mean_kappa=("kappa", "mean"),
            mean_gamma_ball=("gamma_ball", "mean"),
            mean_gamma_standard=("gamma_standard", "mean"),
        )

    attack_with_bounds = attack_summary.merge(
        (
            finite_bound_summary[
                ["model", "mode", "mean_gamma_ball", "mean_gamma_standard"]
            ]
            if not finite_bound_summary.empty
            else pd.DataFrame(
                columns=["model", "mode", "mean_gamma_ball", "mean_gamma_standard"]
            )
        ),
        on=["model", "mode"],
        how="left",
    )

    privacy_utility_rows: list[dict[str, Any]] = []
    for _, row in attack_with_bounds.iterrows():
        if row["model"] == "ERM":
            continue
        if np.isfinite(row.get("mean_gamma_ball", np.nan)):
            privacy_utility_rows.append(
                {
                    "model": row["model"],
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "bound_type": "ball",
                    "bound_value": float(row["mean_gamma_ball"]),
                    "mean_accuracy": float(row["mean_accuracy"]),
                }
            )
        if np.isfinite(row.get("mean_gamma_standard", np.nan)):
            privacy_utility_rows.append(
                {
                    "model": row["model"],
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "bound_type": "standard",
                    "bound_value": float(row["mean_gamma_standard"]),
                    "mean_accuracy": float(row["mean_accuracy"]),
                }
            )
    privacy_utility_df = pd.DataFrame(privacy_utility_rows)

    summaries = {
        "utility_summary": utility_summary,
        "attack_summary": attack_summary,
        "finite_bound_summary": finite_bound_summary,
        "attack_with_bounds": attack_with_bounds,
        "privacy_utility": privacy_utility_df,
    }
    for name, df in summaries.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return summaries


def plot_feasibility(built: BuiltTrials, out_dir: Path) -> None:
    counts = built.neighbor_counts
    m = int(built.feasibility_df.iloc[0]["support_size_m"])
    radius = float(built.feasibility_df.iloc[0]["radius"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.0), constrained_layout=True)
    axes[0].hist(counts, bins=30, alpha=0.85)
    axes[0].axvline(
        m, linestyle="--", linewidth=1.5, color="black", label=rf"threshold $m={m}$"
    )
    axes[0].set_xlabel(r"same-label public neighbors within radius $r$")
    axes[0].set_ylabel(r"count of private anchors")
    axes[0].set_title(rf"Finite-support feasibility at $r={radius:.2f}$")
    axes[0].legend(loc="upper right")

    axes[1].scatter(np.arange(len(counts)), np.sort(counts), s=12, alpha=0.8)
    axes[1].axhline(m, linestyle="--", linewidth=1.5, color="black", label=rf"$m={m}$")
    axes[1].set_xlabel(r"private anchor rank")
    axes[1].set_ylabel(r"candidate count")
    axes[1].set_title(r"Sorted same-label candidate counts")
    axes[1].legend(loc="upper left")
    savefig(fig, out_dir / "fig_feasibility")


def plot_all(
    dfs: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame], out_dir: Path
) -> None:
    utility_summary = summaries["utility_summary"]
    attack_with_bounds = summaries["attack_with_bounds"]
    curve_df = dfs.get("curves", pd.DataFrame())
    history_df = dfs.get("history", pd.DataFrame())

    if utility_summary.empty:
        return

    present_models = ordered_models(utility_summary["model"])
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.2), constrained_layout=True)

    ax = axes[0, 0]
    sub = utility_summary.set_index("model").loc[present_models]
    x = np.arange(len(sub))
    mean_acc = sub["mean_accuracy"].to_numpy(dtype=float)
    err_low = mean_acc - sub["accuracy_ci_low"].to_numpy(dtype=float)
    err_high = sub["accuracy_ci_high"].to_numpy(dtype=float) - mean_acc
    colors = [MODEL_COLORS.get(m, "tab:gray") for m in sub.index]
    ax.bar(
        x,
        mean_acc,
        yerr=np.vstack([err_low, err_high]),
        capsize=4,
        alpha=0.88,
        color=colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"public accuracy")
    ax.set_title(r"Aggregate utility across replacement trials")
    for i, row in enumerate(sub.itertuples()):
        eps_text = (
            r"$\varepsilon=\infty$"
            if not np.isfinite(row.mean_epsilon_primary)
            else rf"$\bar{{\varepsilon}}={row.mean_epsilon_primary:.2f}$"
        )
        noise_text = rf"$\bar{{\sigma}}={row.mean_noise_multiplier:.2f}$"
        ax.text(
            i,
            min(1.02, row.mean_accuracy + 0.045),
            eps_text + "\n" + noise_text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[0, 1]
    if not curve_df.empty:
        kappa_vals = np.sort(curve_df["kappa"].unique())
        ax.plot(
            kappa_vals,
            kappa_vals,
            linestyle="--",
            linewidth=1.3,
            color="black",
            label=r"baseline $\kappa$",
        )
        mode_styles = {"known_inclusion": "-", "unknown_inclusion": "--", "rdp": ":"}
        for model_name in ["Ball-DP", "Std-DP"]:
            for mode_key in ["known_inclusion", "unknown_inclusion", "rdp"]:
                subc = curve_df[
                    (curve_df["model"] == model_name) & (curve_df["mode"] == mode_key)
                ].sort_values("kappa")
                if subc.empty:
                    continue
                ax.plot(
                    subc["kappa"],
                    subc["gamma_ball"],
                    color=MODEL_COLORS[model_name],
                    linestyle=mode_styles[mode_key],
                    linewidth=1.9,
                    label=rf"{model_name}, {MODE_LABELS[mode_key]}",
                )
        ax.set_xscale("log")
        ax.set_xlabel(r"prior mass $\kappa$")
        ax.set_ylabel(r"bound value $\Gamma(\kappa)$")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(r"Representative uniform-ball ReRo curves")
        ax.legend(loc="lower right", ncol=1)
    else:
        ax.text(0.5, 0.5, "curves skipped", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 0]
    plot_rows = []
    for model_name in present_models:
        for mode_key in ["known_inclusion", "unknown_inclusion"]:
            suba = attack_with_bounds[
                (attack_with_bounds["model"] == model_name)
                & (attack_with_bounds["mode"] == mode_key)
            ]
            if not suba.empty:
                plot_rows.append(suba.iloc[0])
    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        x = np.arange(len(plot_df))
        bar_heights = plot_df["empirical_exact_id"].to_numpy(dtype=float)
        err_low = bar_heights - plot_df["exact_id_ci_low"].to_numpy(dtype=float)
        err_high = plot_df["exact_id_ci_high"].to_numpy(dtype=float) - bar_heights
        bar_colors = [MODEL_COLORS.get(m, "tab:gray") for m in plot_df["model"]]
        ax.bar(
            x,
            bar_heights,
            yerr=np.vstack([err_low, err_high]),
            capsize=4,
            alpha=0.86,
            color=bar_colors,
        )
        chance = float(np.nanmean(plot_df["chance_kappa"]))
        if np.isfinite(chance):
            ax.axhline(
                chance,
                linestyle="--",
                linewidth=1.4,
                color="black",
                label=rf"chance baseline $\kappa=1/m={chance:.3f}$",
            )
        mask_ball = np.isfinite(plot_df["mean_gamma_ball"].to_numpy(dtype=float))
        mask_std = np.isfinite(plot_df["mean_gamma_standard"].to_numpy(dtype=float))
        ax.scatter(
            x[mask_ball],
            plot_df.loc[mask_ball, "mean_gamma_ball"],
            s=70,
            marker="o",
            facecolors="none",
            linewidths=1.8,
            edgecolors="tab:blue",
            label=r"mean Ball bound $\overline{\Gamma}_{\mathrm{ball}}$",
            zorder=4,
        )
        ax.scatter(
            x[mask_std],
            plot_df.loc[mask_std, "mean_gamma_standard"],
            s=70,
            marker="s",
            facecolors="none",
            linewidths=1.8,
            edgecolors="tab:red",
            label=r"mean standard bound $\overline{\Gamma}_{\mathrm{std}}$",
            zorder=4,
        )
        labels = [
            f"{m}\n{MODE_LABELS[mode]}"
            for m, mode in zip(plot_df["model"], plot_df["mode"])
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r"empirical exact-ID success")
        ax.set_title(r"Finite-prior transcript attacks vs. bounds")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no successful attacks", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    if not attack_with_bounds.empty:
        for mode_key, linestyle in [
            ("known_inclusion", "-"),
            ("unknown_inclusion", "--"),
        ]:
            subp = attack_with_bounds[
                (attack_with_bounds["mode"] == mode_key)
                & (attack_with_bounds["model"].isin(["Ball-DP", "Std-DP"]))
            ]
            for row in subp.itertuples():
                if np.isfinite(row.mean_gamma_ball) and np.isfinite(
                    row.mean_gamma_standard
                ):
                    ax.plot(
                        [row.mean_gamma_ball, row.mean_gamma_standard],
                        [row.mean_accuracy, row.mean_accuracy],
                        linestyle=linestyle,
                        linewidth=1.8,
                        color=MODEL_COLORS[row.model],
                        alpha=0.75,
                    )
                    ax.scatter(
                        [row.mean_gamma_ball],
                        [row.mean_accuracy],
                        marker="o",
                        s=70,
                        color=MODEL_COLORS[row.model],
                        zorder=4,
                    )
                    ax.scatter(
                        [row.mean_gamma_standard],
                        [row.mean_accuracy],
                        marker="s",
                        s=70,
                        facecolors="white",
                        edgecolors=MODEL_COLORS[row.model],
                        linewidths=1.8,
                        zorder=4,
                    )
                    ax.text(
                        row.mean_gamma_standard + 0.01,
                        row.mean_accuracy + 0.005,
                        f"{row.model}\n{MODE_LABELS[mode_key]}",
                        fontsize=8.4,
                        va="bottom",
                    )
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r"finite-prior upper bound $\Gamma$")
        ax.set_ylabel(r"mean public accuracy")
        ax.set_title(r"Utility vs. Ball/standard finite-prior bounds")
        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="Ball-DP"),
            Line2D([0], [0], color="tab:orange", lw=2, label="Std-DP"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                label=r"Ball bound",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                markerfacecolor="white",
                markeredgecolor="black",
                color="black",
                linestyle="None",
                label=r"Standard bound",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                lw=1.8,
                label=r"known / revealed",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                lw=1.8,
                label=r"unknown / hidden",
            ),
        ]
        ax.legend(handles=legend_handles, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no bound summary", ha="center", va="center")
        ax.set_axis_off()

    savefig(fig, out_dir / "fig_summary")

    if not history_df.empty:
        rows: list[dict[str, Any]] = []
        for (model, step), grp in history_df.groupby(["model", "step"]):
            mu, lo, hi = mean_ci(grp["public_eval_accuracy"])
            rows.append(
                {
                    "model": model,
                    "step": int(step),
                    "mean": mu,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        hist_summary = pd.DataFrame(rows)
        hist_summary.to_csv(out_dir / "history_summary.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
        for model_name in ordered_models(hist_summary["model"]):
            subh = hist_summary[hist_summary["model"] == model_name].sort_values("step")
            ax.plot(
                subh["step"],
                subh["mean"],
                linewidth=2.1,
                color=MODEL_COLORS.get(model_name, "tab:gray"),
                label=model_name,
            )
            ax.fill_between(
                subh["step"],
                subh["ci_low"],
                subh["ci_high"],
                alpha=0.16,
                color=MODEL_COLORS.get(model_name, "tab:gray"),
            )
        ax.set_xlabel(r"training step $t$")
        ax.set_ylabel(r"public accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(r"Aggregate public utility during training")
        ax.legend(loc="lower right")
        savefig(fig, out_dir / "fig_training_curves")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run thesis-scale nonconvex Ball-DP finite-prior transcript experiments."
    )
    p.add_argument("--out-dir", type=Path, default=Path("runs/nonconvex_thesis"))
    p.add_argument("--seed", type=int, default=0)

    # Synthetic data.
    p.add_argument("--n-samples", type=int, default=1600)
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--test-size", type=float, default=0.40)
    p.add_argument("--class-sep", type=float, default=2.6)
    p.add_argument("--label-noise", type=float, default=0.0)
    p.add_argument("--embedding-bound", type=float, default=5.0)

    # Finite-prior setup.
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--m", type=int, default=6)
    p.add_argument(
        "--support-selection",
        choices=["random", "nearest", "farthest"],
        default="farthest",
    )
    p.add_argument(
        "--anchor-selection",
        choices=["random", "rare_class", "large_bank"],
        default="large_bank",
    )
    p.add_argument("--num-supports", type=int, default=8)
    p.add_argument("--support-draws", type=int, default=2)
    p.add_argument("--target-policy", choices=["all", "sample"], default="all")
    p.add_argument("--targets-per-support", type=int, default=2)
    p.add_argument(
        "--strict-feasible-supports",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional debug cap on replacement trials.",
    )

    # Model/training.
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--A", dest="A", type=float, default=5.0)
    p.add_argument("--Lambda", dest="Lambda", type=float, default=5.0)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--clip-norm", type=float, default=50.0)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--target-epsilon", type=float, default=12.0)
    p.add_argument("--capture-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--checkpoint-selection", type=str, default="best_public_eval_accuracy"
    )
    p.add_argument("--orders", type=parse_orders, default=parse_orders("2:13"))
    p.add_argument(
        "--mechanisms",
        type=parse_mechanisms,
        default=parse_mechanisms("erm,ball,standard"),
    )
    p.add_argument(
        "--record-operator-norms", action=argparse.BooleanOptionalAction, default=True
    )

    # Calibration.
    p.add_argument("--calibration-lower", type=float, default=1e-3)
    p.add_argument("--calibration-upper", type=float, default=0.25)
    p.add_argument("--calibration-max-upper", type=float, default=128.0)
    p.add_argument("--calibration-bisection-steps", type=int, default=12)

    # Attacks/curves/plots.
    p.add_argument(
        "--attack-eta-grid",
        type=parse_float_tuple,
        default=parse_float_tuple("0.25,0.5,1.0"),
    )
    p.add_argument(
        "--skip-curves", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--curve-points", type=int, default=80)
    p.add_argument("--curve-kappa-min", type=float, default=1e-4)
    p.add_argument("--curve-kappa-max", type=float, default=0.95)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build data/supports and write config/feasibility files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    out_dir = ensure_dir(args.out_dir)
    (out_dir / "figures").mkdir(exist_ok=True)

    write_json(out_dir / "config.json", vars(args))

    X_train, y_train, X_public, y_public = make_synthetic_embeddings(args)
    feature_dim = int(X_train.shape[1])
    data_summary = pd.DataFrame(
        {
            "split": ["private_train", "public_eval"],
            "n": [len(X_train), len(X_public)],
            "feature_dim": [feature_dim, feature_dim],
            "max_l2_norm": [
                float(np.linalg.norm(X_train, axis=1).max()),
                float(np.linalg.norm(X_public, axis=1).max()),
            ],
            "class_0": [int(np.sum(y_train == 0)), int(np.sum(y_public == 0))],
            "class_1": [int(np.sum(y_train == 1)), int(np.sum(y_public == 1))],
        }
    )
    data_summary.to_csv(out_dir / "data_summary.csv", index=False)

    built = build_trials(args, X_train, y_train, X_public, y_public)
    built.feasibility_df.to_csv(out_dir / "support_feasibility.csv", index=False)
    built.support_df.to_csv(out_dir / "supports.csv", index=False)
    built.trial_df.to_csv(out_dir / "trials.csv", index=False)
    if not args.no_plots:
        plot_feasibility(built, out_dir / "figures")

    spec = TheoremModelSpec(
        d_in=feature_dim,
        hidden_dim=int(args.hidden_dim),
        task="binary",
        parameterization="dense",
        constraint="op",
    )
    bounds = TheoremBounds(
        B=float(args.embedding_bound), A=float(args.A), Lambda=float(args.Lambda)
    )
    lz = float(certified_lz(spec, bounds))
    theorem_summary = pd.DataFrame(
        [
            {
                "feature_dim": feature_dim,
                "hidden_dim": int(args.hidden_dim),
                "B": float(args.embedding_bound),
                "A": float(args.A),
                "Lambda": float(args.Lambda),
                "certified_lz": lz,
                "radius": float(args.radius),
                "delta_ball": lz * float(args.radius),
                "delta_standard": 2.0 * float(args.clip_norm),
                "ball_to_standard_ratio": (lz * float(args.radius))
                / (2.0 * float(args.clip_norm)),
            }
        ]
    )
    theorem_summary.to_csv(out_dir / "theorem_summary.csv", index=False)

    print("Data summary:")
    print(data_summary.to_string(index=False))
    print("\nSupport feasibility:")
    print(built.feasibility_df.to_string(index=False))
    print("\nTheorem constants:")
    print(theorem_summary.to_string(index=False))
    print(f"\nReplacement trials: {len(built.entries)}")

    if args.dry_run:
        print("Dry run requested; stopping after support construction.")
        return

    dfs = run_experiment(
        args,
        X_train=X_train,
        y_train=y_train,
        X_public=X_public,
        y_public=y_public,
        built=built,
        spec=spec,
        bounds=bounds,
        lz=lz,
        out_dir=out_dir,
    )
    summaries = aggregate_summaries(dfs, out_dir)
    if not args.no_plots:
        plot_all(dfs, summaries, out_dir / "figures")

    print("\nUtility summary:")
    print(summaries["utility_summary"].to_string(index=False))
    print("\nAttack summary:")
    print(summaries["attack_summary"].to_string(index=False))
    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
