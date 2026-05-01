#!/usr/bin/env python3
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from typing import Any

import jax.random as jr
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantbayes.ball_dp.api import (
    attack_nonconvex_finite_prior_trial,
    calibrate_ball_sgd_noise_multiplier,
    evaluate_release_classifier,
    extract_privacy_epsilon,
    make_trace_metadata_from_release,
)
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
)
from quantbayes.ball_dp.attacks.gradient_based import (
    DPSGDTraceRecorder,
    subtract_known_batch_gradients,
)
from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    fit_release,
    make_model,
)
from quantbayes.ball_dp.types import ArrayDataset


DEFAULT_ORDERS = tuple(range(2, 13))


@dataclass
class ReleaseBundle:
    name: str
    mechanism: str
    release: Any
    noise_multiplier: float
    epsilon_primary: float
    utility_value: float
    recorder: Any


def make_data(seed: int = 0):
    X, y = make_classification(
        n_samples=700,
        n_features=16,
        n_informative=16,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.8,
        flip_y=0.0,
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_public, y_train, y_public = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_public = scaler.transform(X_public).astype(np.float32)

    B = 5.0
    max_norm = max(
        float(np.linalg.norm(X_train, axis=1).max()),
        float(np.linalg.norm(X_public, axis=1).max()),
        1e-12,
    )
    scale = 0.95 * B / max_norm
    return scale * X_train, y_train, scale * X_public, y_public, B


def evaluate_accuracy_safe(release: Any, X: np.ndarray, y: np.ndarray) -> float:
    try:
        out = evaluate_release_classifier(release, X, y, batch_size=1024)
        return float(out.get("accuracy", np.nan))
    except Exception:
        return float("nan")


def extract_epsilon_safe(release: Any, accounting_view: str) -> float:
    try:
        return float(extract_privacy_epsilon(release, accounting_view=accounting_view))
    except Exception:
        return float("inf")


def calibrate_noise(
    *,
    dataset_size: int,
    radius: float,
    lz: float,
    num_steps: int,
    batch_size: int,
    clip_norm: float,
    target_epsilon: float,
    delta: float,
    privacy: str,
) -> float:
    out = calibrate_ball_sgd_noise_multiplier(
        dataset_size=int(dataset_size),
        radius=float(radius),
        lz=float(lz),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        clip_norm=float(clip_norm),
        target_epsilon=float(target_epsilon),
        delta=float(delta),
        privacy=str(privacy),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        orders=DEFAULT_ORDERS,
        lower=1e-3,
        upper=0.25,
        max_upper=128.0,
        num_bisection_steps=12,
    )
    return float(out["noise_multiplier"])


def train_one_release(
    *,
    name: str,
    mechanism: str,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    radius: float,
    privacy: str,
    noise_multiplier: float,
    delta: float,
    num_steps: int,
    batch_size: int,
    clip_norm: float,
    learning_rate: float,
    seed: int,
    capture_every: int,
) -> ReleaseBundle:
    recorder = DPSGDTraceRecorder(
        capture_every=int(capture_every),
        keep_models=True,
        keep_batch_indices=True,
    )

    model = make_model(
        spec,
        key=jr.PRNGKey(int(seed)),
        init_project=True,
        bounds=bounds,
    )

    cfg = TrainConfig(
        radius=float(radius),
        privacy=str(privacy),
        delta=float(delta),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        learning_rate=float(learning_rate),
        checkpoint_selection="last",
        eval_every=max(10, int(num_steps) // 5),
        eval_batch_size=1024,
        normalize_noisy_sum_by="batch_size",
        seed=int(seed),
    )

    release = fit_release(
        model,
        spec,
        bounds,
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int32),
        X_eval=np.asarray(X_eval, dtype=np.float32),
        y_eval=np.asarray(y_eval, dtype=np.int32),
        train_cfg=cfg,
        orders=DEFAULT_ORDERS,
        trace_recorder=recorder,
        record_operator_norms=False,
    )

    eps_ball = extract_epsilon_safe(release, "ball")
    eps_std = extract_epsilon_safe(release, "standard")
    eps_primary = (
        eps_ball
        if mechanism == "ball"
        else eps_std if mechanism == "standard" else float("inf")
    )

    return ReleaseBundle(
        name=name,
        mechanism=mechanism,
        release=release,
        noise_multiplier=float(noise_multiplier),
        epsilon_primary=float(eps_primary),
        utility_value=evaluate_accuracy_safe(release, X_eval, y_eval),
        recorder=recorder,
    )


def residualized_trace_for_trial(bundle: ReleaseBundle, trial) -> Any:
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

    return subtract_known_batch_gradients(
        trace,
        ArrayDataset(trial.X_full, trial.y_full, name="attack_train"),
        target_index=int(trial.target_index),
        loss_name="binary_logistic",
        seed=0,
    )


def main() -> None:
    seed = 0
    radius = 1.75
    m = 6
    num_steps = 200
    batch_size = 128
    clip_norm = 50.0
    learning_rate = 0.01
    delta = 1e-5
    target_epsilon = 4.0

    X_train, y_train, X_public, y_public, B = make_data(seed)

    spec = TheoremModelSpec(
        d_in=int(X_train.shape[1]),
        hidden_dim=80,
        task="binary",
        parameterization="dense",
        constraint="op",
    )
    bounds = TheoremBounds(B=float(B), A=10.0, Lambda=10.0)
    lz = float(certified_lz(spec, bounds))

    public = CandidateSource("public", X_public, y_public)
    bank = find_feasible_replacement_banks(
        X_train=X_train,
        y_train=y_train,
        candidate_sources=[public],
        radius=radius,
        min_support_size=m,
        num_banks=1,
        seed=seed,
        anchor_selection="large_bank",
        strict=True,
    )[0]

    support = select_support_from_bank(
        bank,
        m=m,
        selection="farthest",
        seed=seed,
        draw_index=0,
    )

    target_pos = target_positions_for_support(
        support,
        policy="sample",
        num_targets=1,
        seed=seed,
    )[0]

    trial = make_replacement_trial(
        X_train=X_train,
        y_train=y_train,
        support=support,
        target_support_position=target_pos,
    )

    ball_noise = calibrate_noise(
        dataset_size=len(trial.X_full),
        radius=radius,
        lz=lz,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        target_epsilon=target_epsilon,
        delta=delta,
        privacy="ball_rdp",
    )

    std_noise = calibrate_noise(
        dataset_size=len(trial.X_full),
        radius=radius,
        lz=lz,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        target_epsilon=target_epsilon,
        delta=delta,
        privacy="standard_rdp",
    )

    release_specs = [
        ("ERM", "erm", "noiseless", 0.0),
        ("Ball-DP", "ball", "ball_rdp", ball_noise),
        ("Std-DP", "standard", "standard_rdp", std_noise),
    ]

    rows = []

    for i, (name, mechanism, privacy, noise) in enumerate(release_specs):
        bundle = train_one_release(
            name=name,
            mechanism=mechanism,
            spec=spec,
            bounds=bounds,
            X_train=trial.X_full,
            y_train=trial.y_full,
            X_eval=X_public,
            y_eval=y_public,
            radius=radius,
            privacy=privacy,
            noise_multiplier=noise,
            delta=delta,
            num_steps=num_steps,
            batch_size=batch_size,
            clip_norm=clip_norm,
            learning_rate=learning_rate,
            seed=seed + 1000 * i,
            capture_every=5,
        )

        trace = residualized_trace_for_trial(bundle, trial)

        for mode in ("known_inclusion", "unknown_inclusion"):
            cfg = BallTraceMapAttackConfig(
                mode=mode,
                step_mode=("present_steps" if mode == "known_inclusion" else "all"),
                seed=seed,
            )

            try:
                attack = attack_nonconvex_finite_prior_trial(
                    trace,
                    trial,
                    cfg=cfg,
                    known_label=int(trial.support.center_y),
                    eta_grid=(0.25, 0.50, 1.00),
                )

                rows.append(
                    {
                        "model": name,
                        "mode": mode,
                        "accuracy": bundle.utility_value,
                        "epsilon": bundle.epsilon_primary,
                        "noise_multiplier": bundle.noise_multiplier,
                        "baseline_kappa": trial.support.oblivious_kappa,
                        "exact_id": attack.metrics.get(
                            "source_exact_identification_success", np.nan
                        ),
                        "predicted_source_id": attack.diagnostics.get(
                            "predicted_source_id"
                        ),
                        "target_source_id": attack.diagnostics.get("target_source_id"),
                        "target_support_position": attack.diagnostics.get(
                            "target_support_position"
                        ),
                        "predicted_prior_index": attack.diagnostics.get(
                            "predicted_prior_index"
                        ),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": name,
                        "mode": mode,
                        "accuracy": bundle.utility_value,
                        "epsilon": bundle.epsilon_primary,
                        "noise_multiplier": bundle.noise_multiplier,
                        "baseline_kappa": trial.support.oblivious_kappa,
                        "exact_id": np.nan,
                        "error": str(exc),
                    }
                )

    print("\n=== Canonical finite-prior trial ===")
    print("support hash:", trial.support.support_hash)
    print("center:", trial.support.center_source_id)
    print("target:", trial.target_source_id)
    print("baseline:", trial.support.oblivious_kappa)

    print("\n=== Attack results ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
