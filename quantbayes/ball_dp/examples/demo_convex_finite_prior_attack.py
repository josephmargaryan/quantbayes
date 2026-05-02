#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / "quantbayes").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantbayes.ball_dp import fit_convex
from quantbayes.ball_dp.api import attack_convex_finite_prior_trial
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
)


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


def main() -> None:
    seed = 0
    radius = 1.75
    standard_radius = 2.0 * 5.0
    m = 6
    epsilon = 4.0
    delta = 1e-6
    lam = 1e-2
    model_family = "ridge_prototype"

    X_train, y_train, X_public, y_public, B = make_data(seed)

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

    rows = []

    release_specs = [
        ("Ball-DP noisy ERM", radius),
        ("Standard-DP noisy ERM", standard_radius),
    ]

    for name, release_radius in release_specs:
        release = fit_convex(
            trial.X_full,
            trial.y_full,
            model_family=model_family,
            privacy="ball_dp",
            radius=float(release_radius),
            lam=float(lam),
            epsilon=float(epsilon),
            delta=float(delta),
            embedding_bound=float(B),
            standard_radius=float(standard_radius),
            ridge_sensitivity_mode="global",
            num_classes=2,
            orders=tuple(float(v) for v in range(2, 65)),
            max_iter=100,
            solver="lbfgs_fullbatch",
            seed=int(seed),
        )

        attack = attack_convex_finite_prior_trial(
            release,
            trial,
            known_label=int(trial.support.center_y),
            eta_grid=(0.25, 0.50, 1.00),
        )

        rows.append(
            {
                "model": name,
                "support_hash": trial.support.support_hash,
                "baseline_kappa": trial.support.oblivious_kappa,
                "target_source_id": attack.diagnostics.get("target_source_id"),
                "predicted_source_id": attack.diagnostics.get("predicted_source_id"),
                "target_support_position": attack.diagnostics.get(
                    "target_support_position"
                ),
                "predicted_prior_index": attack.diagnostics.get(
                    "predicted_prior_index"
                ),
                "source_exact_id": attack.metrics.get(
                    "source_exact_identification_success", np.nan
                ),
                "posterior_top1": attack.metrics.get(
                    "posterior_top1_probability", np.nan
                ),
                "posterior_true": attack.metrics.get(
                    "posterior_true_probability", np.nan
                ),
            }
        )

    print("\n=== Canonical finite-prior trial ===")
    print("center:", trial.support.center_source_id)
    print("support hash:", trial.support.support_hash)
    print("target:", trial.target_source_id)
    print("baseline:", trial.support.oblivious_kappa)

    print("\n=== Convex finite-prior attack results ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
