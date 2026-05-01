#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    make_replacement_trial,
    select_support_from_bank,
    target_positions_for_support,
)


def make_data(seed: int = 0):
    X, y = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=12,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.5,
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

    # Bound the synthetic feature norm.
    B = 5.0
    max_norm = max(
        float(np.linalg.norm(X_train, axis=1).max()),
        float(np.linalg.norm(X_public, axis=1).max()),
        1e-12,
    )
    scale = 0.95 * B / max_norm
    return scale * X_train, y_train, scale * X_public, y_public


def main() -> None:
    seed = 0
    m = 8
    radius = 1.75

    X_train, y_train, X_public, y_public = make_data(seed)

    public = CandidateSource("public", X_public, y_public)

    banks = find_feasible_replacement_banks(
        X_train=X_train,
        y_train=y_train,
        candidate_sources=[public],
        radius=radius,
        min_support_size=m,
        num_banks=1,
        seed=seed,
        anchor_selection="large_bank",
        strict=True,
    )

    bank = banks[0]

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

    print("=== Canonical finite-prior setup ===")
    print("anchor index:", bank.center_index)
    print("anchor label:", bank.center_y)
    print("support size:", support.m)
    print("support hash:", support.support_hash)
    print("target support position:", trial.target_support_position)
    print("target source id:", trial.target_source_id)
    print("oblivious baseline kappa:", support.oblivious_kappa)
    print("D^- shape:", trial.D_minus_X.shape)
    print("D^- ∪ {target} shape:", trial.X_full.shape)
    print("target index in trained dataset:", trial.target_index)

    assert trial.target_index == len(trial.X_full) - 1
    assert np.allclose(
        trial.X_full[trial.target_index],
        support.X[trial.target_support_position],
    )
    assert trial.y_full[trial.target_index] == support.y[trial.target_support_position]
    assert np.all(support.distances_to_center <= radius + 1e-5)

    print("\nAll setup invariants passed.")


if __name__ == "__main__":
    main()
