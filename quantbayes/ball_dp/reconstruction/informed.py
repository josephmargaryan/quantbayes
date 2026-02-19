# quantbayes/ball_dp/reconstruction/informed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .priors import PoolBallPrior
from .types import Candidate, DatasetMinus


@dataclass(frozen=True)
class InformedDataset:
    """
    D = D_minus ∪ {target} with the convention that the target is appended last.
    """

    X_minus: np.ndarray
    y_minus: np.ndarray
    x_target: np.ndarray
    y_target: int

    @property
    def d_minus(self) -> DatasetMinus:
        return (self.X_minus, self.y_minus)

    def full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.concatenate([self.X_minus, self.x_target.reshape(1, -1)], axis=0)
        y = np.concatenate(
            [self.y_minus, np.asarray([int(self.y_target)], dtype=np.int64)], axis=0
        )
        return X, y


def make_informed_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    fixed_indices: np.ndarray,
    target_index: int,
) -> InformedDataset:
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    fixed_indices = np.asarray(fixed_indices, dtype=np.int64).reshape(-1)

    if int(target_index) in set(map(int, fixed_indices.tolist())):
        raise ValueError("target_index must not be in fixed_indices")

    X_minus = X[fixed_indices]
    y_minus = y[fixed_indices]
    x_target = X[int(target_index)]
    y_target = int(y[int(target_index)])

    return InformedDataset(
        X_minus=np.asarray(X_minus, dtype=np.float64),
        y_minus=np.asarray(y_minus, dtype=np.int64),
        x_target=np.asarray(x_target, dtype=np.float64).reshape(-1),
        y_target=int(y_target),
    )


def sample_candidate_set_from_ball_prior(
    *,
    prior: PoolBallPrior,
    center: np.ndarray,
    target_label: Optional[int],
    m: int,
    rng: np.random.Generator,
    include_target: bool = True,
) -> List[Candidate]:
    """
    Sample m candidates from PoolBallPrior; optionally append the true target as a candidate.
    """
    m = int(m)
    if m <= 0:
        raise ValueError("m must be >= 1")

    # prior already supports label_fixed; use it outside if desired
    candidates = list(
        prior.sample(center=center, m=(m - 1 if include_target else m), rng=rng)
    )

    if include_target:
        if target_label is None:
            raise ValueError("include_target=True requires target_label")
        candidates.append(
            Candidate(
                record=np.asarray(center).reshape(-1),
                label=int(target_label),
                meta={"is_target": True},
            )
        )

    rng.shuffle(candidates)
    return candidates


def nearest_neighbor_oracle(
    *,
    center: np.ndarray,
    pool_X: np.ndarray,
    pool_y: Optional[np.ndarray] = None,
    label_fixed: Optional[int] = None,
) -> Candidate:
    """
    Nearest-neighbor baseline (oracle): choose the closest pool point under L2.

    If label_fixed is provided and pool_y is not None, restrict to that label.
    """
    c = np.asarray(center).reshape(1, -1)
    P = np.asarray(pool_X)
    if pool_y is not None and label_fixed is not None:
        y = np.asarray(pool_y).reshape(-1)
        idx = np.where(y == int(label_fixed))[0]
        if idx.size == 0:
            raise ValueError("No pool points with the requested label.")
        P2 = P[idx]
        d = np.linalg.norm(P2 - c, axis=1)
        j = idx[int(np.argmin(d))]
    else:
        d = np.linalg.norm(P - c, axis=1)
        j = int(np.argmin(d))

    lab = None if pool_y is None else int(np.asarray(pool_y).reshape(-1)[j])
    return Candidate(
        record=np.asarray(pool_X)[j].reshape(-1),
        label=lab,
        meta={"pool_index": int(j), "nn_oracle": True},
    )
