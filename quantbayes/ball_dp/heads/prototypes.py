# quantbayes/ball_dp/heads/prototypes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def fit_ridge_prototypes(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ridge prototypes minimizing:

      J(mu) = sum_i ||z_i - mu_{y_i}||^2 + (lam/2) * sum_c ||mu_c||^2

    Solution per class c:
      mu_c = 2 * sum_{i:y=c} z_i / (2*n_c + lam)

    Returns:
      mus: (K,d)
      counts: (K,)
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    y = y.astype(np.int64)
    K = int(num_classes)
    d = int(Z.shape[1])

    mus = np.zeros((K, d), dtype=np.float32)
    counts = np.zeros((K,), dtype=np.int64)

    for c in range(K):
        idx = np.where(y == c)[0]
        counts[c] = idx.size
        if idx.size == 0:
            continue
        s = Z[idx].sum(axis=0)
        mus[c] = (2.0 * s) / (2.0 * float(idx.size) + float(lam))

    return mus, counts


def predict_nearest_prototype(Z: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """
    Nearest prototype under squared Euclidean distance.
    """
    # ||z-mu||^2 = ||z||^2 + ||mu||^2 - 2 z·mu
    z2 = (Z * Z).sum(axis=1, keepdims=True)  # (N,1)
    m2 = (mus * mus).sum(axis=1, keepdims=True).T  # (1,K)
    D = z2 + m2 - 2.0 * (Z @ mus.T)  # (N,K)
    return D.argmin(axis=1)


def prototypes_sensitivity_l2(*, r: float, n_min: int, lam: float) -> float:
    """
    Exact L2 sensitivity of the closed-form ridge prototypes under ball adjacency.

    If a single embedding in class c changes by at most ||z-z'||<=r, then:
      ||mu_c - mu_c'|| <= 2r / (2 n_c + lam)

    Worst case is the smallest class size n_min:
      Δ2 = 2r / (2 n_min + lam)
    """
    if n_min <= 0:
        raise ValueError("n_min must be >= 1")
    if r < 0:
        raise ValueError("r must be >= 0")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    return (2.0 * float(r)) / (2.0 * float(n_min) + float(lam))
