from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


def within_class_nn_distances(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    per_class: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """
    Approximate within-class nearest-neighbor distances by subsampling per class.
    Returns nn_dists of shape (sum_c min(per_class, n_c),).
    """
    rng = np.random.default_rng(seed)
    y = y.astype(np.int64)
    all_nn: List[np.ndarray] = []

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if idx.size < 2:
            continue
        take = int(min(per_class, idx.size))
        sub = rng.choice(idx, size=take, replace=False)
        A = Z[sub].astype(np.float64, copy=False)

        AA = (A * A).sum(axis=1, keepdims=True)
        D2 = AA + AA.T - 2.0 * (A @ A.T)
        np.fill_diagonal(D2, np.inf)
        nn = np.sqrt(np.min(np.maximum(D2, 0.0), axis=1))
        all_nn.append(nn.astype(np.float32))

    if not all_nn:
        raise RuntimeError("Could not compute within-class NN distances.")
    return np.concatenate(all_nn, axis=0)


def radii_from_percentiles(
    nn_dists: np.ndarray, percentiles: Sequence[float]
) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for p in percentiles:
        out[float(p)] = float(np.percentile(nn_dists, float(p)))
    return out
