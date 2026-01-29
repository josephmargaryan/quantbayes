# quantbayes/ball_dp/radius.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

    Returns:
        nn_dists: shape (sum_c min(per_class, n_c),)
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
        A = Z[sub].astype(np.float64, copy=False)  # for numerical stability

        # Pairwise squared distances: ||a||^2 + ||b||^2 - 2 a·b
        AA = (A * A).sum(axis=1, keepdims=True)
        D2 = AA + AA.T - 2.0 * (A @ A.T)
        np.fill_diagonal(D2, np.inf)
        # min distance per row
        nn = np.sqrt(np.min(np.maximum(D2, 0.0), axis=1))
        all_nn.append(nn.astype(np.float32))

    if not all_nn:
        raise RuntimeError(
            "Could not compute within-class NN distances (insufficient per-class samples)."
        )
    return np.concatenate(all_nn, axis=0)


def radii_from_percentiles(
    nn_dists: np.ndarray, percentiles: Sequence[float]
) -> Dict[float, float]:
    """
    Convert NN distance distribution into radius values by percentiles.
    """
    out: Dict[float, float] = {}
    for p in percentiles:
        out[float(p)] = float(np.percentile(nn_dists, float(p)))
    return out


def coverage_curve(
    nn_dists: np.ndarray, radii: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical coverage curve Cov(r)=P(nn_dist <= r) for provided radii.
    """
    nn = np.asarray(nn_dists, dtype=np.float64)
    rs = np.asarray(list(radii), dtype=np.float64)
    cov = np.array([(nn <= r).mean() for r in rs], dtype=np.float64)
    return rs, cov


def dp_quantile_exponential_mechanism(
    values: np.ndarray,
    quantile: float,
    eps: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    A simple DP quantile selector using the exponential mechanism over the set of
    observed values.

    Score function:
        s(i) = -|rank(i) - q*n|
    Sensitivity of score w.r.t changing one record is 1 in rank units, so
    exp-mech uses probability proportional to exp( (eps/2) * s(i) ).

    This is a standard, clean primitive to include in the paper as:
      "DP selection of r" (spend a small epsilon_r once).

    IMPORTANT:
      This is DP with respect to the multiset `values` itself. If `values`
      are derived from private data, you need to argue/allocate privacy for
      that derivation as well (composition, or treat r as public/policy-chosen).

    Returns:
        a DP-selected value approximating the desired quantile.
    """
    if rng is None:
        rng = np.random.default_rng()
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1 or vals.size < 2:
        raise ValueError("values must be a 1D array with at least 2 elements")
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must be in [0,1]")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    # candidate set = sorted observed values
    xs = np.sort(vals)
    n = xs.size
    target_rank = q * (n - 1)

    ranks = np.arange(n, dtype=np.float64)
    score = -np.abs(ranks - target_rank)  # sensitivity 1
    # exp mech: p(i) ∝ exp((eps/2)*score)
    logits = (float(eps) / 2.0) * score
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    idx = rng.choice(np.arange(n), p=w)
    return float(xs[int(idx)])
