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


@dataclass(frozen=True)
class WithinClassNNPairs:
    """
    Within-class NN pairs over a sampled subset per class.

    i, j are ORIGINAL indices into the provided arrays Z,y (not local indices).
    dist is the corresponding Euclidean distance in the embedding space.

    This powers:
      - per-class tail tables
      - surfacing "outlier" examples (largest NN distances)
      - metric-misalignment stress tests (evaluate these pairs under another metric)
    """

    i: np.ndarray  # (N,)
    j: np.ndarray  # (N,)
    dist: np.ndarray  # (N,)
    cls: np.ndarray  # (N,)


def within_class_nn_pairs(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    per_class: int = 400,
    seed: int = 0,
) -> WithinClassNNPairs:
    """
    Return within-class NN pairs (i_idx, nn_idx, dist) on a random subset per class.

    This is like within_class_nn_distances(), but returns indices so you can:
      - attribute tails to specific classes
      - surface concrete outliers (image + its NN)
      - reuse the same pairs to evaluate distances under an alternative metric/space
        for misalignment checks.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)

    I_all: List[np.ndarray] = []
    J_all: List[np.ndarray] = []
    D_all: List[np.ndarray] = []
    C_all: List[np.ndarray] = []

    for c in range(int(num_classes)):
        idx = np.where(y == c)[0]
        if idx.size < 2:
            continue

        take = int(min(per_class, idx.size))
        sub = rng.choice(idx, size=take, replace=False)
        A = Z[sub].astype(np.float64, copy=False)

        # Pairwise squared distances: ||a||^2 + ||b||^2 - 2 a·b
        AA = (A * A).sum(axis=1, keepdims=True)
        D2 = AA + AA.T - 2.0 * (A @ A.T)
        np.fill_diagonal(D2, np.inf)

        j_local = np.argmin(D2, axis=1)
        d2 = D2[np.arange(take), j_local]
        d = np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)

        I_all.append(sub.astype(np.int64))
        J_all.append(sub[j_local].astype(np.int64))
        D_all.append(d)
        C_all.append(np.full((take,), int(c), dtype=np.int64))

    if not D_all:
        raise RuntimeError(
            "Could not compute within-class NN pairs (no valid classes)."
        )

    return WithinClassNNPairs(
        i=np.concatenate(I_all, axis=0),
        j=np.concatenate(J_all, axis=0),
        dist=np.concatenate(D_all, axis=0),
        cls=np.concatenate(C_all, axis=0),
    )


def per_class_tail_stats_from_nn_pairs(
    pairs: WithinClassNNPairs,
    *,
    r: float,
    thresholds: Sequence[float] = (1.25, 1.50),
) -> List[Dict[str, object]]:
    """
    Build a per-class tail table from within-class NN pairs.

    Reports:
      - median/p95/p99 of NN distances
      - fraction of multipliers dist/r exceeding given thresholds
    """
    r = float(r)
    if r <= 0:
        raise ValueError("r must be > 0 for multiplier stats.")
    thresholds = tuple(float(t) for t in thresholds)

    out: List[Dict[str, object]] = []
    cls_all = np.asarray(pairs.cls, dtype=np.int64)
    dist_all = np.asarray(pairs.dist, dtype=np.float64)

    for c in np.unique(cls_all):
        mask = cls_all == int(c)
        d = dist_all[mask]
        if d.size == 0:
            continue
        mult = d / r

        row: Dict[str, object] = {
            "class": int(c),
            "n_pairs": int(d.size),
            "nn_med": float(np.median(d)),
            "nn_p95": float(np.quantile(d, 0.95)),
            "nn_p99": float(np.quantile(d, 0.99)),
            "mult_med": float(np.median(mult)),
            "mult_p95": float(np.quantile(mult, 0.95)),
            "mult_p99": float(np.quantile(mult, 0.99)),
        }
        for t in thresholds:
            row[f"frac_mult_gt_{t:g}"] = float(np.mean(mult > t))
        out.append(row)

    # Sort by worst tail (p99 multiplier), descending
    out.sort(key=lambda r0: float(r0["mult_p99"]), reverse=True)
    return out


def top_outlier_pairs(
    pairs: WithinClassNNPairs,
    *,
    top_k: int = 12,
) -> List[Dict[str, object]]:
    """
    Return the top-k within-class NN pairs by distance, as JSON-safe dict rows.
    Useful for "show me the worst examples" in an appendix.
    """
    d = np.asarray(pairs.dist, dtype=np.float64)
    if d.size == 0:
        return []
    top_k = int(min(max(1, top_k), d.size))
    idx = np.argsort(-d)[:top_k]

    rows: List[Dict[str, object]] = []
    for t in idx.tolist():
        rows.append(
            {
                "i": int(pairs.i[t]),
                "j": int(pairs.j[t]),
                "class": int(pairs.cls[t]),
                "dist": float(pairs.dist[t]),
            }
        )
    return rows


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
    lo: float,
    hi: float,
    grid_size: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    DP quantile selection using the exponential mechanism over a FIXED grid.

    Output range is data-independent: a uniform grid over [lo, hi].
    This avoids the common pitfall of using a data-dependent candidate set.

    Utility for candidate x:
        u(x) = -| rank(x) - q*n |
    where rank(x) = #{v_i <= x}.
    Changing one record changes rank(x) by at most 1 for every x, so sensitivity Δu = 1.

    Therefore:
        P[x] ∝ exp((eps / (2*Δu)) * u(x)) = exp((eps/2) * u(x))

    Args:
      lo, hi: must be PUBLIC bounds on the support of `values`.
              For embedding distances with ||z||<=B, you can take lo=0, hi=2B.
    """
    if rng is None:
        rng = np.random.default_rng()

    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size < 2:
        raise ValueError("values must have at least 2 elements")
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must be in [0,1]")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    lo = float(lo)
    hi = float(hi)
    if not (lo < hi):
        raise ValueError("require lo < hi")
    grid_size = int(grid_size)
    if grid_size < 16:
        raise ValueError("grid_size too small; use at least 16")

    grid = np.linspace(lo, hi, grid_size, dtype=np.float64)

    xs = np.sort(vals)
    # rank(x) = #{v <= x}
    ranks = np.searchsorted(xs, grid, side="right").astype(np.float64)

    n = float(xs.size)
    target = q * n
    score = -np.abs(ranks - target)  # sensitivity 1

    logits = (eps / 2.0) * score
    logits -= np.max(logits)
    w = np.exp(logits)
    w /= np.sum(w)

    idx = rng.choice(np.arange(grid_size), p=w)
    return float(grid[int(idx)])
