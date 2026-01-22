# cohort_dp/analysis.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from .metrics import Metric
from .eval import true_knn


def mean_distance(
    metric: Metric, X: np.ndarray, z: np.ndarray, idx: np.ndarray
) -> float:
    z = np.asarray(z, dtype=float).reshape(1, -1)
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return float("inf")
    d = metric.pairwise(z, X[idx]).reshape(-1)
    return float(np.mean(d))


def estimate_r_global(
    X: np.ndarray, metric: Metric, k0: int, rng: np.random.Generator, m: int = 200
) -> float:
    n = X.shape[0]
    idxs = rng.choice(n, size=min(m, n), replace=False)
    ds = []
    for i in idxs:
        z = X[i]
        nn = true_knn(metric, X, z, k=k0 + 1)
        kth = nn[min(k0, len(nn) - 1)]
        ds.append(float(metric.pairwise(z.reshape(1, -1), X[kth].reshape(1, -1))[0, 0]))
    return float(np.median(ds))


def estimate_r_from_local_density(
    X: np.ndarray, metric: Metric, k0: int, rng: np.random.Generator, m: int = 200
) -> float:
    # same estimator, different naming used in your scripts
    return estimate_r_global(X, metric, k0=k0, rng=rng, m=m)


def compute_ball_density_stats(
    X: np.ndarray, metric: Metric, r: float, rng: np.random.Generator, m: int = 200
) -> Dict[str, float]:
    n = X.shape[0]
    m = min(m, n)
    idxs = rng.choice(n, size=m, replace=False)
    counts = []
    for i in idxs:
        z = X[i].reshape(1, -1)
        d = metric.pairwise(z, X).reshape(-1)
        c = int(np.sum(d <= r) - 1)  # exclude itself
        counts.append(c)

    arr = np.array(counts, dtype=float)
    return {
        "m": float(m),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def attacker_exact_within_r(
    api,
    X_db: np.ndarray,
    metric: Metric,
    r_global: float,
    attacker,
    targets: np.ndarray,
) -> Tuple[float, float]:
    exact = 0
    within = 0
    for t in targets:
        t = int(t)
        pred = int(attacker.attack(api, X_db, t))
        exact += int(pred == t)
        dist = metric.pairwise(X_db[t].reshape(1, -1), X_db[pred].reshape(1, -1))[0, 0]
        within += int(dist <= r_global)
    n = float(len(targets))
    return exact / n, within / n
