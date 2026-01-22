# cohort_dp/mechanisms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .metrics import Metric


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0 or not np.isfinite(s):
        raise FloatingPointError("Softmax normalization failed.")
    return exps / s


@dataclass
class ExponentialMechanismRetriever:
    """
    Exponential mechanism for selecting neighbors by distance.

    We interpret eps_total as the privacy budget PER API CALL.
    For top-k, we do k sequential draws with eps_draw = eps_total / k
    (basic composition => total cost ~ eps_total).
    """

    X: np.ndarray
    metric: Metric
    r: float
    eps_total: float
    rng: np.random.Generator

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("r must be > 0.")
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")
        self.X = np.asarray(self.X, dtype=float)

    def privacy_cost(self, k: int) -> float:
        # basic composition for sequential selection
        return float(self.eps_total)

    def query(
        self, z: np.ndarray, k: int = 1, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")
        z = np.asarray(z, dtype=float).reshape(1, -1)

        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D array of indices.")

        # allocate eps across sequential draws
        eps_draw = self.eps_total / float(k)

        remaining = pool.copy()
        chosen: List[int] = []

        for _ in range(k):
            if remaining.size == 0:
                break
            dists = self.metric.pairwise(z, self.X[remaining]).reshape(-1)
            logits = -(eps_draw / (2.0 * self.r)) * dists
            p = _stable_softmax(logits)
            j = int(self.rng.choice(remaining.size, p=p))
            chosen_idx = int(remaining[j])
            chosen.append(chosen_idx)
            remaining = np.delete(remaining, j)

        return np.array(chosen, dtype=int)


@dataclass
class NoisyTopKRetriever:
    """
    Noisy distances + top-k (indices-only output).
    eps_total is the privacy budget PER API CALL.

    We add Laplace noise to each distance with scale b = r / eps_total.
    If we were to release the entire noisy distance vector, this would be eps_total-DP
    under the restricted neighbor relation (one record changes within distance r),
    since only one coordinate can change by at most r (L1 sensitivity <= r).
    Returning only indices is post-processing.
    """

    X: np.ndarray
    metric: Metric
    r: float
    eps_total: float
    rng: np.random.Generator

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("r must be > 0.")
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")
        self.X = np.asarray(self.X, dtype=float)

    def privacy_cost(self, k: int) -> float:
        return float(self.eps_total)

    def query(
        self, z: np.ndarray, k: int = 10, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")
        z = np.asarray(z, dtype=float).reshape(1, -1)

        if candidates is None:
            pool = np.arange(self.X.shape[0], dtype=int)
        else:
            pool = np.asarray(candidates, dtype=int)
            if pool.ndim != 1 or pool.size == 0:
                raise ValueError("candidates must be a non-empty 1D array of indices.")

        dists = self.metric.pairwise(z, self.X[pool]).reshape(-1)

        scale = self.r / self.eps_total
        noise = self.rng.laplace(loc=0.0, scale=scale, size=dists.shape[0])
        noisy = dists + noise

        kk = min(k, noisy.shape[0])
        idx_local = np.argpartition(noisy, kth=kk - 1)[:kk]
        idx_local = idx_local[np.argsort(noisy[idx_local])]
        return pool[idx_local].astype(int)
