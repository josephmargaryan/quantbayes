# cohort_dp/mechanisms_topk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .metrics import Metric


@dataclass
class OneshotLaplaceTopKRetriever:
    """
    DP-ish baseline: oneshot Laplace top-k on scores.

    We score points by s_i = -dist(z, x_i). Under restricted neighbor relation
    where a single record can move within radius r, the score sensitivity is <= r
    by triangle inequality:
        |dist(z,x) - dist(z,x')| <= dist(x,x') <= r
    Report-noisy-max style selection typically uses Laplace scale 2*Delta/eps.
    Here we use scale = 2*r/eps_total on the scores and select top-k.

    Reference baseline: oneshot DP top-k selection literature. (Use as a strong baseline.)
    """

    X: np.ndarray
    metric: Metric
    r: float
    eps_total: float
    rng: np.random.Generator

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        if self.r <= 0:
            raise ValueError("r must be > 0.")
        if self.eps_total <= 0:
            raise ValueError("eps_total must be > 0.")

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
                raise ValueError("candidates must be a non-empty 1D array.")

        dists = self.metric.pairwise(z, self.X[pool]).reshape(-1)
        scores = -dists  # higher is better

        scale = (2.0 * self.r) / float(self.eps_total)
        noise = self.rng.laplace(loc=0.0, scale=scale, size=scores.shape[0])
        noisy_scores = scores + noise

        kk = min(int(k), int(noisy_scores.shape[0]))
        idx_local = np.argpartition(-noisy_scores, kth=kk - 1)[:kk]
        idx_local = idx_local[np.argsort(-noisy_scores[idx_local])]
        return pool[idx_local].astype(int)
