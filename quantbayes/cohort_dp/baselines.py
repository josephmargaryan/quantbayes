# cohort_dp/baselines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .metrics import Metric


@dataclass
class NonPrivateKNNRetriever:
    """
    Deterministic exact kNN retrieval (no privacy).
    Uses the same API surface as the private retrievers.
    """

    X: np.ndarray
    metric: Metric

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)

    def privacy_cost(self, k: int) -> float:
        return 0.0

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
        kk = min(k, dists.shape[0])
        idx_local = np.argpartition(dists, kth=kk - 1)[:kk]
        idx_local = idx_local[np.argsort(dists[idx_local])]
        return pool[idx_local].astype(int)
