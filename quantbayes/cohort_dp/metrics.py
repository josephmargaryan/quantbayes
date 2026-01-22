# cohort_dp/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


class Metric(Protocol):
    def pairwise(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return pairwise distances with shape (Z.shape[0], X.shape[0])."""


@dataclass(frozen=True)
class L2Metric:
    """Standard Euclidean distance."""

    def pairwise(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float)
        # ||z - x||^2 = ||z||^2 + ||x||^2 - 2 z^T x
        Z2 = np.sum(Z * Z, axis=1, keepdims=True)  # (m, 1)
        X2 = np.sum(X * X, axis=1, keepdims=True).T  # (1, n)
        d2 = np.maximum(Z2 + X2 - 2.0 * (Z @ X.T), 0.0)
        return np.sqrt(d2)


@dataclass(frozen=True)
class WeightedL2Metric:
    """
    Weighted L2 / diagonal Mahalanobis:
      d(x, x') = || W (x - x') ||_2
    where W is diagonal with positive entries (shape (d,)).
    """

    w: np.ndarray  # shape (d,)

    def __post_init__(self):
        w = np.asarray(self.w, dtype=float)
        if w.ndim != 1 or np.any(w <= 0):
            raise ValueError("w must be a 1D array with strictly positive entries.")
        object.__setattr__(self, "w", w)

    def pairwise(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float)
        Zw = Z * self.w[None, :]
        Xw = X * self.w[None, :]
        Z2 = np.sum(Zw * Zw, axis=1, keepdims=True)
        X2 = np.sum(Xw * Xw, axis=1, keepdims=True).T
        d2 = np.maximum(Z2 + X2 - 2.0 * (Zw @ Xw.T), 0.0)
        return np.sqrt(d2)
