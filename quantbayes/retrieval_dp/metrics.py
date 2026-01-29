from __future__ import annotations

from typing import Literal
import numpy as np

ScoreType = Literal["dot", "neg_l2"]


def l2_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X)
    return np.sqrt(np.sum(X * X, axis=1) + float(eps))


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X)
    n = l2_norm_rows(X, eps=eps).reshape(-1, 1)
    return (X / n).astype(X.dtype, copy=False)


def clip_l2_rows(X: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 clipping: x <- x * min(1, max_norm / ||x||).
    """
    X = np.asarray(X)
    max_norm = float(max_norm)
    n = l2_norm_rows(X, eps=eps)
    scale = np.minimum(1.0, max_norm / (n + float(eps)))
    return (X * scale.reshape(-1, 1)).astype(X.dtype, copy=False)


def dot_scores(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Q: (B,d), V: (m,d) => scores: (B,m) where higher is better.
    """
    Q = np.asarray(Q)
    V = np.asarray(V)
    return Q @ V.T


def neg_l2_scores(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Negative Euclidean distance scores: u = -||q - v||_2 (higher is better).
    Uses a stable vectorized computation.
    """
    Q = np.asarray(Q, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    Q2 = np.sum(Q * Q, axis=1, keepdims=True)  # (B,1)
    V2 = np.sum(V * V, axis=1, keepdims=True).T  # (1,m)
    D2 = np.maximum(Q2 + V2 - 2.0 * (Q @ V.T), 0.0)
    D = np.sqrt(D2)
    return (-D).astype(np.float32)
