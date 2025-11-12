from __future__ import annotations
import numpy as np


def clip_rows_l2(X: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    L2-clip each row of X to have norm at most max_norm.
    Returns a copy.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, max_norm / norms)
    return X * scale
