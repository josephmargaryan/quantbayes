# quantbayes/ball_dp/metrics.py
from __future__ import annotations

import numpy as np


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def l2_norms(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z)
    return np.linalg.norm(Z, axis=1)


def maybe_l2_normalize(Z: np.ndarray, enabled: bool, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: z <- z / (||z|| + eps).
    """
    Z = np.asarray(Z)
    if not enabled:
        return Z
    n = np.linalg.norm(Z, axis=1, keepdims=True) + float(eps)
    return (Z / n).astype(Z.dtype, copy=False)


def clip_l2_rows(Z: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 clipping: z <- z * min(1, max_norm / (||z|| + eps)).

    This is important for DP baselines that assume a PUBLIC bound ||z|| <= B.
    If you claim B is a public bound, you should enforce it via clipping/normalization.
    """
    Z = np.asarray(Z)
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")

    n = np.linalg.norm(Z, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (n + float(eps)))
    return (Z * scale).astype(Z.dtype, copy=False)
