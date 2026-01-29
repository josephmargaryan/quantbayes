# quantbayes/ball_dp/metrics.py
from __future__ import annotations

import numpy as np


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_norms(Z: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Z, axis=1)


def maybe_l2_normalize(Z: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return Z
    n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return (Z / n).astype(Z.dtype, copy=False)
