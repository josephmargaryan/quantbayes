from __future__ import annotations
import numpy as np
from typing import Optional
from ..utils.rng import make_rng


def sample_spherical_laplace(
    beta: float, d: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample from the L2-spherical Laplace with density ∝ exp(-beta * ||x||_2) in R^d.
    Construction: x = R * u, where u ~ Unif(S^{d-1}), R ~ Gamma(shape=d, scale=1/beta).
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if d <= 0:
        raise ValueError("d must be >= 1")
    rng = make_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u) + 1e-12
    r = rng.gamma(shape=d, scale=1.0 / beta)
    return r * u
