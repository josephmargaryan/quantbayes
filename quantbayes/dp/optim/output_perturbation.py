from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from ..mechanisms.spherical_laplace import sample_spherical_laplace


def output_perturbation(
    w_hat: np.ndarray,
    n: int,
    lam: float,
    eps: float,
    L: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Output perturbation for strongly convex ERM (Lecture 3.2, Theorem 1).
    Sensitivity Δ = 2L/(n*lam). Use L2-Laplace with parameter beta = eps/Δ = (eps*n*lam)/(2L).
    Returns (w_priv, noise_vector).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")
    d = int(w_hat.shape[0])
    beta = (eps * n * lam) / (2.0 * L)
    b = sample_spherical_laplace(beta=beta, d=d, seed=seed)
    return w_hat + b, b
