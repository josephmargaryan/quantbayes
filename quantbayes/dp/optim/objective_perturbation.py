from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from ..mechanisms.spherical_laplace import sample_spherical_laplace
from ..models.logistic_regression import logistic_grad


def train_objective_perturbed(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    eps: float,
    L: float = 1.0,
    steps: int = 5000,
    lr: float = 0.05,
    seed: Optional[int] = None,
    c: float = 0.25,  # Hessian bound for clipped logistic
    enforce_assumptions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lecture-aligned objective perturbation.
    Minimise: (1/n)∑ ell + (lam/2)||w||^2 + (1/n)<b,w>,  with  b ~ sLaplace(beta),
    where  beta = eps/(2L).  (No 'n' in beta.)
    Requires lam > c to apply the change-of-variables proof (Lecture 3.2, Theorem 2).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if L <= 0:
        raise ValueError("L must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if enforce_assumptions and lam <= c:
        raise ValueError(
            f"Objective perturbation requires lam > c (c={c}); got lam={lam}"
        )

    n, d = X.shape
    beta = eps / (2.0 * L)
    b = sample_spherical_laplace(beta=beta, d=d, seed=seed)

    w = np.zeros(d, dtype=float)
    for _ in range(steps):
        g = logistic_grad(w, X, y) + lam * w + (b / n)
        w -= lr * g
    return w, b
