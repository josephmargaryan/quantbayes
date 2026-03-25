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
    c: float = 0.25,  # Hessian bound under ||x||<=1
    enforce_assumptions: bool = True,
    return_noise: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Objective perturbation (Lecture 3.2, Thm 2).
    Minimise: (1/n)∑ ell + (lam/2)||w||^2 + (1/n)<b,w>, with b ~ sLaplace(beta), beta=eps/(2L).
    Requires lam > c for the theorem to apply.

    By default returns only the private weights. Set return_noise=True for internal testing.
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

    return (w, b) if return_noise else w
