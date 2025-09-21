from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from ..accounting import zcdp_sigma_for_gd
from ..models.logistic_regression import logistic_grad
from .projectors import project_l2_ball


def _lecture_sigma(eps: float, delta: float, T: int, L: float, n: int) -> float:
    """
    Per-step Gaussian std per Lecture 3.2 (Alg. 4 line).
    ν_t ~ N(0, σ^2 I_d) with
        σ = (2 * sqrt(2T) * L / (n * eps)) * log(T/delta) * log(1.25/delta)
    (Conservative; matches the handout.)
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return (
        (2.0 * (2.0 * T) ** 0.5 * L / (n * eps))
        * np.log(T / delta)
        * np.log(1.25 / delta)
    )


def dp_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    eps: float,
    delta: float,
    steps: int = 1000,
    lr: float = 0.1,
    L: float = 1.0,
    proj_radius: Optional[float] = None,
    average: bool = True,
    seed: Optional[int] = None,
    calibration: str = "zcdp",  # "zcdp" or "lecture"
) -> Tuple[np.ndarray, float]:
    """
    Full-batch DP-GD for L2-regularized logistic regression.
    Sensitivity of average gradient: Δ = 2L/n.
    - calibration="lecture": match the handout's σ.
    - calibration="zcdp"   : your tighter default.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if L <= 0:
        raise ValueError("L must be > 0")
    n, d = X.shape

    if calibration == "lecture":
        sigma = _lecture_sigma(eps, delta, steps, L, n)
    elif calibration == "zcdp":
        Delta = (2.0 * L) / n
        sigma = zcdp_sigma_for_gd(eps=eps, delta=delta, T=steps, delta_l2=Delta)
    else:
        raise ValueError("calibration must be 'zcdp' or 'lecture'")

    rng = np.random.RandomState(seed)
    w = np.zeros(d, dtype=float)
    w_sum = np.zeros(d, dtype=float)
    for _ in range(steps):
        g = logistic_grad(w, X, y) + lam * w
        noise = rng.normal(0.0, sigma, size=d)
        w = project_l2_ball(w - lr * (g + noise), proj_radius)
        if average:
            w_sum += w
    return (w_sum / steps) if average else w, float(sigma)
