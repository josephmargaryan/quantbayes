from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Sequence
from ..accounting_rdp import sigma_for_target_eps_rdp, eps_from_sigma_rdp
from ..models.logistic_regression import logistic_grad
from .projectors import project_l2_ball


def dp_gradient_descent_rdp(
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
    orders: Sequence[float] = (1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256),
) -> Tuple[np.ndarray, float, float]:
    """
    DP full-batch GD with Gaussian noise calibrated via RDP (tighter than zCDP).
    Returns (w_priv, sigma_used, eps_achieved_RDP).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if L <= 0:
        raise ValueError("L must be > 0")
    n, d = X.shape

    Delta = (2.0 * L) / n  # sensitivity of average gradient
    sigma, alpha_star = sigma_for_target_eps_rdp(
        eps=eps, delta_l2=Delta, T=steps, delta=delta, orders=orders
    )

    rng = np.random.RandomState(seed)
    w = np.zeros(d, dtype=float)
    w_sum = np.zeros(d, dtype=float)
    for _ in range(steps):
        g = logistic_grad(w, X, y) + lam * w
        noise = rng.normal(0.0, sigma, size=d)
        w = w - lr * (g + noise)
        w = project_l2_ball(w, proj_radius)
        if average:
            w_sum += w
    w_out = (w_sum / steps) if average else w

    # Report achieved eps (optimize over orders again for the used sigma)
    eps_achieved, _ = eps_from_sigma_rdp(
        sigma=sigma, delta_l2=Delta, T=steps, delta=delta, orders=orders
    )
    return w_out, float(sigma), float(eps_achieved)
