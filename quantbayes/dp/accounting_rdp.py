from __future__ import annotations
import math
from typing import Iterable, List, Sequence, Tuple, Optional


def rdp_gaussian(alpha: float, delta_l2: float, sigma: float) -> float:
    """
    Rényi DP of order alpha (>1) for a Gaussian mechanism with L2-sensitivity Δ:
      RDP(alpha) = alpha * Δ^2 / (2 sigma^2)
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return alpha * (delta_l2**2) / (2.0 * (sigma**2))


def compose_rdp_per_step(alpha: float, rdp_per_step: float, T: int) -> float:
    """Additive composition in RDP space."""
    if T <= 0:
        raise ValueError("T must be >= 1")
    return T * rdp_per_step


def rdp_to_eps(rdp_total: float, alpha: float, delta: float) -> float:
    """
    Convert total RDP at order alpha to (eps, delta)-DP:
      eps(alpha) = rdp_total + log(1/delta) / (alpha - 1)
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return rdp_total + math.log(1.0 / delta) / (alpha - 1.0)


def eps_from_sigma_rdp(
    sigma: float,
    delta_l2: float,
    T: int,
    delta: float,
    orders: Sequence[float] = (1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256),
) -> Tuple[float, float]:
    """
    Given sigma, compute the tightest eps over a grid of alpha orders.
    Returns (eps_min, alpha_star).
    """
    best_eps = float("inf")
    best_alpha = None
    for a in orders:
        if a <= 1:
            continue
        rdp_step = rdp_gaussian(a, delta_l2, sigma)
        rdp_tot = compose_rdp_per_step(a, rdp_step, T)
        eps_a = rdp_to_eps(rdp_tot, a, delta)
        if eps_a < best_eps:
            best_eps = eps_a
            best_alpha = a
    return best_eps, float(best_alpha or 2.0)


def sigma_for_target_eps_rdp(
    eps: float,
    delta_l2: float,
    T: int,
    delta: float,
    orders: Sequence[float] = (1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256),
) -> Tuple[float, float]:
    """
    Calibrate sigma to meet target (eps, delta) by minimizing sigma over alpha grid.
      For each alpha: eps >= T * alpha * Δ^2 / (2 sigma^2) + log(1/delta)/(alpha-1)
      => sigma^2 >= T * alpha * Δ^2 / (2 * (eps - log(1/delta)/(alpha-1)))
    Returns (sigma_min, alpha_star).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if T <= 0:
        raise ValueError("T must be >= 1")
    best_sigma2 = float("inf")
    best_alpha = None
    log1d = math.log(1.0 / delta)
    for a in orders:
        if a <= 1:
            continue
        denom = eps - (log1d / (a - 1.0))
        if denom <= 0:
            continue  # infeasible alpha for this eps
        sigma2 = (T * a * (delta_l2**2)) / (2.0 * denom)
        if sigma2 < best_sigma2:
            best_sigma2 = sigma2
            best_alpha = a
    if not math.isfinite(best_sigma2):
        raise ValueError(
            "No feasible alpha found for given (eps, delta, T); increase eps or delta."
        )
    return math.sqrt(best_sigma2), float(best_alpha or 2.0)
