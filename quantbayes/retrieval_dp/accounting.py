# quantbayes/retrieval_dp/accounting.py
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple


def advanced_composition(
    *,
    eps: float,
    delta: float,
    T: int,
    delta_prime: float,
) -> Tuple[float, float]:
    """
    Advanced composition (Dwork-Roth style) for T-fold composition of (eps, delta)-DP.

    For any delta_prime > 0:
      delta_tot = T*delta + delta_prime
      eps_tot = sqrt(2T log(1/delta_prime)) * eps + T*eps*(exp(eps)-1)

    Notes:
    - This is a standard, simple bound (no subsampling).
    - For small eps, the second term is ~ T*eps^2.
    """
    eps = float(eps)
    delta = float(delta)
    delta_prime = float(delta_prime)
    T = int(T)

    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0.0 <= delta < 1.0):
        raise ValueError("delta must be in [0,1)")
    if not (0.0 < delta_prime < 1.0):
        raise ValueError("delta_prime must be in (0,1)")
    if T <= 0:
        raise ValueError("T must be >= 1")

    delta_tot = T * delta + delta_prime
    eps_tot = math.sqrt(2.0 * T * math.log(1.0 / delta_prime)) * eps + T * eps * (
        math.exp(eps) - 1.0
    )
    return float(eps_tot), float(delta_tot)


def gaussian_rdp_epsilon(alpha: float, *, Delta: float, sigma: float) -> float:
    """
    RDP of the Gaussian mechanism at order alpha > 1:

      eps_RDP(alpha) = alpha * Delta^2 / (2 sigma^2)

    where Delta is L2 sensitivity and noise is N(0, sigma^2 I).

    This is the basic (no-subsampling) Gaussian RDP.
    """
    alpha = float(alpha)
    Delta = float(Delta)
    sigma = float(sigma)
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if Delta < 0:
        raise ValueError("Delta must be >= 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return float(alpha * (Delta * Delta) / (2.0 * sigma * sigma))


def rdp_to_dp_epsilon(alpha: float, eps_rdp: float, *, delta: float) -> float:
    """
    Convert RDP to (eps, delta)-DP:

      eps(delta) = eps_rdp(alpha) + log(1/delta) / (alpha - 1)

    Returns eps at the chosen alpha.
    """
    alpha = float(alpha)
    eps_rdp = float(eps_rdp)
    delta = float(delta)
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    return float(eps_rdp + math.log(1.0 / delta) / (alpha - 1.0))


def compose_gaussian_queries_via_rdp(
    *,
    T: int,
    Delta: float,
    sigma: float,
    delta: float,
    alphas: Optional[Iterable[float]] = None,
) -> Tuple[float, float, float]:
    """
    Compose T Gaussian mechanisms (same Delta, sigma) using RDP and convert to (eps, delta)-DP.

    Returns:
      (eps_total, alpha_star, eps_rdp_total_at_alpha_star)

    Choose alpha by minimizing:
      T * eps_rdp(alpha) + log(1/delta)/(alpha-1)
    """
    T = int(T)
    Delta = float(Delta)
    sigma = float(sigma)
    delta = float(delta)

    if T <= 0:
        raise ValueError("T must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if Delta < 0:
        raise ValueError("Delta must be >= 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    if alphas is None:
        # A small but practical grid; adjust as needed
        alphas = [1.25, 1.5, 2, 3, 5, 8, 16, 32, 64, 128]

    best = (float("inf"), None, None)  # (eps_total, alpha, eps_rdp_total)

    for a in alphas:
        a = float(a)
        if a <= 1.0:
            continue
        eps_rdp = gaussian_rdp_epsilon(a, Delta=Delta, sigma=sigma)
        eps_rdp_total = T * eps_rdp
        eps_total = rdp_to_dp_epsilon(a, eps_rdp_total, delta=delta)
        if eps_total < best[0]:
            best = (eps_total, a, eps_rdp_total)

    if best[1] is None:
        raise RuntimeError("No valid alpha values provided.")
    return float(best[0]), float(best[1]), float(best[2])
