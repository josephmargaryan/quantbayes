from __future__ import annotations
import math
from typing import Iterable


def zcdp_rho_gaussian(delta_l2: float, sigma: float) -> float:
    """
    zCDP rho for Gaussian mechanism on an L2-sensitive query.
    rho = Δ^2 / (2 sigma^2)
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return (delta_l2**2) / (2.0 * (sigma**2))


def compose_rho(rhos: Iterable[float]) -> float:
    """Additive composition for zCDP."""
    total = 0.0
    for r in rhos:
        if r < 0:
            raise ValueError("rho must be >= 0")
        total += r
    return total


def zcdp_to_epsdelta(rho: float, delta: float) -> float:
    """
    Convert rho-zCDP to (eps, delta)-DP via:
      eps = rho + 2 sqrt(rho * log(1/delta))
    (Standard bound used in many notes.)
    """
    if rho < 0:
        raise ValueError("rho must be >= 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def epsdelta_to_rho(eps: float, delta: float) -> float:
    """
    Invert eps = rho + 2 sqrt(rho log(1/delta)).

    Let a = log(1/delta), t = sqrt(rho). Then eps = t^2 + 2 t sqrt(a).
    => t = -sqrt(a) + sqrt(a + eps), rho = t^2.

    Returns rho >= 0.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    a = math.log(1.0 / delta)
    t = -math.sqrt(a) + math.sqrt(a + eps)
    rho = max(0.0, t * t)
    return rho


def zcdp_sigma_for_gd(eps: float, delta: float, T: int, delta_l2: float) -> float:
    """
    Calibrate per-step Gaussian noise sigma for DP-GD under zCDP composition.

    Per-step: rho_t = Δ^2 / (2 sigma^2).
    Total: rho = T * rho_t.
    Choose rho from (eps, delta) via epsdelta_to_rho, solve for sigma.

    Returns sigma > 0.
    """
    if T <= 0:
        raise ValueError("T must be >= 1")
    rho_total = epsdelta_to_rho(eps, delta)
    if rho_total <= 0:
        raise ValueError("Derived rho <= 0; increase eps or delta.")
    sigma = delta_l2 * math.sqrt(T / (2.0 * rho_total))
    return sigma
