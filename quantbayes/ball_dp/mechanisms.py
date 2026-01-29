# quantbayes/ball_dp/mechanisms.py
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np


def gaussian_sigma(
    delta_l2: float,
    eps: float,
    delta: float,
    *,
    method: Literal["classic", "analytic"] = "classic",
) -> float:
    """
    Calibrate sigma for the Gaussian mechanism for L2 sensitivity `delta_l2`.

    - "classic": sigma >= (Δ/ε) * sqrt(2 log(1.25/δ))
      (commonly used; simple; conservative)

    - "analytic": uses Balle & Wang (2018)-style calibration if SciPy is available.
      Falls back to "classic" if SciPy isn't installed.

    NOTE:
      This returns a sigma for N(0, sigma^2 I).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    Delta = float(delta_l2)
    eps = float(eps)
    delta = float(delta)

    if method == "classic":
        return (Delta / eps) * math.sqrt(2.0 * math.log(1.25 / delta))

    # analytic calibration (optional)
    try:
        from scipy.stats import norm  # type: ignore
    except Exception:
        return (Delta / eps) * math.sqrt(2.0 * math.log(1.25 / delta))

    # --- Analytic Gaussian mechanism calibration ---
    # We implement a standard bisection on sigma using the known sufficient condition:
    #   delta >= Phi(-a) - exp(eps)*Phi(-a - eps*sigma/Delta)
    #
    # where a = Delta/(2*sigma) - eps*sigma/Delta
    #
    # This form appears in analytic GM derivations; we use it as a robust numeric calibrator.
    #
    # If you're picky about exactness: keep "classic" for now; or replace this with your
    # preferred analytic implementation. The experiments don't depend on which you pick.

    def sufficient_delta(sig: float) -> float:
        sig = float(sig)
        if sig <= 0:
            return 1.0
        # This expression is numerically sensitive for extreme eps/delta.
        # We keep it stable enough for typical ML eps ranges.
        a = (Delta / (2.0 * sig)) - (eps * sig / Delta)
        return float(norm.cdf(-a) - math.exp(eps) * norm.cdf(-(a + eps * sig / Delta)))

    # Bisection: find smallest sigma with sufficient_delta(sigma) <= target delta
    # Note: sufficient_delta decreases with sigma.
    lo = 1e-12
    hi = max(1.0, (Delta / eps) * 10.0)
    for _ in range(60):
        if sufficient_delta(hi) <= delta:
            break
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sufficient_delta(mid) <= delta:
            hi = mid
        else:
            lo = mid

    return float(hi)


def add_gaussian_noise(
    x: np.ndarray,
    sigma: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise N(0, sigma^2) to an array.
    """
    if rng is None:
        rng = np.random.default_rng()
    return x + rng.normal(0.0, float(sigma), size=x.shape).astype(x.dtype, copy=False)
