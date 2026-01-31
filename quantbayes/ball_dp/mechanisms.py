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
    tol: float = 1e-12,
) -> float:
    """
    Calibrate sigma for the Gaussian mechanism for L2 sensitivity `delta_l2`.

    Returns sigma for adding N(0, sigma^2 I) noise.

    - "classic": sigma >= (Δ/ε) * sqrt(2 log(1.25/δ))  (simple sufficient; conservative)
    - "analytic": analytic Gaussian calibration (Balle & Wang, 2018), returning the
      (numerically) minimal sigma for (ε, δ)-DP given L2 sensitivity Δ.
      Uses our stdlib-only implementation in `.analytical_gaussian_mechanism`.
    """
    eps = float(eps)
    delta = float(delta)
    Delta = float(delta_l2)

    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if Delta < 0.0:
        raise ValueError("delta_l2 must be >= 0")
    if Delta == 0.0:
        return 0.0
    if tol <= 0.0:
        raise ValueError("tol must be > 0")

    if method == "classic":
        return (Delta / eps) * math.sqrt(2.0 * math.log(1.25 / delta))

    if method == "analytic":
        from .analytical_gaussian_mechanism import calibrate_analytic_gaussian

        return calibrate_analytic_gaussian(
            epsilon=eps,
            delta=delta,
            GS=Delta,
            tol=tol,
        )

    raise ValueError(f"Unknown method={method!r}. Use 'classic' or 'analytic'.")


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
