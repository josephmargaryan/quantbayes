# quantbayes/ball_dp/analytical_gaussian_mechanism.py
from __future__ import annotations

import math
from typing import Callable


def _phi(t: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _case_a(eps: float, s: float) -> float:
    return _phi(math.sqrt(eps * s)) - math.exp(eps) * _phi(-math.sqrt(eps * (s + 2.0)))


def _case_b(eps: float, s: float) -> float:
    return _phi(-math.sqrt(eps * s)) - math.exp(eps) * _phi(-math.sqrt(eps * (s + 2.0)))


def _doubling_trick(
    predicate: Callable[[float], bool],
    start: float = 0.0,
    step: float = 1.0,
    max_s: float = 1e12,
) -> float:
    s = float(start)
    step = float(step)
    while not predicate(s):
        s += step
        step *= 2.0
        if s > max_s:
            raise RuntimeError(
                "Failed to bracket solution in analytic Gaussian calibration."
            )
    return s


def _binary_search(
    predicate: Callable[[float], bool],
    left: float,
    right: float,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> float:
    left = float(left)
    right = float(right)
    for _ in range(int(max_iter)):
        mid = 0.5 * (left + right)
        if (right - left) <= float(tol) * (1.0 + abs(mid)):
            return mid
        if predicate(mid):
            right = mid
        else:
            left = mid
    return 0.5 * (left + right)


def calibrate_analytic_gaussian(
    *, epsilon: float, delta: float, GS: float, tol: float = 1e-12
) -> float:
    """
    Analytic calibration (Balle & Wang, 2018) for the Gaussian mechanism.

    Returns the minimal sigma such that adding N(0, sigma^2) noise to a vector-valued
    release with L2 sensitivity GS is (epsilon, delta)-DP.

    Args:
      epsilon: > 0
      delta: in (0,1)
      GS: L2 sensitivity (>= 0)
      tol: numeric tolerance for the internal root find
    """
    eps = float(epsilon)
    delt = float(delta)
    gs = float(GS)

    if eps <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if not (0.0 < delt < 1.0):
        raise ValueError("delta must be in (0,1).")
    if gs < 0.0:
        raise ValueError("GS must be >= 0.")
    if gs == 0.0:
        return 0.0

    delta_thr = _case_a(eps, 0.0)

    if abs(delt - delta_thr) <= 1e-15:
        alpha = 1.0
    elif delt > delta_thr:
        # case A: _case_a increases in s
        pred = lambda s: _case_a(eps, s) >= delt
        s_hi = _doubling_trick(pred, start=0.0, step=1.0)
        s_star = _binary_search(pred, left=0.0, right=s_hi, tol=float(tol))
        alpha = math.sqrt(1.0 + s_star / 2.0) - math.sqrt(s_star / 2.0)
    else:
        # case B: _case_b decreases in s
        pred = lambda s: _case_b(eps, s) <= delt
        s_hi = _doubling_trick(pred, start=0.0, step=1.0)
        s_star = _binary_search(pred, left=0.0, right=s_hi, tol=float(tol))
        alpha = math.sqrt(1.0 + s_star / 2.0) + math.sqrt(s_star / 2.0)

    return float((alpha * gs) / math.sqrt(2.0 * eps))
