# quantbayes/ball_dp/accountants/gaussian.py
from __future__ import annotations

import math
from typing import Callable, Literal


def _phi(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _delta_a(eps: float, s: float) -> float:
    return _phi(math.sqrt(eps * s)) - math.exp(eps) * _phi(-math.sqrt(eps * (s + 2.0)))


def _delta_b(eps: float, s: float) -> float:
    return _phi(-math.sqrt(eps * s)) - math.exp(eps) * _phi(-math.sqrt(eps * (s + 2.0)))


def _doubling_trick(
    predicate: Callable[[float], bool],
    *,
    start: float = 0.0,
    step: float = 1.0,
    max_x: float = 1e12,
) -> float:
    x = float(start)
    delta = float(step)
    while not predicate(x):
        x += delta
        delta *= 2.0
        if x > max_x:
            raise RuntimeError(
                "Failed to bracket root in analytic Gaussian calibration."
            )
    return x


def _bisection(
    predicate: Callable[[float], bool],
    *,
    lo: float,
    hi: float,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> float:
    lo = float(lo)
    hi = float(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if (hi - lo) <= tol * (1.0 + abs(mid)):
            return mid
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate_analytic_gaussian(
    *, epsilon: float, delta: float, sensitivity: float, tol: float = 1e-12
) -> float:
    eps = float(epsilon)
    delt = float(delta)
    sens = float(sensitivity)
    if eps <= 0.0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 < delt < 1.0):
        raise ValueError("delta must be in (0,1)")
    if sens < 0.0:
        raise ValueError("sensitivity must be >= 0")
    if sens == 0.0:
        return 0.0

    delta0 = _delta_a(eps, 0.0)
    if abs(delt - delta0) <= 1e-15:
        alpha = 1.0
    elif delt > delta0:
        pred = lambda s: _delta_a(eps, s) >= delt
        s_hi = _doubling_trick(pred)
        s_star = _bisection(pred, lo=0.0, hi=s_hi, tol=tol)
        alpha = math.sqrt(1.0 + s_star / 2.0) - math.sqrt(s_star / 2.0)
    else:
        pred = lambda s: _delta_b(eps, s) <= delt
        s_hi = _doubling_trick(pred)
        s_star = _bisection(pred, lo=0.0, hi=s_hi, tol=tol)
        alpha = math.sqrt(1.0 + s_star / 2.0) + math.sqrt(s_star / 2.0)
    return float(alpha * sens / math.sqrt(2.0 * eps))


def gaussian_sigma(
    sensitivity: float,
    epsilon: float,
    delta: float,
    *,
    method: Literal["classic", "analytic"] = "classic",
    tol: float = 1e-12,
) -> float:
    sensitivity = float(sensitivity)
    epsilon = float(epsilon)
    delta = float(delta)
    if sensitivity < 0.0:
        raise ValueError("sensitivity must be >= 0")
    if sensitivity == 0.0:
        return 0.0
    if method == "classic":
        return float((sensitivity / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta)))
    if method == "analytic":
        return calibrate_analytic_gaussian(
            epsilon=epsilon, delta=delta, sensitivity=sensitivity, tol=tol
        )
    raise ValueError(method)


def epsilon_from_sigma_classic(
    *, sensitivity: float, sigma: float, delta: float
) -> float:
    sensitivity = float(sensitivity)
    sigma = float(sigma)
    delta = float(delta)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if sensitivity == 0.0:
        return 0.0
    return float((sensitivity / sigma) * math.sqrt(2.0 * math.log(1.25 / delta)))


def epsilon_from_sigma_analytic(
    *, sensitivity: float, sigma: float, delta: float, tol: float = 1e-12
) -> float:
    sensitivity = float(sensitivity)
    sigma = float(sigma)
    delta = float(delta)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if sensitivity == 0.0:
        return 0.0

    # exp(eps) overflows in float64 around eps ~= 709.78.
    # If the required epsilon is beyond this range, it is effectively enormous
    # for reporting purposes, so return +inf instead of crashing.
    MAX_SAFE_EPS = 700.0

    def sigma_required(eps: float) -> float:
        return calibrate_analytic_gaussian(
            epsilon=eps, delta=delta, sensitivity=sensitivity, tol=tol
        )

    lo = 1e-12
    hi = 1.0

    while True:
        if hi > MAX_SAFE_EPS:
            return float("inf")
        req = sigma_required(hi)
        if req <= sigma:
            break
        lo = hi
        hi *= 2.0

    pred = lambda eps: sigma_required(eps) <= sigma
    return float(_bisection(pred, lo=lo, hi=hi, tol=tol))


def epsilon_from_sigma(
    *,
    sensitivity: float,
    sigma: float,
    delta: float,
    method: Literal["classic", "analytic"] = "analytic",
    tol: float = 1e-12,
) -> float:
    if method == "classic":
        return epsilon_from_sigma_classic(
            sensitivity=sensitivity, sigma=sigma, delta=delta
        )
    if method == "analytic":
        return epsilon_from_sigma_analytic(
            sensitivity=sensitivity, sigma=sigma, delta=delta, tol=tol
        )
    raise ValueError(method)
