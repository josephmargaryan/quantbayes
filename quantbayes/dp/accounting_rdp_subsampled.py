# quantbayes/dp/accounting_rdp_subsampled.py
from __future__ import annotations
import math
from typing import Sequence, Tuple
import numpy as np
from math import lgamma


def _logbinom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _log_expm1_stable(x: float) -> float:
    """
    Stable log(expm1(x)) across regimes:
      - x ≈ 0:  expm1(x) ~ x        -> log(expm1(x)) ~ log(x)
      - x small/moderate: use log(expm1(x))
      - x large:          expm1(x) ~ exp(x) -> log(expm1(x)) ~ x
    """
    if x < 1e-8:
        return math.log(x + 1e-18)  # linear regime, avoid log(0)
    if x < 30.0:
        return math.log(math.expm1(x))  # safe here
    # very large: expm1(x) ~ exp(x)
    return x


def rdp_subsampled_gaussian_integer_alpha(alpha: int, q: float, sigma: float) -> float:
    """
    RDP (order alpha, integer >= 2) for Poisson-subsampled Gaussian with sampling rate q
    and noise stddev sigma. Uses the binomial-sum bound:

      RDP(alpha) = (1/(alpha-1)) * log( 1 + sum_{j=2}^alpha C(alpha, j) q^j (1-q)^{alpha-j}
                                            * (exp( j*(j-1)/(2 sigma^2) ) - 1) )

    Computed fully in log-domain with stable approximations.
    """
    if alpha < 2:
        raise ValueError("alpha must be integer >= 2")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    log_q = math.log(q)
    log_1mq = math.log(1.0 - q)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    log_terms = []
    for j in range(2, alpha + 1):
        logC = _logbinom(alpha, j)
        exp_arg = j * (j - 1) * inv_two_sigma2
        # term: C(alpha,j) * q^j * (1-q)^(alpha-j) * (exp(exp_arg) - 1)
        # log term = logC + j log q + (alpha-j) log(1-q) + log(expm1(exp_arg))
        log_expm1 = _log_expm1_stable(exp_arg)
        log_t = logC + j * log_q + (alpha - j) * log_1mq + log_expm1
        log_terms.append(log_t)

    if not log_terms:
        return 0.0

    # We need log( 1 + sum_j exp(log_terms_j) ) stably.
    # Do a log-sum-exp including the "+1" as exp(0).
    m = max(0.0, max(log_terms))  # include 0 for the "1" term
    sumexp = math.exp(0.0 - m)  # contribution of the "+1" term
    sumexp += sum(math.exp(t - m) for t in log_terms)
    log_one_plus_sum = m + math.log(sumexp)
    return log_one_plus_sum / (alpha - 1.0)


def rdp_subsampled_gaussian(
    q: float,
    sigma: float,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-step RDP for a list of orders. Only integer orders are supported here.
    Returns (orders_int_array, rdp_values_array).
    """
    orders_int = []
    rdp_vals = []
    for a in orders:
        if abs(a - round(a)) < 1e-9 and int(round(a)) >= 2:
            ai = int(round(a))
            orders_int.append(ai)
            rdp_vals.append(rdp_subsampled_gaussian_integer_alpha(ai, q, sigma))
        # ignore non-integer orders in this simple implementation
    if not orders_int:
        raise ValueError(
            "No valid integer orders provided. Use e.g. orders=(2,3,4,5,8,16,32,64,128)."
        )
    return np.array(orders_int, dtype=float), np.array(rdp_vals, dtype=float)


def compose_rdp_steps(rdp_per_step: np.ndarray, T: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be >= 1")
    return rdp_per_step * float(T)


def rdp_to_eps(
    rdp_total: np.ndarray, orders: np.ndarray, delta: float
) -> Tuple[float, float]:
    """
    Convert total RDP (vector over orders) to the tightest eps for given delta:
      eps(α) = rdp_total(α) + log(1/delta)/(α - 1)
    Returns (eps_min, alpha_star).
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    penalties = np.log(1.0 / delta) / (orders - 1.0)
    eps_all = rdp_total + penalties
    idx = int(np.argmin(eps_all))
    return float(eps_all[idx]), float(orders[idx])


def eps_from_sigma_subsampled_rdp(
    sigma: float,
    q: float,
    T: int,
    delta: float,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
) -> Tuple[float, float]:
    """
    For given sigma, return the tightest eps (and its order) at Poisson rate q over T steps.
    """
    ords, rdp_step = rdp_subsampled_gaussian(q=q, sigma=sigma, orders=orders)
    rdp_tot = compose_rdp_steps(rdp_step, T)
    return rdp_to_eps(rdp_tot, ords, delta)


def sigma_for_target_eps_subsampled_rdp(
    eps: float,
    q: float,
    T: int,
    delta: float,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
    tol: float = 1e-4,
    sigma_lo: float = 1e-3,
    sigma_hi: float = 50.0,
    max_iter: int = 60,
) -> Tuple[float, float]:
    """
    Calibrate sigma by binary search to meet target (eps, delta) under Poisson subsampling.
    Returns (sigma_min, alpha_star).
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1)")
    if T <= 0:
        raise ValueError("T must be >= 1")

    # Helper for feasibility at given sigma
    def feasible(s):
        e, _ = eps_from_sigma_subsampled_rdp(
            sigma=s, q=q, T=T, delta=delta, orders=orders
        )
        return e <= eps

    # Expand upper bound until feasible
    while not feasible(sigma_hi):
        sigma_hi *= 2.0
        if sigma_hi > 1e6:
            raise ValueError(
                "Could not find feasible sigma_hi; try increasing eps or delta."
            )

    lo, hi = sigma_lo, sigma_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e_mid, _ = eps_from_sigma_subsampled_rdp(
            sigma=mid, q=q, T=T, delta=delta, orders=orders
        )
        if e_mid <= eps:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    eps_final, alpha_star = eps_from_sigma_subsampled_rdp(
        sigma=hi, q=q, T=T, delta=delta, orders=orders
    )
    return float(hi), float(alpha_star)
