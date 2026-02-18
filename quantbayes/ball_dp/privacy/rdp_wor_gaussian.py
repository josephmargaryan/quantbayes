# quantbayes/ball_dp/privacy/rdp_wor_gaussian.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def _logsumexp(log_terms: Sequence[float]) -> float:
    if len(log_terms) == 0:
        return float("-inf")
    m = max(log_terms)
    if not math.isfinite(m):
        return m
    s = sum(math.exp(t - m) for t in log_terms)
    return m + math.log(s + 1e-300)


def rdp_step_wor_subsampled_gaussian(
    *,
    alpha: int,
    q: float,
    u: float,
) -> float:
    """
    Fixed-size subsampling without replacement amplification bound specialized to
    a Gaussian base mechanism where:
        eps_base(j) = j/2 * u,    u = (Δ/σ)^2.

    This matches your thesis corollary where Gaussian => eps(∞)=∞ => coefficients simplify.

    Returns:
        epsilon_step(alpha)   (RDP at order alpha for ONE step)
    """
    if alpha < 2:
        return float("inf")
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1].")
    u = float(u)
    if u < 0:
        raise ValueError("u must be >= 0.")

    logq = math.log(max(q, 1e-300))

    # c2 = min{ 4(e^u - 1), 2 e^u }
    # compute in log-space robustly
    log_c2_a = math.log(4.0) + math.log(max(math.expm1(u), 1e-300))
    log_c2_b = math.log(2.0) + u
    log_c2 = min(log_c2_a, log_c2_b)

    logs = []

    # j=2 term
    logs.append(_log_binom(alpha, 2) + 2.0 * logq + log_c2)

    # j>=3 terms: cj = 2 * exp(j(j-1)/2 * u)
    for j in range(3, alpha + 1):
        log_cj = math.log(2.0) + 0.5 * float(j) * float(j - 1) * u
        logs.append(_log_binom(alpha, j) + float(j) * logq + log_cj)

    log_sum = _logsumexp(logs)  # log( sum_j q^j C(alpha,j) c_j )
    # eps_step = (1/(alpha-1)) * log( 1 + sum )
    if not math.isfinite(log_sum):
        return float("inf")

    # stable log(1 + exp(log_sum))
    if log_sum > 0:
        log_1p_sum = log_sum + math.log1p(math.exp(-log_sum))
    else:
        log_1p_sum = math.log1p(math.exp(log_sum))

    return log_1p_sum / float(alpha - 1)


@dataclass
class RDPAccountantWOR:
    """
    Tracks total RDP across steps for the fixed-size subsampling W/O replacement bound.

    You supply per-step (q, u) where u=(Δ/σ)^2 for that step.
    """

    orders: Tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 256])
    _rdp: Array = None

    def __post_init__(self):
        self.orders = tuple(int(o) for o in self.orders if int(o) >= 2)
        self._rdp = np.zeros((len(self.orders),), dtype=np.float64)

    def reset(self) -> None:
        self._rdp[:] = 0.0

    def accumulate(self, *, steps: int, q: float, u: float) -> None:
        steps = int(steps)
        if steps <= 0:
            return
        per = np.array(
            [rdp_step_wor_subsampled_gaussian(alpha=a, q=q, u=u) for a in self.orders],
            dtype=np.float64,
        )
        self._rdp += float(steps) * per

    def epsilon(self, *, delta: float) -> float:
        delta = float(delta)
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1).")
        eps = self._rdp + math.log(1.0 / delta) / (
            np.array(self.orders, dtype=np.float64) - 1.0
        )
        return float(np.min(eps))


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    q: float,
    steps: int,
    orders: Iterable[int] = tuple(list(range(2, 65)) + [80, 96, 128, 256]),
    tol: float = 1e-3,
    max_iter: int = 60,
    nm_lo: float = 0.1,
    nm_hi: float = 50.0,
) -> float:
    """
    Binary-search a noise multiplier nm so that epsilon(delta) <= target_epsilon.
    Here nm is defined relative to sensitivity Δ:
        σ = nm * Δ   =>  u=(Δ/σ)^2 = 1/nm^2.

    This is the natural definition for Ball-DP-SGD where Δ is (Lz*r) for the *sum*,
    or (Lz*r/b) for the mean; u is invariant to the b scaling as long as σ scales with Δ.
    """
    target_epsilon = float(target_epsilon)
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0.")
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be > 0.")

    orders = tuple(int(o) for o in orders if int(o) >= 2)
    acc = RDPAccountantWOR(orders=orders)

    lo, hi = float(nm_lo), float(nm_hi)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        u = 1.0 / (mid * mid)
        acc.reset()
        acc.accumulate(steps=steps, q=float(q), u=float(u))
        eps_mid = acc.epsilon(delta=float(delta))

        if eps_mid <= target_epsilon:
            hi = mid
        else:
            lo = mid

        if (hi - lo) <= tol * (1.0 + mid):
            break

    return float(hi)
