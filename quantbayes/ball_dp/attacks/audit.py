# quantbayes/ball_dp/attacks/audit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import math
import numpy as np


def gaussian_llr(
    y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, sigma: float
) -> float:
    """
    Log-likelihood ratio for Gaussian mechanism outputs.

    If:
      Y | D0 ~ N(mu0, sigma^2 I)
      Y | D1 ~ N(mu1, sigma^2 I)

    then:
      log p(Y|D0) - log p(Y|D1)
        = (||Y-mu1||^2 - ||Y-mu0||^2) / (2 sigma^2)

    Positive => favor D0.
    """
    sig2 = float(sigma) ** 2
    if sig2 <= 0.0:
        raise ValueError("sigma must be > 0")
    y = np.asarray(y)
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    return float((np.sum((y - mu1) ** 2) - np.sum((y - mu0) ** 2)) / (2.0 * sig2))


def _phi_std(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def gaussian_expected_llr_attack_acc(
    mu0: np.ndarray, mu1: np.ndarray, sigma: float
) -> float:
    """
    Bayes-optimal (LLR) attack accuracy under equal priors for
    N(mu0, sigma^2 I) vs N(mu1, sigma^2 I).
    acc = Phi(||mu0-mu1|| / (2 sigma)).
    """
    d = float(np.linalg.norm(np.asarray(mu0) - np.asarray(mu1)))
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if d == 0.0:
        return 0.5
    return float(_phi_std(d / (2.0 * float(sigma))))


def gaussian_dp_slack_closed_form(
    mu0: np.ndarray, mu1: np.ndarray, sigma: float, eps: float
) -> tuple[float, float]:
    """
    Closed-form directional DP slacks for a fixed neighbor pair (mu0, mu1).

    Returns:
      (delta_D_to_Dprime, delta_Dprime_to_D)

    where
      delta_D_to_Dprime = max(0, P_D[L>eps] - e^eps P_{D'}[L>eps])
      delta_Dprime_to_D = max(0, P_{D'}[L<-eps] - e^eps P_{D}[L<-eps])
    """
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    d = float(np.linalg.norm(mu0 - mu1))
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if d == 0.0:
        return 0.0, 0.0

    s = d / float(sigma)  # std of L is s
    m = 0.5 * s * s  # mean under D is +m, under D' is -m
    exp_eps = math.exp(float(eps))

    # P_D[L > eps] with L ~ N(m, s^2)
    p_D_gt = 1.0 - _phi_std((float(eps) - m) / s)

    # P_D'[L > eps] with L ~ N(-m, s^2)
    p_Dp_gt = 1.0 - _phi_std((float(eps) + m) / s)

    delta_dir = max(0.0, p_D_gt - exp_eps * p_Dp_gt)

    # Reverse-direction event uses L < -eps
    # P_D[L < -eps]  = Phi((-eps - m)/s)
    p_D_lt = _phi_std((-float(eps) - m) / s)

    # P_D'[L < -eps] = Phi((-eps + m)/s)
    p_Dp_lt = _phi_std((-float(eps) + m) / s)

    delta_rev = max(0.0, p_Dp_lt - exp_eps * p_D_lt)

    return float(delta_dir), float(delta_rev)


def llr_attack_predict(
    y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, sigma: float
) -> int:
    """Return 0 if attacker predicts dataset D0, else 1."""
    return 0 if gaussian_llr(y, mu0, mu1, sigma) >= 0.0 else 1


@dataclass(frozen=True)
class LLRAuditResult:
    # attack metric (optimal LRT under equal priors)
    attack_acc: float

    # LLR stats, conditioned on which dataset generated y
    llr_mean_under_D: float
    llr_std_under_D: float
    llr_mean_under_Dprime: float
    llr_std_under_Dprime: float

    # directional tail probabilities for the privacy-loss RV L = log p(y|D)/p(y|D')
    p_L_gt_eps_under_D: float
    p_L_gt_eps_under_Dprime: float

    # estimated DP slack:  max(0, P_D[L>eps] - e^eps P_{D'}[L>eps])
    delta_hat_D_to_Dprime: float

    # reverse-direction tails for L_rev = log p(y|D')/p(y|D) = -L
    # i.e. tails of (L < -eps) under D and D'
    p_L_lt_minus_eps_under_D: float
    p_L_lt_minus_eps_under_Dprime: float

    # estimated reverse DP slack: max(0, P_{D'}[L<-eps] - e^eps P_D[L<-eps])
    delta_hat_Dprime_to_D: float

    n_trials: int
    n_D: int
    n_Dprime: int


def run_llr_audit_trials(
    *,
    f: Callable[[np.ndarray], np.ndarray],
    make_neighbor: Callable[
        [np.ndarray, np.random.Generator],
        Tuple[np.ndarray, int, np.ndarray, np.ndarray],
    ],
    D: np.ndarray,
    sigma: float,
    eps: float,
    n_trials: int = 2000,
    seed: int = 0,
) -> LLRAuditResult:
    """
    Threat-model-aligned auditing for a deterministic query f(D) + N(0, sigma^2 I).

    make_neighbor should return (D_prime, bit, mu0, mu1) where:
      - bit in {0,1} indicates ground truth mechanism run: 0 => D, 1 => D'
      - mu0 = f(D), mu1 = f(D')
    """
    if float(sigma) <= 0.0:
        raise ValueError("sigma must be > 0")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0")
    if int(n_trials) <= 0:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(int(seed))
    eps = float(eps)
    exp_eps = math.exp(eps)

    correct = 0

    # counts by generating dataset
    n_D = 0
    n_Dp = 0

    # store LLRs by generating dataset (for conditional stats)
    llrs_D: list[float] = []
    llrs_Dp: list[float] = []

    # tails for event {L > eps} (same L definition always: log p(y|D)/p(y|D'))
    cnt_D_L_gt = 0
    cnt_Dp_L_gt = 0

    # tails for event {L < -eps} (equivalently, L_rev > eps)
    cnt_D_L_lt = 0
    cnt_Dp_L_lt = 0

    for _ in range(int(n_trials)):
        _Dp, bit, mu0, mu1 = make_neighbor(D, rng)
        if bit not in (0, 1):
            raise ValueError("make_neighbor must return bit in {0,1}")

        mu = mu0 if bit == 0 else mu1
        y = mu + rng.normal(0.0, float(sigma), size=mu.shape).astype(
            mu.dtype, copy=False
        )

        pred = llr_attack_predict(y, mu0, mu1, sigma)
        correct += int(pred == bit)

        L = gaussian_llr(y, mu0, mu1, sigma)  # L = log p(y|D)/p(y|D')

        if bit == 0:
            n_D += 1
            llrs_D.append(L)
            if L > eps:
                cnt_D_L_gt += 1
            if L < -eps:
                cnt_D_L_lt += 1
        else:
            n_Dp += 1
            llrs_Dp.append(L)
            if L > eps:
                cnt_Dp_L_gt += 1
            if L < -eps:
                cnt_Dp_L_lt += 1

    # probabilities (guard against pathological n_D or n_Dp = 0)
    p_D_L_gt = float(cnt_D_L_gt) / float(max(1, n_D))
    p_Dp_L_gt = float(cnt_Dp_L_gt) / float(max(1, n_Dp))
    p_D_L_lt = float(cnt_D_L_lt) / float(max(1, n_D))
    p_Dp_L_lt = float(cnt_Dp_L_lt) / float(max(1, n_Dp))

    # DP slack estimators
    delta_hat = max(0.0, p_D_L_gt - exp_eps * p_Dp_L_gt)
    delta_hat_rev = max(0.0, p_Dp_L_lt - exp_eps * p_D_L_lt)

    # conditional stats
    llrs_D_arr = np.asarray(llrs_D, dtype=np.float64) if llrs_D else np.asarray([0.0])
    llrs_Dp_arr = (
        np.asarray(llrs_Dp, dtype=np.float64) if llrs_Dp else np.asarray([0.0])
    )

    return LLRAuditResult(
        attack_acc=float(correct) / float(n_trials),
        llr_mean_under_D=float(llrs_D_arr.mean()),
        llr_std_under_D=float(llrs_D_arr.std()),
        llr_mean_under_Dprime=float(llrs_Dp_arr.mean()),
        llr_std_under_Dprime=float(llrs_Dp_arr.std()),
        p_L_gt_eps_under_D=p_D_L_gt,
        p_L_gt_eps_under_Dprime=p_Dp_L_gt,
        delta_hat_D_to_Dprime=float(delta_hat),
        p_L_lt_minus_eps_under_D=p_D_L_lt,
        p_L_lt_minus_eps_under_Dprime=p_Dp_L_lt,
        delta_hat_Dprime_to_D=float(delta_hat_rev),
        n_trials=int(n_trials),
        n_D=int(n_D),
        n_Dprime=int(n_Dp),
    )
