# quantbayes/ball_dp/attacks/audit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
    if sig2 <= 0:
        raise ValueError("sigma must be > 0")
    y = np.asarray(y)
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    return float((np.sum((y - mu1) ** 2) - np.sum((y - mu0) ** 2)) / (2.0 * sig2))


def llr_attack_predict(
    y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, sigma: float
) -> int:
    """
    Return 0 if attacker predicts dataset D0, else 1.
    """
    return 0 if gaussian_llr(y, mu0, mu1, sigma) >= 0.0 else 1


@dataclass(frozen=True)
class LLRAuditResult:
    attack_acc: float
    llr_mean: float
    llr_std: float
    frac_llr_gt_eps: float
    n_trials: int


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

    Inputs
    ------
    f:
        Deterministic function on datasets (e.g., prototypes, ERM params).
    make_neighbor:
        Function that, given D and RNG, returns:
          (D_prime, bit, mu0, mu1)
        where:
          - bit in {0,1} is the ground truth: 0 => mechanism run on D, 1 => on D'
          - mu0 = f(D), mu1 = f(D')
    D:
        Dataset in the representation space used by f (e.g., embeddings per class aggregated).
    sigma:
        Noise std.
    eps:
        The DP epsilon to test exceedance rates for the privacy loss random variable.
    n_trials:
        Monte Carlo trials.
    """
    rng = np.random.default_rng(seed)
    llrs = []
    correct = 0

    muD = f(D)

    for _ in range(int(n_trials)):
        Dp, bit, mu0, mu1 = make_neighbor(D, rng)
        assert bit in (0, 1)

        if bit == 0:
            mu = mu0
        else:
            mu = mu1

        y = mu + rng.normal(0.0, float(sigma), size=mu.shape).astype(
            mu.dtype, copy=False
        )

        pred = llr_attack_predict(y, mu0, mu1, sigma)
        correct += int(pred == bit)

        llr = gaussian_llr(y, mu0, mu1, sigma)  # log p(y|D) - log p(y|D')
        llrs.append(llr)

    llrs = np.asarray(llrs, dtype=np.float64)
    return LLRAuditResult(
        attack_acc=float(correct) / float(n_trials),
        llr_mean=float(llrs.mean()),
        llr_std=float(llrs.std()),
        frac_llr_gt_eps=float((llrs > float(eps)).mean()),
        n_trials=int(n_trials),
    )
