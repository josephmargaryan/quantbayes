# quantbayes/pkdiffusion/f_divergences.py
from __future__ import annotations

import numpy as np


def hellinger2_gaussian_nd(
    m0: np.ndarray, C0: np.ndarray, m1: np.ndarray, C1: np.ndarray
) -> float:
    """
    Squared Hellinger distance between Gaussians:
      H^2(P,Q) = 1 - BC(P,Q)
    """
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    m1 = np.asarray(m1, dtype=float).reshape(-1)
    C0 = np.asarray(C0, dtype=float)
    C1 = np.asarray(C1, dtype=float)

    Cbar = 0.5 * (C0 + C1)

    sign0, logdet0 = np.linalg.slogdet(C0)
    sign1, logdet1 = np.linalg.slogdet(C1)
    signb, logdetb = np.linalg.slogdet(Cbar)
    if sign0 <= 0 or sign1 <= 0 or signb <= 0:
        raise ValueError("Covariance matrices must be SPD for Hellinger distance.")

    dm = (m0 - m1).reshape(-1, 1)
    L = np.linalg.cholesky(Cbar)
    y = np.linalg.solve(L, dm)
    quad = float((y.T @ y).squeeze())

    logBC = 0.25 * (logdet0 + logdet1) - 0.5 * logdetb - 0.125 * quad
    BC = float(np.exp(logBC))
    return float(max(1.0 - BC, 0.0))


def js_divergence_mc(
    *, sample_p: np.ndarray, sample_q: np.ndarray, logp_fn, logq_fn
) -> float:
    """
    Jensen-Shannon divergence (nats) via Monte Carlo:
      JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M),  M = 0.5(P+Q)
    This is bounded and numerically stable.
    """
    x_p = np.asarray(sample_p, dtype=float)
    x_q = np.asarray(sample_q, dtype=float)

    lp_p = np.asarray(logp_fn(x_p), dtype=float).reshape(-1)
    lq_p = np.asarray(logq_fn(x_p), dtype=float).reshape(-1)
    logm_p = np.logaddexp(lp_p, lq_p) - np.log(2.0)
    kl_p_m = float(np.mean(lp_p - logm_p))

    lp_q = np.asarray(logp_fn(x_q), dtype=float).reshape(-1)
    lq_q = np.asarray(logq_fn(x_q), dtype=float).reshape(-1)
    logm_q = np.logaddexp(lp_q, lq_q) - np.log(2.0)
    kl_q_m = float(np.mean(lq_q - logm_q))

    return 0.5 * (kl_p_m + kl_q_m)
