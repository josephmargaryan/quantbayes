# quantbayes/pkdiffusion/f_divergences.py
from __future__ import annotations

import numpy as np


def hellinger2_gaussian_nd(
    m0: np.ndarray, C0: np.ndarray, m1: np.ndarray, C1: np.ndarray
) -> float:
    """
    Squared Hellinger distance between Gaussians:
      H^2(P,Q) = 1 - BC(P,Q)
    where Bhattacharyya coefficient:
      BC = |C0|^{1/4}|C1|^{1/4} / |(C0+C1)/2|^{1/2}
           * exp(-1/8 (m0-m1)^T ((C0+C1)/2)^{-1} (m0-m1))
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
    H2 = float(max(1.0 - BC, 0.0))
    return H2


def chi2_divergence_mc_gaussian(
    *,
    sample_q: np.ndarray,
    logp_fn,
    logq_fn,
) -> float:
    """
    Monte Carlo estimate of chi-square divergence:
      chi2(P||Q) = E_{x~Q}[(p/q - 1)^2]
    using samples x~Q and access to log densities.
    """
    x = np.asarray(sample_q)
    lp = np.asarray(logp_fn(x), dtype=float).reshape(-1)
    lq = np.asarray(logq_fn(x), dtype=float).reshape(-1)

    lr = lp - lq
    lr = np.clip(lr, -50.0, 50.0)  # stabilize
    r = np.exp(lr)
    return float(np.mean((r - 1.0) ** 2))
