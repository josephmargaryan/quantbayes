from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import betaln, digamma


def kl_gaussian_1d(m0: float, v0: float, m1: float, v1: float) -> float:
    """KL(N(m0,v0) || N(m1,v1))."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError("variances must be > 0")
    return 0.5 * (v0 / v1 + (m1 - m0) ** 2 / v1 - 1.0 + np.log(v1 / v0))


def kl_gaussian_nd(
    m0: np.ndarray, C0: np.ndarray, m1: np.ndarray, C1: np.ndarray
) -> float:
    """KL(N(m0,C0) || N(m1,C1))."""
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    m1 = np.asarray(m1, dtype=float).reshape(-1)
    C0 = np.asarray(C0, dtype=float)
    C1 = np.asarray(C1, dtype=float)
    d = m0.shape[0]

    # Solve C1^{-1} * C0 and C1^{-1}*(m1-m0) stably via solves
    L = np.linalg.cholesky(C1)
    # inv(C1)C0 = solve(L.T, solve(L, C0))
    X = np.linalg.solve(L, C0)
    invC1_C0 = np.linalg.solve(L.T, X)
    tr_term = float(np.trace(invC1_C0))

    dm = (m1 - m0).reshape(-1, 1)
    y = np.linalg.solve(L, dm)
    quad_term = float((y.T @ y).squeeze())

    logdet_C0 = float(np.linalg.slogdet(C0)[1])
    logdet_C1 = float(np.linalg.slogdet(C1)[1])
    return 0.5 * (tr_term + quad_term - d + (logdet_C1 - logdet_C0))


def _spd_sqrt(C: np.ndarray) -> np.ndarray:
    """Matrix square-root for SPD matrices via eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[None, :]) @ V.T


def w2_gaussian_nd(
    m0: np.ndarray, C0: np.ndarray, m1: np.ndarray, C1: np.ndarray
) -> float:
    """
    2-Wasserstein distance between Gaussians.
      W2^2 = ||m0-m1||^2 + tr(C0 + C1 - 2*(C1^{1/2} C0 C1^{1/2})^{1/2})
    """
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    m1 = np.asarray(m1, dtype=float).reshape(-1)
    C0 = np.asarray(C0, dtype=float)
    C1 = np.asarray(C1, dtype=float)

    dm2 = float(np.sum((m0 - m1) ** 2))
    S1 = _spd_sqrt(C1)
    M = S1 @ C0 @ S1
    SM = _spd_sqrt(M)
    tr_term = float(np.trace(C0 + C1 - 2.0 * SM))
    return float(np.sqrt(max(dm2 + tr_term, 0.0)))


def w2_empirical_1d(x: np.ndarray, y: np.ndarray) -> float:
    """W2 for equal-weight 1D empirical measures."""
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    y = np.sort(np.asarray(y, dtype=float).reshape(-1))
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    return float(np.sqrt(np.mean((x - y) ** 2)))


def w2_empirical_2d_hungarian(x: np.ndarray, y: np.ndarray) -> float:
    """
    W2 for equal-weight empirical measures in R^2 using Hungarian assignment.
    O(n^3) — keep n ~ 200–800.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    # Cost matrix: squared distances
    C = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)  # (n,n)
    row, col = linear_sum_assignment(C)
    return float(np.sqrt(np.mean(C[row, col])))


def kl_beta(a: float, b: float, c: float, d: float) -> float:
    """
    KL(Beta(a,b) || Beta(c,d)).

    Formula:
      KL = log B(c,d) - log B(a,b)
            + (a-c) ψ(a) + (b-d) ψ(b) + (c+d-a-b) ψ(a+b)
    """
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    if min(a, b, c, d) <= 0:
        raise ValueError("Beta parameters must be > 0")

    return float(
        (betaln(c, d) - betaln(a, b))
        + (a - c) * digamma(a)
        + (b - d) * digamma(b)
        + (c + d - a - b) * digamma(a + b)
    )
