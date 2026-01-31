# quantbayes/ball_dp/lz.py
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np


def lz_prototypes_exact() -> float:
    """
    For squared prototype loss: l(mu; z, y)=||z-mu_y||^2, the gradient in mu is
    Lipschitz in z with constant exactly 2.
    """
    return 2.0


def lz_logistic_binary_bound(
    *, B: float, lam: float, include_bias: bool = True
) -> float:
    """
    Conservative bound for binary logistic regression gradient Lipschitz-in-data constant L_z.

    Without bias (w only):   L_z <= 1 + B^2/(4 lam)
    With bias via augmentation \tilde e=(e,1): replace B^2 by (B^2+1).

    Assumes L2 regularization makes the ERM strongly convex with parameter lam
    over the FULL parameter vector (including bias if include_bias=True).
    """
    B = float(B)
    lam = float(lam)
    if B <= 0:
        raise ValueError("B must be > 0.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    Btilde_sq = B * B + (1.0 if include_bias else 0.0)
    return 1.0 + Btilde_sq / (4.0 * lam)


def lz_softmax_linear_bound(
    *, B: float, lam: float, include_bias: bool = True
) -> float:
    """
    Conservative L_z bound for K-class softmax linear head (optionally with bias).

    Assumes:
      - ||e||_2 <= B
      - If include_bias=True, work with augmented embedding \tilde e=(e,1)
        so ||\tilde e||^2 <= B^2 + 1.
      - L2 regularization over the FULL augmented weight matrix (bias included)
        gives strong convexity parameter lam.

    Bound (matches your draft):
      L_z <= sqrt(2) * (1 + \tilde B^2 / (2 lam)),
      where \tilde B^2 = B^2 + 1 if bias is included.
    """
    B = float(B)
    lam = float(lam)
    if B <= 0:
        raise ValueError("B must be > 0.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")

    Btilde_sq = B * B + (1.0 if include_bias else 0.0)
    return math.sqrt(2.0) * (1.0 + Btilde_sq / (2.0 * lam))


def lz_glm_data_independent_bound(
    *, A: float, L: float, B: float, lam: float, include_bias: bool = True
) -> float:
    """
    Data-independent GLM-style bound corresponding to Remark in the paper.

    For loss l(theta; e,y)=phi_y(theta^T e), if:
      |phi'(t)| <= A,   |phi''(t)| <= L,   ||e|| <= B,
    and L2-regularized ERM has strong convexity lam over FULL parameter vector,
    then at optimum ||theta*|| <= (A * Btilde)/lam, where Btilde^2 = B^2 + 1 if bias is included.

    Plugging into L_z <= A + L Btilde ||theta*|| yields:
      L_z <= A (1 + L * Btilde^2 / lam).
    """
    A = float(A)
    L = float(L)
    B = float(B)
    lam = float(lam)

    if A < 0 or L < 0:
        raise ValueError("A and L must be >= 0")
    if B <= 0:
        raise ValueError("B must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")

    Btilde_sq = B * B + (1.0 if include_bias else 0.0)
    return A * (1.0 + (L * Btilde_sq) / lam)


def lz_squared_hinge_bound(*, B: float, lam: float, include_bias: bool = True) -> float:
    """
    Data-independent L_z for squared hinge head (paper Section "squared hinge").

    From the paper:
      L_z <= 2 + 4 B R, where ||w|| <= R.
    Under L2-regularized ERM:
      ||w*|| <= sqrt(2/lam).

    With bias via augmentation, replace B by Btilde = sqrt(B^2+1).

    So:
      L_z <= 2 + 4 * Btilde * sqrt(2/lam).
    """
    B = float(B)
    lam = float(lam)
    if B <= 0:
        raise ValueError("B must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")

    Btilde = math.sqrt(B * B + (1.0 if include_bias else 0.0))
    R = math.sqrt(2.0 / lam)
    return 2.0 + 4.0 * Btilde * R


def estimate_lz_empirical_torch(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    *,
    pairs: Tuple[np.ndarray, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    eps: float = 1e-12,
) -> float:
    """
    Empirically estimate L_z from sampled pairs (z, z'):

      Lz_hat = max ||g(z) - g(z')|| / (d(z,z') + eps)

    where grad_fn returns the per-example gradient w.r.t parameters at a fixed theta.

    Notes:
    - This is NOT a worst-case certificate; it’s an empirical diagnostic.
    - Useful to show that Lipschitz training / normalization shrinks L_z in practice.
    """
    Z, Zp = pairs
    assert Z.shape == Zp.shape and Z.ndim == 2, "expected (N,d) arrays"

    max_ratio = 0.0
    for i in range(Z.shape[0]):
        z = Z[i]
        zp = Zp[i]
        d = float(metric(z, zp))
        g = grad_fn(z)
        gp = grad_fn(zp)
        num = float(np.linalg.norm(g - gp))
        den = d + float(eps)
        max_ratio = max(max_ratio, num / den)
    return float(max_ratio)
