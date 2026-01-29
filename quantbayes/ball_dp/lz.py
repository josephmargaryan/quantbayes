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


def lz_logistic_binary_bound(*, B: float, lam: float) -> float:
    """
    A clean bound derived in your notes:

      L_z <= 1 + (B^2)/(4*lam)

    where:
      - ||z|| <= B (enforced e.g. via L2 normalization or clipping),
      - lam is L2 regularization coefficient in (lam/2)||theta||^2.

    This uses the fact ||theta_hat|| <= B/lam at the optimum (standard for logistic + L2 reg).
    """
    if B <= 0:
        raise ValueError("B must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    return 1.0 + (float(B) * float(B)) / (4.0 * float(lam))


def lz_softmax_linear_bound(*, B: float, lam: float) -> float:
    """
    Conservative bound for K-class softmax linear classifier W in R^{Kxd} with CE loss.

    A usable (conservative) form:
      L_z <= sqrt(2) + (B * ||W||_2)/2

    and at the L2-regularized optimum:
      ||W||_F <= (sqrt(2)*B)/lam  =>  ||W||_2 <= ||W||_F <= (sqrt(2)*B)/lam

    giving:
      L_z <= sqrt(2) + B * (sqrt(2)*B/lam)/2
           = sqrt(2) * (1 + B^2/(2*lam))

    This is intentionally conservative but clean and monotone in (B, lam).
    """
    if B <= 0:
        raise ValueError("B must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    return math.sqrt(2.0) * (1.0 + (float(B) * float(B)) / (2.0 * float(lam)))


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
