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

    Model: logits = W e + b, p = softmax(logits), loss = -log p_y.
    If we include a bias b, we can write this as an augmented linear map:
        \tilde e = (e, 1),   \tilde W = [W  b]
    and regularize \tilde W with (lam/2)||\tilde W||_F^2 (i.e., bias is regularized too).

    Assumptions:
      - ||e||_2 <= B.
      - If include_bias=True, then ||\tilde e||_2 <= sqrt(B^2 + 1) =: \tilde B.

    Derivation sketch (matches your LaTeX):
      ||a(e)||_2 = ||softmax(\tilde W \tilde e) - onehot(y)||_2 <= sqrt(2)
      ||J_softmax||_op <= 1/2
      => ||∇_W ℓ(e) - ∇_W ℓ(e')||_F <= (sqrt(2) + (||\tilde W||_2 * \tilde B)/2) ||e - e'||_2

    For the L2-regularized ERM minimizer \tilde W*:
      ||\tilde W*||_2 <= ||\tilde W*||_F <= sqrt(2) * \tilde B / lam

    Plugging in yields:
      L_z <= sqrt(2) * (1 + \tilde B^2 / (2 lam))
          = sqrt(2) * (1 + (B^2 + 1)/(2 lam))  if include_bias=True.
    """
    B = float(B)
    lam = float(lam)
    if B <= 0:
        raise ValueError("B must be > 0.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")

    Btilde_sq = B * B + (1.0 if include_bias else 0.0)
    return math.sqrt(2.0) * (1.0 + Btilde_sq / (2.0 * lam))


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
