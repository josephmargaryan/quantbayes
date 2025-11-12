# quantbayes/dp/optim/dp_sgd_rdp.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Sequence
from ..accounting_rdp_subsampled import (
    sigma_for_target_eps_subsampled_rdp,
    eps_from_sigma_subsampled_rdp,
)


def _logistic_grad_per_example(
    w: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Per-example gradient of logistic loss: g_i = -(y_i x_i) * sigmoid(-y_i w^T x_i).
    Returns array of shape (n, d).
    """
    z = y * (X @ w)  # shape (n,)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    # sigmoid(-z) stably
    out[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = 1.0 / (1.0 + ez)
    g = -(y[:, None] * X) * out[:, None]
    return g  # (n, d)


def _poisson_sample_mask(n: int, q: float, rng: np.random.RandomState) -> np.ndarray:
    return rng.rand(n) < q


def dp_sgd_rdp_logreg(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    eps: float,
    delta: float,
    steps: int = 2000,
    lr: float = 0.05,
    clip_norm: float = 1.0,
    sample_rate: float = 0.1,
    seed: Optional[int] = None,
    orders: Sequence[float] = (2, 3, 4, 5, 8, 16, 32, 64, 128),
) -> Tuple[np.ndarray, float, float]:
    """
    DP-SGD for L2-regularized logistic regression using Poisson subsampling and RDP accounting.

    - Per-example grads, clip to clip_norm (L2).
    - Poisson subsampling with rate q = sample_rate.
    - Noise added to the SUM of clipped grads: Normal(0, (sigma*clip_norm)^2 I_d).
    - Update uses the noisy average: (g_sum + noise) / max(1, |batch|).
    - sigma is calibrated to meet target (eps, delta) over 'steps' iterations via RDP.

    Returns (w_priv, sigma_used, eps_achieved_for_sigma_used).
    """
    if not (0 < sample_rate < 1):
        raise ValueError("sample_rate must be in (0,1)")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    n, d = X.shape
    rng = np.random.RandomState(seed)

    # Calibrate sigma via subsampled-RDP accountant (per-step rate q, total steps)
    sigma, alpha_star = sigma_for_target_eps_subsampled_rdp(
        eps=eps, q=sample_rate, T=steps, delta=delta, orders=orders
    )

    w = np.zeros(d, dtype=float)
    for _ in range(steps):
        # Poisson subsample
        mask = _poisson_sample_mask(n, sample_rate, rng)
        idx = np.where(mask)[0]
        denom = max(1, idx.size)
        if idx.size == 0:
            g_sum = np.zeros(d, dtype=float)
        else:
            Xb, yb = X[idx], y[idx]
            g_i = _logistic_grad_per_example(w, Xb, yb)  # (b, d)
            norms = np.linalg.norm(g_i, axis=1, keepdims=True) + 1e-12
            scale = np.minimum(1.0, clip_norm / norms)
            g_clipped = g_i * scale
            g_sum = g_clipped.sum(axis=0)

        noise = rng.normal(0.0, sigma * clip_norm, size=d)
        noisy_avg = (g_sum + noise) / float(denom)
        g_step = noisy_avg + lam * w
        w = w - lr * g_step

    # Report achieved eps for sigma_used (min over orders)
    eps_achieved, _ = eps_from_sigma_subsampled_rdp(
        sigma=sigma, q=sample_rate, T=steps, delta=delta, orders=orders
    )
    return w, float(sigma), float(eps_achieved)
