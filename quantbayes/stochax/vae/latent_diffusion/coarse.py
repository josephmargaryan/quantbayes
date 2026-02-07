# quantbayes/stochax/vae/latent_diffusion/coarse.py
from __future__ import annotations

from typing import Tuple
import jax
import jax.numpy as jnp


def ink_fraction_and_grad_01(
    x01: jnp.ndarray,
    *,
    thr: float = 0.35,
    temp: float = 0.08,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Ink fraction on images in [0,1]. x01: (B,1,H,W) or (1,H,W).
    Returns:
      d: (B,)
      grad_x: same shape as x01
    """
    x = jnp.asarray(x01)
    if x.ndim == 3:
        x = x[None, ...]

    temp = jnp.maximum(jnp.asarray(temp, x.dtype), eps)
    u = (x - thr) / temp
    s = jax.nn.sigmoid(u)  # (B,1,H,W)

    d = jnp.mean(s, axis=(1, 2, 3))  # (B,)

    n_pix = x.shape[1] * x.shape[2] * x.shape[3]
    grad = s * (1.0 - s) * (1.0 / temp) * (1.0 / n_pix)

    return d, grad


def ink_fraction_01(
    x01: jnp.ndarray, *, thr: float = 0.35, temp: float = 0.08
) -> jnp.ndarray:
    d, _ = ink_fraction_and_grad_01(x01, thr=thr, temp=temp)
    return d
