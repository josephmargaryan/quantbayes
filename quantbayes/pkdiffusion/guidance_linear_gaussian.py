# quantbayes/pkdiffusion/guidance_linear_gaussian.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.parameterizations import vp_alpha_sigma

Array = jax.Array


@dataclass(frozen=True)
class LinearGaussianRRGuidanceConfig:
    """
    Evidence on y = a^T x0:
      log ratio(y) = log q(y) - log p_Y(y)
    where q and p_Y are 1D Gaussians.

    We apply DPS-style guidance using x0_hat from Tweedie:
      x0_hat = (x_t + sigma(t)^2 * score_t) / alpha(t)
    and approximate:
      ∇_{x_t} log ratio(a^T x0) ≈ (1/alpha(t)) ∇_{x0} log ratio(a^T x0_hat).
    """

    a: Array  # (d,)
    q_mean: float
    q_var: float
    py_mean: float
    py_var: float

    guidance_scale: float = 1.0

    eps: float = 1e-6
    min_alpha: float = 0.05
    max_guidance_norm: float = 25.0
    snr_gamma: float = 1.0
    noise_aware: bool = True


def make_linear_gaussian_rr_guidance(cfg: LinearGaussianRRGuidanceConfig) -> Callable:
    a_vec = jnp.asarray(cfg.a).reshape(-1)

    q_mean = float(cfg.q_mean)
    q_var = float(cfg.q_var)
    py_mean = float(cfg.py_mean)
    py_var = float(cfg.py_var)

    if q_var <= 0 or py_var <= 0:
        raise ValueError("q_var and py_var must be > 0")

    def dlogratio_dy(y: Array) -> Array:
        # d/dy log N(y|m,v) = -(y-m)/v
        return (-(y - q_mean) / q_var) - (-(y - py_mean) / py_var)

    def guidance(
        t: Array,
        x_t: Array,  # (d,)
        score_t: Array,  # (d,)
        *,
        int_beta_fn,
    ) -> Array:
        a_t, s_t = vp_alpha_sigma(int_beta_fn, t)
        a_t = jnp.asarray(a_t, dtype=x_t.dtype)
        s_t = jnp.asarray(s_t, dtype=x_t.dtype)

        gate = (a_t >= cfg.min_alpha).astype(x_t.dtype)
        a_safe = jnp.maximum(a_t, cfg.eps)
        s2 = s_t * s_t

        # Tweedie x0-hat
        x0_hat = (x_t + s2 * score_t) / a_safe

        # y-hat and gradient
        y_hat = jnp.dot(a_vec, x0_hat)
        sigma_x0_sq = s2 / (a_safe * a_safe + cfg.eps)  # scalar

        if cfg.noise_aware:
            q_var_eff = jnp.asarray(cfg.q_var, dtype=x_t.dtype) + sigma_x0_sq
            py_var_eff = jnp.asarray(cfg.py_var, dtype=x_t.dtype) + sigma_x0_sq
        else:
            q_var_eff = jnp.asarray(cfg.q_var, dtype=x_t.dtype)
            py_var_eff = jnp.asarray(cfg.py_var, dtype=x_t.dtype)

        # d/dy log N(y|m,v) = -(y-m)/v
        dldy = (-(y_hat - cfg.q_mean) / q_var_eff) - (
            -(y_hat - cfg.py_mean) / py_var_eff
        )
        # i.e. = -(y_hat-q_mean)/q_var_eff + (y_hat-py_mean)/py_var_eff

        # ∇_{x0} log ratio = a * d/dy
        grad_x0 = a_vec * dldy

        # Map to x_t (DPS-style)
        grad_xt = grad_x0 / a_safe

        # SNR ramp
        snr = (a_safe * a_safe) / (s2 + cfg.eps)
        w = (snr / (1.0 + snr)) ** cfg.snr_gamma

        extra = cfg.guidance_scale * gate * w * grad_xt

        # Norm clip
        nrm = jnp.linalg.norm(extra)
        maxn = jnp.asarray(cfg.max_guidance_norm, dtype=extra.dtype)
        extra = extra * jnp.minimum(1.0, maxn / (nrm + cfg.eps))
        return extra

    return guidance
