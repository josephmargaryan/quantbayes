# quantbayes/pkdiffusion/guidance_ratio_rff.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.parameterizations import vp_alpha_sigma
from quantbayes.pkstruct.ratio.rff_ratio import RFFRatioModel1DJax

Array = jax.Array


@dataclass(frozen=True)
class LinearRFFRatioGuidanceConfig:
    """
    Guidance for scalar evidence on y = a^T x0, but using a learned log ratio model:
      log ratio(y) ≈ log(q(y)/p_Y(y))

    The guidance uses d/dy log ratio at y_hat = a^T x0_hat.
    """

    a: Array  # (d,)
    ratio: RFFRatioModel1DJax

    guidance_scale: float = 1.0
    eps: float = 1e-6
    min_alpha: float = 0.05
    max_guidance_norm: float = 25.0
    snr_gamma: float = 1.0

    # Optional tempering (cheap noise-aware stabilization):
    noise_aware: bool = True
    noise_power: float = 1.0
    var_y: Optional[float] = None  # if set, shrink by var_y/(var_y + sigma_y^2)


def make_linear_rff_ratio_guidance(cfg: LinearRFFRatioGuidanceConfig) -> Callable:
    a_vec = jnp.asarray(cfg.a).reshape(-1)

    a_norm2 = jnp.sum(a_vec * a_vec)

    def guidance(
        t: Array,
        x_t: Array,
        score_t: Array,
        *,
        int_beta_fn,
    ) -> Array:
        alpha_t, sigma_t = vp_alpha_sigma(int_beta_fn, t)
        alpha_t = jnp.asarray(alpha_t, dtype=x_t.dtype)
        sigma_t = jnp.asarray(sigma_t, dtype=x_t.dtype)

        gate = (alpha_t >= cfg.min_alpha).astype(x_t.dtype)
        alpha_safe = jnp.maximum(alpha_t, cfg.eps)
        sigma2 = sigma_t * sigma_t

        # Tweedie estimate of x0
        x0_hat = (x_t + sigma2 * score_t) / alpha_safe

        y_hat = jnp.dot(a_vec, x0_hat)
        dldy = cfg.ratio.dlog_ratio_dy(y_hat)  # scalar

        if cfg.noise_aware and (cfg.var_y is not None):
            # isotropic x0 uncertainty => Var(y_hat) ≈ (sigma2/alpha^2) * ||a||^2
            sigma_x0_sq = sigma2 / (alpha_safe * alpha_safe + cfg.eps)
            sigma_y_sq = sigma_x0_sq * a_norm2
            var_y = jnp.asarray(cfg.var_y, dtype=x_t.dtype)
            shrink = var_y / (var_y + sigma_y_sq + cfg.eps)
            shrink = shrink ** jnp.asarray(cfg.noise_power, dtype=x_t.dtype)
            dldy = dldy * shrink

        grad_x0 = a_vec * dldy
        grad_xt = grad_x0 / alpha_safe

        snr = (alpha_safe * alpha_safe) / (sigma2 + cfg.eps)
        w = (snr / (1.0 + snr)) ** cfg.snr_gamma

        extra = cfg.guidance_scale * gate * w * grad_xt

        # Norm clip
        nrm = jnp.linalg.norm(extra)
        maxn = jnp.asarray(cfg.max_guidance_norm, dtype=extra.dtype)
        extra = extra * jnp.minimum(1.0, maxn / (nrm + cfg.eps))
        return extra

    return guidance


@dataclass(frozen=True)
class RadialRFFRatioGuidanceConfig:
    """
    Guidance for radial evidence using a learned log ratio model on r=||x0||.

    d/dr log ratio(r_hat) gives the scalar radial force, applied along x0_hat direction.
    """

    N: int
    ratio: RFFRatioModel1DJax

    guidance_scale: float = 1.0
    eps: float = 1e-6
    min_alpha: float = 0.05
    max_guidance_norm: float = 25.0
    snr_gamma: float = 1.0

    noise_aware: bool = True
    noise_power: float = 1.0
    var_r: Optional[float] = None  # if set, shrink by var_r/(var_r + sigma_r^2)

    support_barrier: float = 0.0
    support_eps: float = 1e-3


def make_radial_rff_ratio_guidance(cfg: RadialRFFRatioGuidanceConfig) -> Callable:
    Nf = float(cfg.N)

    def guidance(
        t: Array,
        x_t: Array,  # (2,)
        score_t: Array,  # (2,)
        *,
        int_beta_fn,
    ) -> Array:
        alpha_t, sigma_t = vp_alpha_sigma(int_beta_fn, t)
        alpha_t = jnp.asarray(alpha_t, dtype=x_t.dtype)
        sigma_t = jnp.asarray(sigma_t, dtype=x_t.dtype)

        gate = (alpha_t >= cfg.min_alpha).astype(x_t.dtype)
        alpha_safe = jnp.maximum(alpha_t, cfg.eps)
        sigma2 = sigma_t * sigma_t

        x0_hat = (x_t + sigma2 * score_t) / alpha_safe

        r_raw = jnp.linalg.norm(x0_hat)
        direction = x0_hat / (r_raw + cfg.eps)
        r_eval = jnp.clip(r_raw, cfg.eps, Nf - cfg.eps)

        dldr = cfg.ratio.dlog_ratio_dy(r_eval)

        # Optional barrier for r > N - eps
        if cfg.support_barrier > 0.0:
            thr = jnp.asarray(Nf - cfg.support_eps, dtype=x_t.dtype)
            overflow = jnp.maximum(r_raw - thr, 0.0)
            dldr = dldr - jnp.asarray(cfg.support_barrier, dtype=x_t.dtype) * overflow

        if cfg.noise_aware and (cfg.var_r is not None):
            sigma_x0_sq = sigma2 / (alpha_safe * alpha_safe + cfg.eps)
            # delta method for r=||x|| under isotropic noise => Var(r) ≈ sigma_x0_sq
            sigma_r_sq = sigma_x0_sq
            var_r = jnp.asarray(cfg.var_r, dtype=x_t.dtype)
            shrink = var_r / (var_r + sigma_r_sq + cfg.eps)
            shrink = shrink ** jnp.asarray(cfg.noise_power, dtype=x_t.dtype)
            dldr = dldr * shrink

        grad_x0 = direction * dldr
        grad_xt = grad_x0 / alpha_safe

        snr = (alpha_safe * alpha_safe) / (sigma2 + cfg.eps)
        w = (snr / (1.0 + snr)) ** cfg.snr_gamma

        extra = cfg.guidance_scale * gate * w * grad_xt

        nrm = jnp.linalg.norm(extra)
        maxn = jnp.asarray(cfg.max_guidance_norm, dtype=extra.dtype)
        extra = extra * jnp.minimum(1.0, maxn / (nrm + cfg.eps))
        return extra

    return guidance
