from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.parameterizations import vp_alpha_sigma
from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig
from quantbayes.pkstruct.utils.stats import log_scaled_beta_pdf


Array = jax.Array


@dataclass(frozen=True)
class VRWRadialRRGuidanceConfig:
    """
    Guidance for VRW endpoint diffusion:
      log L(r) = log q(r) - log pi_ref(r)

    where q is ScaledBeta on r in (0,N) and pi_ref is Stephens approx.
    """

    N: int = 5
    kappa: float = 10.0
    alpha: float = 10.0
    beta: float = 10.0
    guidance_scale: float = 1.0
    eps: float = 1e-6


def make_vrw_radial_rr_guidance(cfg: VRWRadialRRGuidanceConfig) -> Callable:
    """
    Returns a function:
      extra_score = g(t, x_t, score_t, *, int_beta_fn) -> R^2

    This implements a DPS-style "denoise-then-guide" approximation:
      x0_hat = (x_t + sigma(t)^2 * score_t) / alpha(t)
      ∇_{x0} log L(||x0||) is radial
      ∇_{x_t} log L(x0_hat) ≈ (1/alpha(t)) * ∇_{x0} log L

    and we add it directly to the score.
    """
    steph_cfg = StephensConfig(kappa=float(cfg.kappa), N=int(cfg.N))

    def log_ratio_r(r: Array) -> Array:
        # r is scalar
        return log_scaled_beta_pdf(r, cfg.alpha, cfg.beta, cfg.N) - stephens_logpdf_r(
            r, cfg=steph_cfg
        )

    dlog_ratio_dr = jax.grad(log_ratio_r)

    def guidance(
        t: Array,
        x_t: Array,  # (2,)
        score_t: Array,  # (2,)
        *,
        int_beta_fn,  # callable int_beta(t)
    ) -> Array:
        # VP scalars
        a, s = vp_alpha_sigma(int_beta_fn, t)
        a = jnp.maximum(a, cfg.eps)
        s2 = s * s

        # Tweedie denoiser for VP:
        x0_hat = (x_t + s2 * score_t) / a  # (2,)

        r = jnp.linalg.norm(x0_hat)
        # Keep r in (0,N) for logpdf stability; this is a demo, not a theorem.
        r = jnp.clip(r, cfg.eps, float(cfg.N) - cfg.eps)

        dldr = dlog_ratio_dr(r)  # scalar

        # radial gradient in x0 space
        grad_x0 = (x0_hat / (r + cfg.eps)) * dldr

        # approximate chain to x_t
        grad_xt = grad_x0 / a
        return float(cfg.guidance_scale) * grad_xt

    return guidance
