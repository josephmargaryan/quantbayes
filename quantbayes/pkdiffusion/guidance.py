# quantbayes/pkdiffusion/guidance.py
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
    Guidance for VRW endpoint diffusion with RR/PK radial evidence:

        log L(r) = log q(r) - log pi_ref(r)

    where q is ScaledBeta on r in (0,N) and pi_ref is Stephens approx.

    Key numerical issue:
      For VP SDE with large t1 (e.g. 10), alpha(t) can be extremely small at large t.
      Any Tweedie-based x0_hat uses division by alpha(t). So we *gate* guidance:
        - if alpha(t) < min_alpha: return 0 guidance.

    We also:
      - compute the radial direction using r_raw (not clipped r),
      - clip r only for logpdf evaluation,
      - optionally clip the final guidance norm.
    """

    N: int = 5
    kappa: float = 10.0
    alpha: float = 10.0
    beta: float = 10.0

    guidance_scale: float = 1.0

    # Numerical stability
    eps: float = 1e-6
    min_alpha: float = 0.05  # <-- IMPORTANT: guidance off when alpha(t) too small
    max_guidance_norm: float = 25.0  # norm clip on extra score term

    # Soft ramp-in based on SNR; bounded in [0,1]
    snr_gamma: float = 1.0


def make_vrw_radial_rr_guidance(cfg: VRWRadialRRGuidanceConfig) -> Callable:
    """
    Returns:
        guidance(t, x_t, score_t, *, int_beta_fn) -> extra_score (same shape as x_t)

    Uses a DPS-style approximation:
        x0_hat = (x_t + sigma(t)^2 * score_t) / alpha(t)    (Tweedie)
        grad_x0 log L(||x0||) is radial
        grad_xt approx = (1/alpha(t)) * grad_x0  (ignoring d(score)/dx)

    Then applies:
        - alpha gate (turn off when alpha < min_alpha),
        - SNR ramp-in in [0,1],
        - norm clip.
    """
    steph_cfg = StephensConfig(kappa=float(cfg.kappa), N=int(cfg.N))

    def log_ratio_r(r: Array) -> Array:
        # scalar r
        return log_scaled_beta_pdf(r, cfg.alpha, cfg.beta, cfg.N) - stephens_logpdf_r(
            r, cfg=steph_cfg
        )

    dlog_ratio_dr = jax.grad(log_ratio_r)

    def guidance(
        t: Array,
        x_t: Array,  # (2,)
        score_t: Array,  # (2,)
        *,
        int_beta_fn,
    ) -> Array:
        # VP scalars
        a, s = vp_alpha_sigma(int_beta_fn, t)
        a = jnp.asarray(a)
        s = jnp.asarray(s)

        # Gate: if alpha is too small, Tweedie x0_hat is numerically explosive.
        gate = (a >= cfg.min_alpha).astype(x_t.dtype)

        # Tweedie estimate for x0
        a_safe = jnp.maximum(a, cfg.eps)
        s2 = s * s
        x0_hat = (x_t + s2 * score_t) / a_safe  # (2,)

        # Use RAW norm for direction (stable)
        r_raw = jnp.linalg.norm(x0_hat)
        direction = x0_hat / (r_raw + cfg.eps)  # unit-ish vector

        # Clip r only for evaluating log densities / derivatives
        r_eval = jnp.clip(r_raw, cfg.eps, float(cfg.N) - cfg.eps)
        dldr = dlog_ratio_dr(r_eval)  # scalar

        # grad in x0 space
        grad_x0 = direction * dldr  # (2,)

        # chain approx to xt
        grad_xt = grad_x0 / a_safe  # (2,)

        # bounded SNR-based ramp-in: snr/(1+snr) in [0,1]
        snr = (a_safe * a_safe) / (s2 + cfg.eps)
        w = (snr / (1.0 + snr)) ** cfg.snr_gamma

        extra = cfg.guidance_scale * gate * w * grad_xt  # (2,)

        # Norm clip for safety
        nrm = jnp.linalg.norm(extra)
        maxn = jnp.asarray(cfg.max_guidance_norm, dtype=extra.dtype)
        extra = extra * jnp.minimum(1.0, maxn / (nrm + cfg.eps))
        return extra

    return guidance
