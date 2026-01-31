# quantbayes/pkdiffusion/guidance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.parameterizations import vp_alpha_sigma
from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig
from quantbayes.pkstruct.utils.stats import log_scaled_beta_pdf

Array = jax.Array


@dataclass(frozen=True)
class VRWRadialRRGuidanceConfig:
    """
    RR/PK radial guidance for VRW endpoint diffusion:

        log L(r) = log q(r) - log ref(r)

    q(r): ScaledBeta(alpha, beta) on r in (0, N)
    ref(r): either
      - Stephens approximation ("stephens"), or
      - a ScaledBeta fit to prior r/N samples ("beta")

    The guidance uses a DPS-style Tweedie x0-hat:
        x0_hat = (x_t + sigma(t)^2 * score_t) / alpha(t)

    and radial gradient in x0 space, approximately mapped to x_t space via 1/alpha(t).
    """

    N: int = 5
    kappa: float = 10.0

    # Evidence q(r)
    alpha: float = 10.0
    beta: float = 10.0

    # Reference ref(r)
    ref_kind: Literal["stephens", "beta"] = "stephens"
    ref_alpha: float | None = None
    ref_beta: float | None = None

    # Guidance strength
    guidance_scale: float = 1.0

    # Numerical stability
    eps: float = 1e-6
    min_alpha: float = 0.05
    max_guidance_norm: float = 25.0

    # Soft ramp-in based on SNR; bounded in [0,1]
    snr_gamma: float = 1.0

    noise_aware: bool = True
    noise_power: float = 1.0  # optional; keep 1.0


def make_vrw_radial_rr_guidance(cfg: VRWRadialRRGuidanceConfig) -> Callable:
    # Build Stephens config only if needed
    steph_cfg = (
        StephensConfig(kappa=float(cfg.kappa), N=int(cfg.N))
        if cfg.ref_kind == "stephens"
        else None
    )
    # evidence variance of r when u=r/N ~ Beta(alpha,beta)
    a = float(cfg.alpha)
    b = float(cfg.beta)
    N = float(cfg.N)
    var_u = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    var_r = (N * N) * var_u
    var_r = jnp.asarray(var_r)

    def ref_logpdf_r(r: Array) -> Array:
        if cfg.ref_kind == "stephens":
            assert steph_cfg is not None
            return stephens_logpdf_r(r, cfg=steph_cfg)
        if cfg.ref_kind == "beta":
            if cfg.ref_alpha is None or cfg.ref_beta is None:
                raise ValueError("ref_alpha/ref_beta must be set when ref_kind='beta'")
            return log_scaled_beta_pdf(r, cfg.ref_alpha, cfg.ref_beta, cfg.N)
        raise ValueError(f"Unknown ref_kind={cfg.ref_kind!r}")

    def log_ratio_r(r: Array) -> Array:
        return log_scaled_beta_pdf(r, cfg.alpha, cfg.beta, cfg.N) - ref_logpdf_r(r)

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

        # Gate: avoid pathological Tweedie when alpha is extremely small
        gate = (a >= cfg.min_alpha).astype(x_t.dtype)

        # Tweedie estimate for x0
        a_safe = jnp.maximum(a, cfg.eps)
        s2 = s * s
        x0_hat = (x_t + s2 * score_t) / a_safe  # (2,)

        sigma_x0_sq = s2 / (a_safe * a_safe + cfg.eps)  # scalar

        if cfg.noise_aware:
            shrink = var_r / (var_r + sigma_x0_sq + cfg.eps)
            shrink = shrink ** jnp.asarray(cfg.noise_power, dtype=x_t.dtype)
            dldr = dldr * shrink

        # Direction uses raw norm
        r_raw = jnp.linalg.norm(x0_hat)
        direction = x0_hat / (r_raw + cfg.eps)

        # Clip r only for evaluating derivative
        r_eval = jnp.clip(r_raw, cfg.eps, float(cfg.N) - cfg.eps)
        dldr = dlog_ratio_dr(r_eval)

        grad_x0 = direction * dldr
        grad_xt = grad_x0 / a_safe

        # bounded SNR ramp
        snr = (a_safe * a_safe) / (s2 + cfg.eps)
        w = (snr / (1.0 + snr)) ** cfg.snr_gamma

        extra = cfg.guidance_scale * gate * w * grad_xt

        # Norm clip
        nrm = jnp.linalg.norm(extra)
        maxn = jnp.asarray(cfg.max_guidance_norm, dtype=extra.dtype)
        extra = extra * jnp.minimum(1.0, maxn / (nrm + cfg.eps))
        return extra

    return guidance
