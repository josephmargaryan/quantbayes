# quantbayes/stochax/vae/latent_diffusion/sample_prior_conditional.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.samplers.dpm_solver_pp import make_dpmpp_3m_sampler

from .pk_guidance import wrap_denoise_fn_with_x0_guidance, DecodedInkPKGuidance


def _inference_copy(model):
    maybe = eqx.nn.inference_mode(model)
    enter = getattr(maybe, "__enter__", None)
    exit_ = getattr(maybe, "__exit__", None)
    if callable(enter) and callable(exit_):
        try:
            m = enter()
            return m
        finally:
            exit_(None, None, None)
    return maybe


@dataclass(frozen=True)
class LatentEDMCondSampleConfig:
    steps: int = 40
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    cfg_scale: float = 3.0  # classifier-free guidance strength


def make_latent_cond_denoise_cfg_fn(
    ema_model,
    *,
    sigma_data: float,
    label: int,
    cfg_scale: float,
    null_label: int,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Returns EDM denoise_fn(log_sigma, z)->D for fixed label with CFG mixing.
    Shape-safe:
      - if input z is (D,), returns (D,)
      - if input z is (B,D), returns (B,D) even when B==1
    """
    ema_eval = _inference_copy(ema_model)
    sd = float(sigma_data)
    label = int(label)
    null_label = int(null_label)
    cfg_scale = float(cfg_scale)

    def denoise_fn(log_sigma: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        squeeze = z.ndim == 1
        if squeeze:
            z_b = z[None, :]
        else:
            z_b = z

        b = z_b.shape[0]
        sigma = jnp.exp(log_sigma)
        c_in, _, _ = edm_precond_scalars(sigma, sd)
        z_in = z_b * c_in

        y_null = jnp.full((b,), null_label, dtype=jnp.int32)
        Du = ema_eval(log_sigma, z_in, key=None, train=False, label=y_null)

        y = jnp.full((b,), label, dtype=jnp.int32)
        Dc = ema_eval(log_sigma, z_in, key=None, train=False, label=y)

        D = Du + cfg_scale * (Dc - Du)
        return D[0] if squeeze else D

    return denoise_fn


def sample_latent_edm_conditional_cfg(
    *,
    ema_model,
    key: jr.PRNGKey,
    label: int,
    num_samples: int,
    latent_dim: int,
    cfg: LatentEDMCondSampleConfig,
    null_label: int,
    pk_guidance: Optional[DecodedInkPKGuidance] = None,
) -> jnp.ndarray:
    """
    Builds CFG denoise_fn, optional PK wrapper, then samples with batched JIT DPM++(3M).
    Returns (num_samples, latent_dim).
    """
    denoise = make_latent_cond_denoise_cfg_fn(
        ema_model,
        sigma_data=cfg.sigma_data,
        label=label,
        cfg_scale=cfg.cfg_scale,
        null_label=null_label,
    )

    if pk_guidance is not None:
        denoise = wrap_denoise_fn_with_x0_guidance(
            denoise,
            sigma_data=cfg.sigma_data,
            guidance=pk_guidance,
        )

    sampler = make_dpmpp_3m_sampler(
        denoise,
        sample_shape=(latent_dim,),
        steps=cfg.steps,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        sigma_data=cfg.sigma_data,
        rho=cfg.rho,
    )
    return sampler(key, num_samples)
