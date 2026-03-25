# quantbayes/stochax/vae/latent_diffusion/sample_prior.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.diffusion.edm import edm_precond_scalars

from quantbayes.stochax.diffusion.samplers.dpm_solver_pp import (
    make_dpmpp_3m_sampler,
    sample_dpmpp_2m,  # fallback (single-sample)
)
from quantbayes.stochax.diffusion.samplers.edm_heun import make_edm_heun_sampler
from quantbayes.stochax.diffusion.samplers.unipc import make_unipc_sampler
from quantbayes.stochax.diffusion.samplers.ipndm import (
    make_ipndm_sampler,
    make_ipndm4_sampler,
)


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
class LatentEDMSampleConfig:
    steps: int = 30
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    sampler: str = "dpmpp_3m"  # dpmpp_3m | unipc | ipndm | ipndm4 | heun | dpmpp_2m


def make_latent_denoise_fn(
    ema_model, *, sigma_data: float
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    EDM-correct denoise_fn(log_sigma, z)->D with input preconditioning z_in = c_in * z.
    """
    ema_eval = _inference_copy(ema_model)
    sd = float(sigma_data)

    def denoise_fn(log_sigma: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        c_in, _, _ = edm_precond_scalars(sigma, sd)
        return ema_eval(log_sigma, z * c_in, key=None, train=False)

    return denoise_fn


def sample_latent_edm(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    key: jr.PRNGKey,
    num_samples: int,
    latent_dim: int,
    cfg: LatentEDMSampleConfig,
) -> jnp.ndarray:
    """
    Fast latent sampling using batched+JIT samplers where available.
    Returns (num_samples, latent_dim).
    """
    s = cfg.sampler.lower().replace("-", "").replace("_", "").strip()
    sample_shape = (latent_dim,)

    if s == "dpmpp3m":
        sampler = make_dpmpp_3m_sampler(
            denoise_fn,
            sample_shape=sample_shape,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return sampler(key, num_samples)

    if s in ("heun", "edmheun"):
        sampler = make_edm_heun_sampler(
            denoise_fn,
            sample_shape=sample_shape,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return sampler(key, num_samples)

    if s in ("unipc", "unifiedpc"):
        sampler = make_unipc_sampler(
            denoise_fn,
            sample_shape=sample_shape,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return sampler(key, num_samples)

    if s == "ipndm":
        sampler = make_ipndm_sampler(
            denoise_fn,
            sample_shape=sample_shape,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return sampler(key, num_samples)

    if s in ("ipndm4",):
        sampler = make_ipndm4_sampler(
            denoise_fn,
            sample_shape=sample_shape,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return sampler(key, num_samples)

    if s == "dpmpp2m":
        # fallback (single-sample) + vmap
        keys = jr.split(key, num_samples)
        fn = lambda k: sample_dpmpp_2m(
            denoise_fn,
            sample_shape,
            key=k,
            steps=cfg.steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
        )
        return jax.vmap(fn)(keys)

    raise ValueError(f"Unknown sampler={cfg.sampler!r}")
