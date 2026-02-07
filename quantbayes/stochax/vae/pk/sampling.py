# quantbayes/stochax/vae/pk/sampling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def get_sigmas_karras(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    include_zero: bool = False,
) -> jnp.ndarray:
    ramp = jnp.linspace(0.0, 1.0, n)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    if include_zero:
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=sigmas.dtype)])
    return sigmas


@dataclass(frozen=True)
class AnnealedLangevinConfig:
    n_sigmas: int = 30
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    rho: float = 7.0

    steps_per_sigma: int = 5
    step_scale: float = 0.10  # step_size = step_scale * sigma^2

    final_denoise: bool = True
    max_norm: Optional[float] = None  # clip ||z|| per sample
    eps: float = 1e-12


def make_annealed_langevin_sampler(
    score_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ],  # (log_sigma, z)->score
    *,
    shape: tuple[int, ...],
    cfg: AnnealedLangevinConfig,
) -> Callable[[jr.PRNGKey], jnp.ndarray]:
    """
    Factory returning a compiled sampler:
        sample(key) -> z  (shape)

    This avoids defining the jitted loop “fresh” in every sampling call.
    """
    sigmas = get_sigmas_karras(
        cfg.n_sigmas, cfg.sigma_min, cfg.sigma_max, rho=cfg.rho, include_zero=False
    )
    sigmas_rep = jnp.repeat(sigmas, cfg.steps_per_sigma)
    K = int(sigmas_rep.shape[0])

    @eqx.filter_jit
    def sample(key: jr.PRNGKey) -> jnp.ndarray:
        k_init, k_loop = jr.split(key, 2)
        z = jr.normal(k_init, shape) * float(cfg.sigma_max)

        def body(i, zt):
            sigma = sigmas_rep[i]
            log_sigma = jnp.log(jnp.maximum(sigma, cfg.eps))
            step = jnp.asarray(cfg.step_scale, zt.dtype) * (sigma**2)

            nk = jr.fold_in(k_loop, i)
            noise = jr.normal(nk, zt.shape)

            sc = score_fn(log_sigma, zt)
            zt = zt + step * sc + jnp.sqrt(jnp.maximum(2.0 * step, cfg.eps)) * noise

            if cfg.max_norm is not None:
                axes = tuple(range(1, zt.ndim))
                n = jnp.sqrt(jnp.sum(zt * zt, axis=axes, keepdims=True) + cfg.eps)
                clip = jnp.minimum(1.0, float(cfg.max_norm) / n)
                zt = zt * clip

            return zt

        z = jax.lax.fori_loop(0, K, body, z)

        if cfg.final_denoise:
            sigma = jnp.asarray(cfg.sigma_min, dtype=z.dtype)
            log_sigma = jnp.log(jnp.maximum(sigma, cfg.eps))
            step = jnp.asarray(cfg.step_scale, z.dtype) * (sigma**2)
            z = z + step * score_fn(log_sigma, z)

        return z

    return sample


def sample_annealed_langevin(
    score_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    key: jr.PRNGKey,
    shape: tuple[int, ...],
    cfg: AnnealedLangevinConfig,
) -> jnp.ndarray:
    """
    Back-compat wrapper. For repeated sampling, prefer:
        sampler = make_annealed_langevin_sampler(score_fn, shape=..., cfg=...)
        z = sampler(key)
    """
    sampler = make_annealed_langevin_sampler(score_fn, shape=shape, cfg=cfg)
    return sampler(key)
