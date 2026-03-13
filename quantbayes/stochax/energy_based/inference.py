# quantbayes/stochax/energy_based/inference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from .base import BaseEBM


ScoreFn = Callable[
    [jnp.ndarray, jnp.ndarray], jnp.ndarray
]  # (log_sigma, x) -> score same shape as x


@dataclass(frozen=True)
class SGLDConfig:
    n_steps: int = 60
    step_size: float = 1e-2
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = 1.0
    max_norm: Optional[float] = None
    eps: float = 1e-12


@dataclass(frozen=True)
class AnnealedLangevinConfig:
    n_sigmas: int = 30
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    rho: float = 7.0
    steps_per_sigma: int = 6
    step_scale: float = 0.08  # step = step_scale * sigma^2
    final_denoise: bool = True
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = 1.0
    max_norm: Optional[float] = None
    eps: float = 1e-12


def make_score_fn_from_ebm(ebm: BaseEBM) -> ScoreFn:
    """Base score: ∇x log p(x) ≈ -∇x E(x). Ignores log_sigma."""

    def score_fn(log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return ebm.score(x)

    return score_fn


def make_sgld_sampler(
    score_fn: ScoreFn,
    *,
    sample_shape: tuple[int, ...],
    cfg: SGLDConfig,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    Returns: sample(key, num_samples) -> (B, *sample_shape)
    Init: x0 ~ N(0, I)
    """
    step_size = float(cfg.step_size)
    n_steps = int(cfg.n_steps)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        k0, kloop = jr.split(key, 2)
        x = jr.normal(k0, (num_samples, *sample_shape))

        def body(i, x_t):
            k = jr.fold_in(kloop, i)
            noise = jr.normal(k, x_t.shape)
            # score is grad log p; SGLD: x <- x + 0.5*η*score + sqrt(η)*N
            sc = score_fn(jnp.asarray(0.0, x_t.dtype), x_t)
            x_t = x_t + 0.5 * step_size * sc + jnp.sqrt(step_size) * noise

            if cfg.max_norm is not None:
                axes = tuple(range(1, x_t.ndim))
                n = jnp.sqrt(jnp.sum(x_t * x_t, axis=axes, keepdims=True) + cfg.eps)
                clip = jnp.minimum(1.0, float(cfg.max_norm) / n)
                x_t = x_t * clip

            if cfg.clamp_min is not None and cfg.clamp_max is not None:
                x_t = jnp.clip(x_t, float(cfg.clamp_min), float(cfg.clamp_max))
            return x_t

        x = jax.lax.fori_loop(0, n_steps, body, x)
        return x

    return sample


def make_annealed_langevin_sampler(
    score_fn: ScoreFn,
    *,
    sample_shape: tuple[int, ...],
    cfg: AnnealedLangevinConfig,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    Returns: sample(key, num_samples) -> (B, *sample_shape)
    Init: x0 ~ N(0, sigma_max^2 I)
    Anneals sigma from sigma_max -> sigma_min.
    """
    sigmas = get_sigmas_karras(
        cfg.n_sigmas, cfg.sigma_min, cfg.sigma_max, rho=cfg.rho, include_zero=False
    )
    sigmas_rep = jnp.repeat(sigmas, int(cfg.steps_per_sigma))
    K = int(sigmas_rep.shape[0])
    step_scale = float(cfg.step_scale)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        k0, kloop = jr.split(key, 2)
        x = jr.normal(k0, (num_samples, *sample_shape)) * float(cfg.sigma_max)

        def body(i, x_t):
            sigma = sigmas_rep[i]
            log_sigma = jnp.log(jnp.maximum(sigma, cfg.eps))
            step = step_scale * (sigma**2)

            k = jr.fold_in(kloop, i)
            noise = jr.normal(k, x_t.shape)

            sc = score_fn(log_sigma, x_t)
            x_t = x_t + 0.5 * step * sc + jnp.sqrt(jnp.maximum(step, cfg.eps)) * noise

            if cfg.max_norm is not None:
                axes = tuple(range(1, x_t.ndim))
                n = jnp.sqrt(jnp.sum(x_t * x_t, axis=axes, keepdims=True) + cfg.eps)
                clip = jnp.minimum(1.0, float(cfg.max_norm) / n)
                x_t = x_t * clip

            if cfg.clamp_min is not None and cfg.clamp_max is not None:
                x_t = jnp.clip(x_t, float(cfg.clamp_min), float(cfg.clamp_max))
            return x_t

        x = jax.lax.fori_loop(0, K, body, x)

        if cfg.final_denoise:
            sigma = jnp.asarray(cfg.sigma_min, dtype=x.dtype)
            log_sigma = jnp.log(jnp.maximum(sigma, cfg.eps))
            step = step_scale * (sigma**2)
            x = x + 0.5 * step * score_fn(log_sigma, x)  # no noise
            if cfg.clamp_min is not None and cfg.clamp_max is not None:
                x = jnp.clip(x, float(cfg.clamp_min), float(cfg.clamp_max))

        return x

    return sample
