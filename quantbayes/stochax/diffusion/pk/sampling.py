from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0
from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras


def _inference_copy(model):
    """
    Return a copy of `model` with dropout/etc. set to inference mode.
    Compatible with Equinox versions where inference_mode returns either:
      - a model copy, or
      - a context manager that yields a model copy.
    """
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


def make_preconditioned_edm_denoise_fn(
    ema_model,
    *,
    sigma_data: float,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Build an EDM denoise_fn(log_sigma, x)->D that matches your EDM training:
      model sees x_in = c_in * x
    """
    ema_eval = _inference_copy(ema_model)
    sd = float(sigma_data)

    def denoise_fn(log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        c_in, _, _ = edm_precond_scalars(sigma, sd)
        return ema_eval(log_sigma, x * c_in, key=None, train=False)

    return denoise_fn


@dataclass(frozen=True)
class EDMHeunConfig:
    steps: int = 40
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5


def make_edm_heun_sampler(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    sample_shape: tuple[int, ...],
    cfg: EDMHeunConfig,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    JIT-friendly batched EDM-Heun sampler (matches your notebook logic),
    but takes any EDM denoise_fn(log_sigma,x)->D (optionally guided).
    """

    sigmas = get_sigmas_karras(
        cfg.steps,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        rho=cfg.rho,
        include_zero=True,
    )
    sigma_data = float(cfg.sigma_data)

    def _x0_and_v(x_state: jnp.ndarray, sigma: jnp.ndarray):
        log_sigma = jnp.log(jnp.maximum(sigma, 1e-12))
        D = denoise_fn(log_sigma, x_state)
        x0 = edm_denoise_to_x0(x_state, D, sigma, sigma_data=sigma_data)
        v = (x_state - x0) / jnp.maximum(sigma, 1e-12)
        return x0, v

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        def step_fn(i, x_state):
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0, v = _x0_and_v(x_state, s)
            x_e = x_state + (sn - s) * v

            def do_final(_):
                return x0

            def do_heun(_):
                x0n, vn = _x0_and_v(x_e, sn)
                return x_state + 0.5 * (sn - s) * (v + vn)

            return jax.lax.cond(sn == 0.0, do_final, do_heun, operand=None)

        x = jax.lax.fori_loop(0, sigmas.shape[0] - 1, step_fn, x)
        return x

    return sample
