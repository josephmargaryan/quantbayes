# quantbayes/stochax/diffusion/samplers/edm_heun.py
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0


def _denoise_edm(denoise_fn, x, sigma, sigma_data):
    """denoise_fn(log_sigma, x) -> D; returns (x0_hat, velocity)."""
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-8))
    D = denoise_fn(log_sigma, x)  # EDM head
    x0 = edm_denoise_to_x0(x, D, sigma, sigma_data)
    v = (x - x0) / jnp.maximum(sigma, 1e-8)  # dx/dσ
    return x0, v


def sample_edm_heun(
    denoise_fn,  # callable(log_sigma, x)->D
    shape,
    *,
    key,
    steps: int = 30,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
):
    """
    Single-sample EDM-Heun sampler (Python loop).

    Uses Karras schedule with include_zero=True and index-based final snap-to-x0.
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]
    last = len(sigmas) - 2

    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0, v = _denoise_edm(denoise_fn, x, s, sigma_data)

        # Final step: snap to x0 (sn is the appended 0)
        if i == last:
            x = x0
            continue

        x_e = x + (sn - s) * v  # Euler
        x0n, vn = _denoise_edm(denoise_fn, x_e, sn, sigma_data)
        x = x + 0.5 * (sn - s) * (v + vn)  # Heun

    return x


# ---------------------------------------------------------------------
# Batched + JIT EDM-Heun
# ---------------------------------------------------------------------
def make_edm_heun_sampler(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],  # (log_sigma, x)->D
    *,
    sample_shape: tuple[int, ...],
    steps: int = 30,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    Returns a batched sampler:
        sample(key, num_samples) -> (num_samples, *sample_shape)

    - JIT-friendly: lax.fori_loop
    - Index-based final snap-to-x0 (no float equality checks)
    - denoise_fn(log_sigma, x) must support batched x (B, ...)
    """
    sigmas = get_sigmas_karras(
        steps, sigma_min, sigma_max, rho=rho, include_zero=True
    )  # (steps+1,)
    last = sigmas.shape[0] - 2
    sd = float(sigma_data)

    def x0_and_v(x_state: jnp.ndarray, sigma: jnp.ndarray):
        log_sigma = jnp.log(jnp.maximum(sigma, 1e-12))
        D = denoise_fn(log_sigma, x_state)
        x0 = edm_denoise_to_x0(x_state, D, sigma, sigma_data=sd)
        v = (x_state - x0) / jnp.maximum(sigma, 1e-12)  # dx/dσ
        return x0, v

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        def body(i, x_state):
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0, v = x0_and_v(x_state, s)

            def do_final(_):
                return x0

            def do_step(_):
                x_e = x_state + (sn - s) * v
                x0n, vn = x0_and_v(x_e, sn)
                return x_state + 0.5 * (sn - s) * (v + vn)

            return jax.lax.cond(i == last, do_final, do_step, operand=None)

        x = jax.lax.fori_loop(0, sigmas.shape[0] - 1, body, x)
        return x

    return sample
