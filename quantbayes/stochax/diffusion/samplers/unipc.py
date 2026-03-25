# quantbayes/stochax/diffusion/samplers/unipc.py
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0


def _x0_from_D(denoise_fn, x, sigma, sigma_data):
    """EDM head -> x0 at noise level `sigma`."""
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-8))
    D = denoise_fn(log_sigma, x)
    return edm_denoise_to_x0(x, D, sigma, sigma_data)


def sample_unipc(
    denoise_fn,  # callable(log_sigma, x) -> D  (EDM head)
    shape: tuple,
    *,
    key,
    steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
):
    """
    UniPC (2nd-order) in log-sigma time, single-sample (non-batched).

    ODE in t = log(sigma):
        dx/dt = g(x,t) = x - x0(x, sigma)
    Predictor-corrector (Heun) in t:
      - predictor:  x_e = x + h * g_s
      - corrector:  x   = x + 0.5*h*(g_s + g_n)
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]
    last = len(sigmas) - 2

    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        if i == last:
            x = x0_s
            continue

        ls = jnp.log(jnp.maximum(s, 1e-12))
        lsn = jnp.log(jnp.maximum(sn, 1e-12))
        h = lsn - ls

        g_s = x - x0_s
        x_e = x + h * g_s

        x0_n = _x0_from_D(denoise_fn, x_e, sn, sigma_data)
        g_n = x_e - x0_n
        x = x + 0.5 * h * (g_s + g_n)

    return x


# ---------------------------------------------------------------------
# Batched + JIT UniPC (2nd order)
# ---------------------------------------------------------------------
def make_unipc_sampler(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],  # (log_sigma, x)->D
    *,
    sample_shape: tuple[int, ...],
    steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    Returns a batched sampler:
        sample(key, num_samples) -> (num_samples, *sample_shape)

    - JIT-friendly (lax.fori_loop).
    - Index-based final-step snap-to-x0 (no float-equality checks).
    - denoise_fn(log_sigma, x) must support batched x (B, ...).
    """
    sigmas = get_sigmas_karras(
        steps, sigma_min, sigma_max, rho=rho, include_zero=True
    )  # (steps+1,)
    last = sigmas.shape[0] - 2
    sd = float(sigma_data)

    def x0_hat(x_state: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return _x0_from_D(denoise_fn, x_state, sigma, sd)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        def body(i, x_state):
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0_s = x0_hat(x_state, s)

            def do_final(_):
                return x0_s

            def do_step(_):
                ls = jnp.log(jnp.maximum(s, 1e-12))
                lsn = jnp.log(jnp.maximum(sn, 1e-12))
                h = lsn - ls

                g_s = x_state - x0_s
                x_e = x_state + h * g_s

                x0_n = x0_hat(x_e, sn)
                g_n = x_e - x0_n
                return x_state + 0.5 * h * (g_s + g_n)

            return jax.lax.cond(i == last, do_final, do_step, operand=None)

        x = jax.lax.fori_loop(0, sigmas.shape[0] - 1, body, x)
        return x

    return sample
