# quantbayes/stochax/diffusion/samplers/dpm_solver_pp.py
from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0


def _x0_from_D(denoise_fn, x, sigma, sigma_data):
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-8))
    D = denoise_fn(log_sigma, x)  # EDM head
    return edm_denoise_to_x0(x, D, sigma, sigma_data)


def sample_dpmpp_2m(
    denoise_fn,  # callable(log_sigma, x) -> D
    shape,
    *,
    key,
    steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
):
    """
    DPM-Solver++(2M) using x0-prediction with Karras sigmas.
    Compatible with EDM denoiser heads via D -> x0 conversion.
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    last = len(sigmas) - 2  # last transition is to appended sigma=0

    for i in range(len(sigmas) - 1):
        s = sigmas[i]
        sn = sigmas[i + 1]

        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        # JIT-safe final step: snap to x0 (sn is appended 0)
        if i == last:
            x = x0_s
            continue

        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))

        # Midpoint eval
        eh2 = jnp.exp(0.5 * h)
        s_mid = eh2 * s
        x_mid = eh2 * x - (eh2 - 1.0) * x0_s
        x0_mid = _x0_from_D(denoise_fn, x_mid, s_mid, sigma_data)

        # 2M update
        eh = jnp.exp(h)
        x = eh * x - (eh - 1.0) * x0_mid

    return x


def sample_dpmpp_3m(
    denoise_fn,
    shape,
    *,
    key,
    steps=20,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
):
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    last = len(sigmas) - 2  # last transition is to appended sigma=0

    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]

        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        # JIT-safe final step: snap to x0
        if i == last:
            x = x0_s
            continue

        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))

        e1 = jnp.exp(h / 3.0)
        x1 = e1 * x - (e1 - 1.0) * x0_s
        x0_1 = _x0_from_D(denoise_fn, x1, e1 * s, sigma_data)

        e2 = jnp.exp(2.0 * h / 3.0)
        x2 = e2 * x - (e2 - 1.0) * x0_1
        x0_2 = _x0_from_D(denoise_fn, x2, e2 * s, sigma_data)

        eh = jnp.exp(h)
        x = eh * x - (eh - 1.0) * ((1.0 / 4.0) * x0_s + (3.0 / 4.0) * x0_2)

    return x


# ---------------------------------------------------------------------
# Batched + JIT DPM-Solver++(3M)
# ---------------------------------------------------------------------
def make_dpmpp_3m_sampler(
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

    Notes:
    - This is JIT-friendly and uses lax.fori_loop.
    - Expects denoise_fn(log_sigma, x) to support batched x (B, ...).
    - Uses index-based final-step snap-to-x0 (no float equality checks).
    """
    sigmas = get_sigmas_karras(
        steps, sigma_min, sigma_max, rho=rho, include_zero=True
    )  # (steps+1,)
    last = sigmas.shape[0] - 2  # final transition is to appended sigma=0
    sd = float(sigma_data)

    def _x0(x_state: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        # sigma is scalar; x_state is (B, ...)
        return _x0_from_D(denoise_fn, x_state, sigma, sd)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        def body(i, x_state):
            s = sigmas[i]
            sn = sigmas[i + 1]

            x0_s = _x0(x_state, s)

            def do_final(_):
                return x0_s

            def do_step(_):
                # log-sigma step
                h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))

                # internal points
                e1 = jnp.exp(h / 3.0)
                x1 = e1 * x_state - (e1 - 1.0) * x0_s
                x0_1 = _x0(x1, e1 * s)

                e2 = jnp.exp(2.0 * h / 3.0)
                x2 = e2 * x_state - (e2 - 1.0) * x0_1
                x0_2 = _x0(x2, e2 * s)

                # 3M update
                eh = jnp.exp(h)
                return eh * x_state - (eh - 1.0) * (0.25 * x0_s + 0.75 * x0_2)

            return jax.lax.cond(i == last, do_final, do_step, operand=None)

        x = jax.lax.fori_loop(0, sigmas.shape[0] - 1, body, x)
        return x

    return sample
