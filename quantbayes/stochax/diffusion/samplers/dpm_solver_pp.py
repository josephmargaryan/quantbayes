# quantbayes/stochax/diffusion/samplers/dpm_solver_pp.py
from __future__ import annotations
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

    for i in range(len(sigmas) - 1):
        s = sigmas[i]
        sn = sigmas[i + 1]
        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))

        # x0 at current sigma
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        if sn == 0.0:
            # Final step: snap to x0
            x = x0_s
            continue

        # Midpoint eval
        s_mid = jnp.exp(0.5 * h) * s
        x_mid = jnp.exp(0.5 * h) * x - (jnp.exp(0.5 * h) - 1.0) * x0_s
        x0_mid = _x0_from_D(denoise_fn, x_mid, s_mid, sigma_data)

        # 2M update
        x = jnp.exp(h) * x - (jnp.exp(h) - 1.0) * x0_mid

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
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))
        # x0 at s
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)
        if sn == 0.0:
            x = x0_s
            continue
        # 3M midpoints
        s1 = jnp.exp(h / 3.0) * s
        x1 = jnp.exp(h / 3.0) * x - (jnp.exp(h / 3.0) - 1.0) * x0_s
        x0_1 = _x0_from_D(denoise_fn, x1, s1, sigma_data)

        s2 = jnp.exp(2.0 * h / 3.0) * s
        x2 = jnp.exp(2.0 * h / 3.0) * x - (jnp.exp(2.0 * h / 3.0) - 1.0) * x0_1
        x0_2 = _x0_from_D(denoise_fn, x2, s2, sigma_data)

        # 3M update
        x = jnp.exp(h) * x - (jnp.exp(h) - 1.0) * (
            (1.0 / 4.0) * x0_s + (3.0 / 4.0) * x0_2
        )
    return x
