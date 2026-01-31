# quantbayes/stochax/diffusion/samplers/edm_heun.py
from __future__ import annotations
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
):
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max)
    x = jr.normal(key, shape) * sigmas[0]
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0, v = _denoise_edm(denoise_fn, x, s, sigma_data)
        x_e = x + (sn - s) * v  # Euler
        if sn == 0.0:
            x = x0
        else:
            x0n, vn = _denoise_edm(denoise_fn, x_e, sn, sigma_data)
            x = x + 0.5 * (sn - s) * (v + vn)  # Heun
    return x
