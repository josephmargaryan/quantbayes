# quantbayes/stochax/diffusion/samplers/unipc.py
from __future__ import annotations
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
    UniPC (2nd-order) in log-sigma time.

    We treat t = log(sigma) as the integration variable. The ODE becomes:
        dx/dt = x - x0(x, sigma)
    where x0 is obtained from the EDM head D via standard preconditioning.

    This is a predictor-corrector (Heun/RK2-style) in t = log sigma:
      - predictor:  x_e = x + h * g_s,   g_s = x - x0_s
      - corrector:  x   = x + 0.5*h*(g_s + g_n)
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]

        # compute x0 at current level
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        # final snap-to-x0 at sigma -> 0
        if sn == 0.0:
            x = x0_s
            continue

        # step in log-sigma
        ls = jnp.log(jnp.maximum(s, 1e-12))
        lsn = jnp.log(jnp.maximum(sn, 1e-12))
        h = lsn - ls

        # predictor
        g_s = x - x0_s
        x_e = x + h * g_s

        # corrector
        x0_n = _x0_from_D(denoise_fn, x_e, sn, sigma_data)
        g_n = x_e - x0_n
        x = x + 0.5 * h * (g_s + g_n)

    return x
