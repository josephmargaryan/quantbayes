# quantbayes/stochax/diffusion/samplers/ipndm.py
from __future__ import annotations
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0


def _x0_from_D(denoise_fn, x, sigma, sigma_data):
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-8))
    D = denoise_fn(log_sigma, x)
    return edm_denoise_to_x0(x, D, sigma, sigma_data)


def sample_ipndm(
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
    iPNDM-lite: 2-step Adams–Bashforth (AB2) in t=log(sigma).
    ODE: dx/dt = g(x,t) = x - x0(x,sigma)
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    g_prev = None
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]

        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)
        if jnp.isclose(sn, 0.0):
            x = x0_s
            continue

        ls = jnp.log(jnp.maximum(s, 1e-12))
        lsn = jnp.log(jnp.maximum(sn, 1e-12))
        h = lsn - ls

        g_s = x - x0_s
        if g_prev is None:
            x = x + h * g_s
        else:
            x = x + h * (1.5 * g_s - 0.5 * g_prev)
        g_prev = g_s

    return x


def sample_ipndm4(
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
    iPNDM-AB4: 4th-order Adams–Bashforth in t=log(sigma) with standard startup.

    ODE: dx/dt = g(x,t) = x - x0(x,sigma)

    Startup:
      step 0: Euler
      step 1: AB2
      step 2: AB3
      steps>=3: AB4
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    g_hist = []  # store up to last 3 derivatives g
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]

        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)
        if jnp.isclose(sn, 0.0):
            x = x0_s
            continue

        ls = jnp.log(jnp.maximum(s, 1e-12))
        lsn = jnp.log(jnp.maximum(sn, 1e-12))
        h = lsn - ls

        g_s = x - x0_s
        k = len(g_hist)

        if k == 0:
            # Euler
            x = x + h * g_s
        elif k == 1:
            # AB2
            x = x + h * (1.5 * g_s - 0.5 * g_hist[-1])
        elif k == 2:
            # AB3
            x = x + h * (
                (23.0 / 12.0) * g_s
                - (4.0 / 3.0) * g_hist[-1]
                + (5.0 / 12.0) * g_hist[-2]
            )
        else:
            # AB4
            x = x + h * (
                (55.0 / 24.0) * g_s
                - (59.0 / 24.0) * g_hist[-1]
                + (37.0 / 24.0) * g_hist[-2]
                - (9.0 / 24.0) * g_hist[-3]
            )

        # update history
        g_hist.append(g_s)
        if len(g_hist) > 3:
            g_hist.pop(0)

    return x
