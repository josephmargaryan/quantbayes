# quantbayes/stochax/diffusion/samplers/dpm_solver_v3.py
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


def sample_dpmv3(
    denoise_fn,  # callable(log_sigma, x) -> D  (EDM head)
    shape: tuple,
    *,
    key,
    steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    order: int = 3,  # 2 or 3 (single-step RK in log-sigma time)
):
    """
    Minimal DPM-Solver-v3-style single-step integrator in t = log(sigma).

    ODE: dx/dt = g(x,t) = x - x0(x, sigma)
    We integrate from s -> sn (log step h = log(sn) - log(s)) with 2nd- or 3rd-order
    single-step updates using x0 predictions converted from the EDM head.

    order=2:
        midpoint rule in log-sigma (one internal eval)
    order=3:
        two internal points at 1/3 and 2/3 of the log step (three total evals)
    """
    order = int(order)
    if order not in (2, 3):
        raise ValueError("order must be 2 or 3")

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]

    last = len(sigmas) - 2  # last transition is to the appended sigma=0

    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        # JIT-safe final step: snap to x0
        if i == last:
            x = x0_s
            continue

        ls = jnp.log(jnp.maximum(s, 1e-12))
        lsn = jnp.log(jnp.maximum(sn, 1e-12))
        h = lsn - ls

        if order == 2:
            # midpoint at 1/2 h
            eh2 = jnp.exp(0.5 * h)
            x_mid = eh2 * x - (eh2 - 1.0) * x0_s
            x0_mid = _x0_from_D(denoise_fn, x_mid, eh2 * s, sigma_data)

            eh = jnp.exp(h)
            x = eh * x - (eh - 1.0) * x0_mid

        else:
            # order 3: internal points at 1/3 and 2/3 of the log step
            e1 = jnp.exp(h / 3.0)
            x1 = e1 * x - (e1 - 1.0) * x0_s
            x0_1 = _x0_from_D(denoise_fn, x1, e1 * s, sigma_data)

            e2 = jnp.exp(2.0 * h / 3.0)
            x2 = e2 * x - (e2 - 1.0) * x0_1
            x0_2 = _x0_from_D(denoise_fn, x2, e2 * s, sigma_data)

            # 3rd-order composition in log-sigma space
            eh = jnp.exp(h)
            x = eh * x - (eh - 1.0) * (0.25 * x0_s + 0.75 * x0_2)

    return x
