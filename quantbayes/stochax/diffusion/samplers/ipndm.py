# quantbayes/stochax/diffusion/samplers/ipndm.py
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
    last = len(sigmas) - 2

    g_prev = None
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        if i == last:
            x = x0_s
            continue

        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))
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

    Startup:
      step 0: Euler
      step 1: AB2
      step 2: AB3
      steps>=3: AB4
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    x = jr.normal(key, shape) * sigmas[0]
    last = len(sigmas) - 2

    g_hist = []  # store up to last 3 derivatives g
    for i in range(len(sigmas) - 1):
        s, sn = sigmas[i], sigmas[i + 1]
        x0_s = _x0_from_D(denoise_fn, x, s, sigma_data)

        if i == last:
            x = x0_s
            continue

        h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))
        g_s = x - x0_s
        k = len(g_hist)

        if k == 0:
            x = x + h * g_s
        elif k == 1:
            x = x + h * (1.5 * g_s - 0.5 * g_hist[-1])
        elif k == 2:
            x = x + h * (
                (23.0 / 12.0) * g_s
                - (4.0 / 3.0) * g_hist[-1]
                + (5.0 / 12.0) * g_hist[-2]
            )
        else:
            x = x + h * (
                (55.0 / 24.0) * g_s
                - (59.0 / 24.0) * g_hist[-1]
                + (37.0 / 24.0) * g_hist[-2]
                - (9.0 / 24.0) * g_hist[-3]
            )

        g_hist.append(g_s)
        if len(g_hist) > 3:
            g_hist.pop(0)

    return x


# ---------------------------------------------------------------------
# Batched + JIT iPNDM (AB2) and iPNDM4 (AB4)
# ---------------------------------------------------------------------
def make_ipndm_sampler(
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
    Batched + JIT AB2 in t=log(sigma).
      - step0 Euler, then AB2 thereafter
      - last step snaps to x0
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    last = sigmas.shape[0] - 2
    sd = float(sigma_data)

    def x0_hat(x_state: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return _x0_from_D(denoise_fn, x_state, sigma, sd)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]
        g_prev = jnp.zeros_like(x)
        have_prev = jnp.asarray(0, dtype=jnp.int32)  # 0 for first step, then 1

        def body(i, carry):
            x_state, g_p, hp = carry
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0_s = x0_hat(x_state, s)

            def do_final(_):
                return (x0_s, g_p, hp)

            def do_step(_):
                h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))
                g_s = x_state - x0_s

                def euler(_):
                    return x_state + h * g_s

                def ab2(_):
                    return x_state + h * (1.5 * g_s - 0.5 * g_p)

                x_next = jax.lax.cond(hp == 0, euler, ab2, operand=None)
                return (x_next, g_s, jnp.asarray(1, dtype=jnp.int32))

            return jax.lax.cond(i == last, do_final, do_step, operand=None)

        x, g_prev, have_prev = jax.lax.fori_loop(
            0, sigmas.shape[0] - 1, body, (x, g_prev, have_prev)
        )
        return x

    return sample


def make_ipndm4_sampler(
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
    Batched + JIT AB4 in t=log(sigma) with standard startup:
      - step0 Euler
      - step1 AB2
      - step2 AB3
      - step>=3 AB4
      - last step snaps to x0
    """
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, include_zero=True)
    last = sigmas.shape[0] - 2
    sd = float(sigma_data)

    def x0_hat(x_state: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return _x0_from_D(denoise_fn, x_state, sigma, sd)

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        g1 = jnp.zeros_like(x)  # g_{n-1}
        g2 = jnp.zeros_like(x)  # g_{n-2}
        g3 = jnp.zeros_like(x)  # g_{n-3}
        count = jnp.asarray(
            0, dtype=jnp.int32
        )  # how many history terms are valid (0..3)

        def body(i, carry):
            x_state, g1_, g2_, g3_, cnt = carry
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0_s = x0_hat(x_state, s)

            def do_final(_):
                return (x0_s, g1_, g2_, g3_, cnt)

            def do_step(_):
                h = jnp.log(jnp.maximum(sn, 1e-12)) - jnp.log(jnp.maximum(s, 1e-12))
                g_s = x_state - x0_s

                def euler(_):
                    return x_state + h * g_s

                def ab2(_):
                    return x_state + h * (1.5 * g_s - 0.5 * g1_)

                def ab3(_):
                    return x_state + h * (
                        (23.0 / 12.0) * g_s - (4.0 / 3.0) * g1_ + (5.0 / 12.0) * g2_
                    )

                def ab4(_):
                    return x_state + h * (
                        (55.0 / 24.0) * g_s
                        - (59.0 / 24.0) * g1_
                        + (37.0 / 24.0) * g2_
                        - (9.0 / 24.0) * g3_
                    )

                x_next = jax.lax.cond(
                    cnt == 0,
                    euler,
                    lambda __: jax.lax.cond(
                        cnt == 1,
                        ab2,
                        lambda ___: jax.lax.cond(cnt == 2, ab3, ab4, operand=None),
                        operand=None,
                    ),
                    operand=None,
                )

                # shift history
                g3n = g2_
                g2n = g1_
                g1n = g_s
                cntn = jnp.minimum(cnt + 1, 3).astype(jnp.int32)

                return (x_next, g1n, g2n, g3n, cntn)

            return jax.lax.cond(i == last, do_final, do_step, operand=None)

        x, g1, g2, g3, count = jax.lax.fori_loop(
            0, sigmas.shape[0] - 1, body, (x, g1, g2, g3, count)
        )
        return x

    return sample
