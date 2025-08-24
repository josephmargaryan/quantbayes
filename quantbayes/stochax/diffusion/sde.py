# quantbayes/stochax/diffusion/sde.py
from __future__ import annotations

import functools as ft
from typing import Callable, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta

# ---------------------------------------------------------------------
# Weighting choices
# Each factory returns w(t), a JAX-scalar function of t.
# ---------------------------------------------------------------------


def make_weight_fn(
    int_beta: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    name: str = "likelihood",  # "likelihood" | "sigma2" | "snr_gamma"
    gamma: float = 0.5,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    name = name.lower()

    if name == "likelihood":
        # w(t) = 1 - exp(-∫β); your original default
        return lambda t: 1.0 - jnp.exp(-int_beta(t))

    if name == "sigma2":
        # same as likelihood for VP-SDE
        return lambda t: 1.0 - jnp.exp(-int_beta(t))

    if name == "snr_gamma":
        # SNR(t) = alpha_bar / (1 - alpha_bar) where alpha_bar = exp(-∫β)
        def _w(t):
            ab = jnp.exp(-int_beta(t))
            snr = ab / jnp.clip(1.0 - ab, 1e-12, 1.0)
            return jnp.power(snr, gamma)

        return _w

    raise ValueError(f"Unknown weight name: {name!r}")


# Back-compat: original simple weight for examples/tests
def weight_fn(t):
    int_beta = make_vp_int_beta()  # defaults: linear, t1=1.0
    return 1.0 - jnp.exp(-int_beta(t))


# ---------------------------------------------------------------------
# Training losses (score matching with MSE on the score)
# Model must output the score sθ(t, y) ≈ ∇_y log p_t(y).
# For VP, the true score is -(y - mean)/var = -(noise/std).
# ---------------------------------------------------------------------


def _mean_var_from_int_beta(
    data: jnp.ndarray, intb_t: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean = data * jnp.exp(-0.5 * intb_t)
    var = jnp.maximum(1.0 - jnp.exp(-intb_t), 1e-5)
    return mean, var


def single_loss_fn(
    model,
    weight: Callable[[jnp.ndarray], jnp.ndarray],
    int_beta: Callable[[jnp.ndarray], jnp.ndarray],
    data: jnp.ndarray,
    t: jnp.ndarray,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """Per-example score loss at time t."""
    intb_t = int_beta(t)
    mean, var = _mean_var_from_int_beta(data, intb_t)
    std = jnp.sqrt(var)

    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    # model must accept (t, y, key=...) — pass the same per-example key.
    pred = model(t, y, key=key)  # score prediction

    # Target score = -(y - mean)/var = -(noise/std)
    target = -(noise / std)
    sq = (pred - target) ** 2
    per_ex_loss = weight(t) * jnp.mean(sq)
    return per_ex_loss


def batch_loss_fn(
    model,
    weight: Callable[[jnp.ndarray], jnp.ndarray],
    int_beta: Callable[[jnp.ndarray], jnp.ndarray],
    data: jnp.ndarray,  # [B, ...]
    t1: float,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """
    Stratified t-sampling in [0, t1], and proper per-example vmap over (data, t, key).
    """
    bsz = data.shape[0]
    tkey, nkey = jr.split(key)

    # stratified uniform sampling over [0, t1)
    # t_i ~ U[i*(t1/B), (i+1)*(t1/B)]
    u = jr.uniform(tkey, (bsz,), minval=0.0, maxval=t1 / bsz)
    t = u + (t1 / bsz) * jnp.arange(bsz)

    nkeys = jr.split(nkey, bsz)

    per_ex = jax.vmap(
        ft.partial(single_loss_fn, model, weight, int_beta),
        in_axes=(0, 0, 0),  # data, t, key are all per-example
    )(data, t, nkeys)
    return jnp.mean(per_ex)


# ---------------------------------------------------------------------
# Probability Flow ODE sampler (Song et al. 2021)
# dx/dt = -0.5 β(t) [ x + score(t, x) ]
# ---------------------------------------------------------------------


@eqx.filter_jit
def single_sample_fn(
    model,
    int_beta: Callable[[jnp.ndarray], jnp.ndarray],
    data_shape: tuple,
    dt0: float,
    t1: float,
    key: jr.PRNGKey,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    solver: dfx.AbstractSolver | None = None,
):
    """Integrate the probability-flow ODE from t=t1 → 0."""
    if solver is None:
        solver = dfx.Tsit5()

    def drift(t, y, args):
        # Recover β(t) = d/dt ∫β via JVP; returns (intb, beta)
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        score = model(
            t, y
        )  # at sampling we call with no key; model should be deterministic
        return -0.5 * beta * (y + score)

    term = dfx.ODETerm(drift)

    # start from standard normal at t1
    y1 = jr.normal(key, data_shape)

    # Integrate from t0=t1 → t1=0 with negative dt
    sol = dfx.diffeqsolve(
        term=term,
        solver=solver,
        t0=t1,
        t1=0.0,
        dt0=-abs(dt0),
        y0=y1,
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
    )
    return sol.ys[-1]
