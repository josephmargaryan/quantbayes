# quantbayes/stochax/pkstruct/samplers.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.pkstruct.typing import Array, PRNGKey
from quantbayes.pkstruct.utils.angles import wrap_angle


@dataclass(frozen=True)
class ULAConfig:
    step_size: float
    num_steps: int
    burn_in: int = 0
    thin: int = 1


@dataclass(frozen=True)
class MALAConfig:
    step_size: float
    num_steps: int
    burn_in: int = 0
    thin: int = 1
    wrap_k: int | None = None
    wrap_tol: float = 1e-12


def _default_wrap_k(step_size: float, tol: float) -> int:
    """
    Choose truncation K for wrapped normal sum.
    Term at image k roughly scales like exp(-(2πk)^2/(4ε)).
    """
    eps = float(step_size)
    if eps <= 0.0:
        return 0
    # Solve exp(-(2πk)^2/(4ε)) < tol
    val = (4.0 * eps * jnp.log(1.0 / tol)) / ((2.0 * jnp.pi) ** 2)
    k = int(jnp.ceil(jnp.sqrt(jnp.maximum(val, 0.0))))
    return max(1, min(k, 10))


def wrapped_normal_logpdf(x: Array, mean: Array, var: Array, K: int) -> Array:
    """
    Factorized wrapped normal on (-pi, pi]^D:
      q(x|mean) = ∏_d Σ_{k=-K..K} Normal(x_d + 2πk | mean_d, var)
    """
    x = jnp.asarray(x)
    mean = jnp.asarray(mean, dtype=x.dtype)
    var = jnp.asarray(var, dtype=x.dtype)

    shifts = (2.0 * jnp.pi) * jnp.arange(-K, K + 1, dtype=x.dtype)  # (2K+1,)
    x_shift = x[None, :] + shifts[:, None]  # (2K+1, D)

    log_norm = -0.5 * jnp.log(2.0 * jnp.pi * var)
    log_terms = log_norm - 0.5 * ((x_shift - mean[None, :]) ** 2) / var  # (2K+1, D)

    logpdf_per_dim = jax.scipy.special.logsumexp(log_terms, axis=0)  # (D,)
    return jnp.sum(logpdf_per_dim)


# IMPORTANT: energy_fn is static (Python callable), otherwise JAX errors.
@partial(jax.jit, static_argnames=("energy_fn",))
def ula_step(
    z: Array, key: PRNGKey, step_size: float, energy_fn
) -> tuple[Array, PRNGKey, Array]:
    U, g = jax.value_and_grad(energy_fn)(z)
    key, sub = jr.split(key)
    noise = jr.normal(sub, shape=z.shape, dtype=z.dtype)
    z = z - step_size * g + jnp.sqrt(2.0 * step_size) * noise
    z = wrap_angle(z)
    return z, key, U


# IMPORTANT: both energy_fn and K must be static (K affects array shapes).
@partial(jax.jit, static_argnames=("energy_fn", "K"))
def mala_step_wrapped(
    z: Array,
    key: PRNGKey,
    step_size: float,
    energy_fn,
    K: int,
) -> tuple[Array, PRNGKey, Array, Array]:
    """
    Wrapped-normal MALA on the torus with truncation K.
    Returns: z_new, key, accept(0/1), U_new
    """
    U, g = jax.value_and_grad(energy_fn)(z)

    mean_fwd = z - step_size * g
    key, sub = jr.split(key)
    noise = jr.normal(sub, shape=z.shape, dtype=z.dtype)
    z_prop = wrap_angle(mean_fwd + jnp.sqrt(2.0 * step_size) * noise)

    U_prop, g_prop = jax.value_and_grad(energy_fn)(z_prop)
    mean_rev = z_prop - step_size * g_prop

    var = jnp.asarray(2.0 * step_size, dtype=z.dtype)
    log_q_fwd = wrapped_normal_logpdf(z_prop, mean_fwd, var, K)
    log_q_rev = wrapped_normal_logpdf(z, mean_rev, var, K)

    log_acc = (-U_prop + U) + (log_q_rev - log_q_fwd)
    key, sub = jr.split(key)
    accept = (jnp.log(jr.uniform(sub, (), dtype=z.dtype)) < log_acc).astype(z.dtype)

    z_new = jax.lax.select(accept.astype(bool), z_prop, z)
    U_new = jax.lax.select(accept.astype(bool), U_prop, U)
    return z_new, key, accept, U_new


def run_ula(key: PRNGKey, z0: Array, cfg: ULAConfig, energy_fn) -> Array:
    """
    Run ULA, return all states (num_steps, D). Use thin_chain() to subsample.
    """
    z = z0
    zs = []
    for _ in range(int(cfg.num_steps)):
        z, key, _ = ula_step(z, key, float(cfg.step_size), energy_fn)
        zs.append(z)
    return jnp.stack(zs, axis=0)


def run_mala_wrapped(
    key: PRNGKey, z0: Array, cfg: MALAConfig, energy_fn
) -> tuple[Array, float]:
    """
    Run wrapped-normal MALA, return (states, acceptance_rate).
    """
    K = cfg.wrap_k
    if K is None:
        K = _default_wrap_k(cfg.step_size, cfg.wrap_tol)

    z = z0
    zs = []
    accs = []
    for _ in range(int(cfg.num_steps)):
        z, key, acc, _ = mala_step_wrapped(
            z, key, float(cfg.step_size), energy_fn, int(K)
        )
        zs.append(z)
        accs.append(acc)
    zs = jnp.stack(zs, axis=0)
    acc_rate = float(jnp.mean(jnp.stack(accs)))
    return zs, acc_rate


def thin_chain(zs: Array, burn_in: int = 0, thin: int = 1) -> Array:
    zs = jnp.asarray(zs)
    if burn_in < 0 or thin <= 0:
        raise ValueError("burn_in must be >=0 and thin must be >=1")
    return zs[burn_in::thin]
