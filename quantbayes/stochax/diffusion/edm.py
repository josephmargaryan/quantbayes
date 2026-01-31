# quantbayes/stochax/diffusion/edm.py
from __future__ import annotations

from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def edm_precond_scalars(
    sigma: jnp.ndarray, sigma_data: float
) -> Tuple[jnp.ndarray, ...]:
    """Compute c_in, c_skip, c_out for EDM preconditioning."""
    sd2 = sigma_data * sigma_data
    s2 = sigma * sigma
    denom = jnp.sqrt(s2 + sd2)
    c_in = 1.0 / denom
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * sigma_data / denom
    return c_in, c_skip, c_out


def edm_lambda_weight(sigma: jnp.ndarray, sigma_data: float) -> jnp.ndarray:
    """EDM recommended weighting λ(σ) = (σ^2 + σ_data^2) / (σ σ_data)^2."""
    sd2 = sigma_data * sigma_data
    s2 = sigma * sigma
    return (s2 + sd2) / jnp.maximum((sigma * sigma_data) ** 2, 1e-20)


def _edm_single_loss(
    model,
    x: jnp.ndarray,
    sigma: jnp.ndarray,
    *,
    sigma_data: float,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """
    Single-sample EDM loss.
    - y = x + sigma * n
    - cond = log(sigma)  (use as 'time' input to model)
    - y_in = c_in * y
    - D = c_skip * y + c_out * model(cond, y_in)
    - loss = λ(σ) * ||D - x||^2
    """
    k_noise, k_model = jr.split(key)
    noise = jr.normal(k_noise, x.shape)
    y = x + sigma * noise
    c_in, c_skip, c_out = edm_precond_scalars(sigma, sigma_data)
    cond = jnp.log(sigma)

    # Pack scalars to match x dims where needed
    def expand(v):
        return jnp.asarray(v, dtype=x.dtype)

    y_in = expand(c_in) * y
    out = model(cond, y_in, key=k_model, train=True)  # <-- enable training path
    denoised = expand(c_skip) * y + expand(c_out) * out

    w = edm_lambda_weight(sigma, sigma_data)
    return w * jnp.mean((denoised - x) ** 2)


def edm_batch_loss(
    model,
    data: jnp.ndarray,  # [B,...]
    key: jr.PRNGKey,
    *,
    sigma_data: float = 0.5,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    sample: str = "uniform",
    p_mean: float = 0.0,
    p_std: float = 1.0,
) -> jnp.ndarray:
    """EDM denoising loss; plug into trainer via custom_loss."""
    b = data.shape[0]
    k_sigma, k_noise = jr.split(key)
    if sample == "uniform":
        rho = jr.uniform(k_sigma, (b,), minval=rho_min, maxval=rho_max)
    elif sample == "normal":
        rho = (
            jr.truncated_normal(k_sigma, lower=rho_min, upper=rho_max, shape=(b,))
            * p_std
            + p_mean
        )
    else:
        raise ValueError(f"Unknown sigma sampling: {sample!r}")
    sigmas = jnp.exp(rho)

    keys = jr.split(k_noise, b)

    # IMPORTANT: map the key as a mapped positional arg
    per_ex = jax.vmap(
        lambda x_i, s_i, k_i: _edm_single_loss(
            model, x_i, s_i, sigma_data=sigma_data, key=k_i
        ),
        in_axes=(0, 0, 0),
        out_axes=0,
    )(data, sigmas, keys)

    return jnp.mean(per_ex)
