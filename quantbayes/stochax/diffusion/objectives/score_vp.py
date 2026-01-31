# quantbayes/stochax/diffusion/objectives/score_vp.py
from __future__ import annotations
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr


def _single_loss(model, int_beta, weight, x, t, key):
    # VP mixture forward data -> y ~ p_t(y|x)
    mean = x * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1.0 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    n = jr.normal(key, x.shape)
    y = mean + std * n
    pred = model(t, y, key=key)  # score-prediction by default
    # Fisher-weighted score-matching loss
    return weight(t) * jnp.mean((pred + n / std) ** 2)


def batch_loss(model, weight_fn, int_beta_fn, data, t1, key):
    b = data.shape[0]
    k_t, k_noise = jr.split(key)
    t = jr.uniform(k_t, (b,), minval=0.0, maxval=t1 / b)
    t = t + (t1 / b) * jnp.arange(b)
    part = ft.partial(_single_loss, model, int_beta_fn, weight_fn)
    keys = jr.split(k_noise, b)
    losses = jax.vmap(part, in_axes=(0, 0, 0))(data, t, keys)
    return jnp.mean(losses)
