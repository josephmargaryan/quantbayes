# quantbayes/stochax/federated/robust_agg.py
from __future__ import annotations
from typing import List, Any
import jax, jax.numpy as jnp, equinox as eqx


def coord_median(models: List[eqx.Module]) -> eqx.Module:
    params = [eqx.filter(m, eqx.is_inexact_array) for m in models]

    def med(*leaves):
        stack = jnp.stack(leaves, axis=0)
        return jnp.median(stack, axis=0)

    med_params = jax.tree_util.tree_map(med, *params)
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(med_params, static)


def trimmed_mean(models: List[eqx.Module], trim_ratio: float = 0.1) -> eqx.Module:
    params = [eqx.filter(m, eqx.is_inexact_array) for m in models]
    k = len(models)
    t = int(jnp.floor(trim_ratio * k))

    def tmean(*leaves):
        stack = jnp.sort(jnp.stack(leaves, axis=0), axis=0)
        trimmed = stack[t : k - t] if (k - 2 * t) > 0 else stack
        return jnp.mean(trimmed, axis=0)

    tm_params = jax.tree_util.tree_map(tmean, *params)
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(tm_params, static)
