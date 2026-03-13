# quantbayes/stochax/robust_inference/probits.py
from __future__ import annotations
import jax.numpy as jnp

Array = jnp.ndarray


def margin(u: Array) -> Array:
    # u: (..., K) probs on simplex
    top2 = jnp.sort(u, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def sigma_x(P: Array) -> Array:
    # P: (n, K) client probits for one x
    Pbar = jnp.mean(P, axis=0, keepdims=True)
    var_per_class = jnp.mean((P - Pbar) ** 2, axis=0)  # (K,)
    return jnp.sqrt(jnp.max(var_per_class) + 1e-12)


def cwtm(P: Array, f: int) -> Array:
    # Coordinate-wise trimmed mean in probit space; renormalize to simplex.
    n, K = P.shape
    assert 2 * f < n, "Need n > 2f."
    vals = jnp.sort(P, axis=0)
    trimmed = vals[f : n - f]
    u = jnp.mean(trimmed, axis=0)
    u = jnp.clip(u, 1e-12, 1.0)
    return u / jnp.sum(u)


def cwmed(P: Array) -> Array:
    # Coordinate-wise median + renormalize.
    u = jnp.median(P, axis=0)
    u = jnp.clip(u, 1e-12, 1.0)
    return u / jnp.sum(u)
