# quantbayes/stochax/robust_inference/simplex.py
from __future__ import annotations
import jax
import jax.numpy as jnp


def project_simplex_row(u: jnp.ndarray) -> jnp.ndarray:
    v = jnp.sort(u)[::-1]
    cssv = jnp.cumsum(v)
    idx = jnp.arange(u.size) + 1
    cond = v - (cssv - 1.0) / idx > 0
    rho = jnp.argmax(jnp.where(cond, idx, 0))
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return jnp.maximum(u - theta, 0.0)


project_rows_to_simplex = jax.vmap(project_simplex_row, in_axes=0, out_axes=0)
