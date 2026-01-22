from __future__ import annotations

import jax.numpy as jnp

from ..typing import Array


def wrap_angle(theta: Array) -> Array:
    """
    Wrap angles to (-pi, pi]. JAX-friendly.
    """
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=theta.dtype)
    return (theta + jnp.pi) % two_pi - jnp.pi
