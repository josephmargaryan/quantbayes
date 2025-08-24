# quantbayes/stochax/diffusion/schedules/karras.py
from __future__ import annotations
import jax.numpy as jnp


def get_sigmas_karras(
    n: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    include_zero: bool = True,
):
    """
    Karras et al. (EDM) noise schedule. Returns decreasing sigmas.
    Args:
      n: number of steps (excludes final zero if include_zero=True).
    """
    ramp = jnp.linspace(0.0, 1.0, n)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho  # [n]
    if include_zero:
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0])], axis=0)  # [n+1]
    return sigmas
