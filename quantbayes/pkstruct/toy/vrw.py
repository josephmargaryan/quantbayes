# quantbayes/pkstruct/toy/vrw.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..typing import Array
from ..utils.angles import wrap_angle


def vrw_endpoint(theta: Array) -> Array:
    """
    VRW endpoint for angles theta shape (N,).
    steps_i = (cos θ_i, sin θ_i), endpoint = sum steps.
    """
    theta = jnp.asarray(theta)
    steps = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)  # (N,2)
    return jnp.sum(steps, axis=0)  # (2,)


def vrw_r(theta: Array) -> Array:
    """
    Coarse variable r = ||endpoint|| in [0, N].
    (This matches the paper’s support and the Stephens mapping.)
    """
    end = vrw_endpoint(theta)
    return jnp.linalg.norm(end)


@dataclass(frozen=True)
class StephensConfig:
    """
    Stephens approximation (paper Eq. (27)-style):

      u = 2 N γ (1 - r/N)  ~  ChiSquare(df=N-1)

    with:
      1/γ = 1/κ + 3/(8 κ^2)     (paper text)

    IMPORTANT: gamma = 1 / (1/kappa + 3/(8kappa^2))
    """

    kappa: float
    N: int


def stephens_logpdf_r(r: Array, *, cfg: StephensConfig, eps: float = 1e-12) -> Array:
    """
    Log pdf of r under Stephens approximation.
    Uses ChiSquare(df=N-1) = Gamma(df/2, rate=1/2).

    Adds log|du/dr| where u = 2Nγ(1 - r/N) = 2Nγ - 2γ r => |du/dr| = 2γ.
    """
    r = jnp.asarray(r)
    Nf = jnp.asarray(float(cfg.N), dtype=r.dtype)
    r = jnp.clip(r, eps, Nf - eps)

    kappa = jnp.asarray(cfg.kappa, dtype=r.dtype)
    inv_gamma = 1.0 / kappa + 3.0 / (8.0 * kappa**2)
    gamma = 1.0 / inv_gamma

    u = 2.0 * Nf * gamma * (1.0 - r / Nf)
    df = Nf - 1.0

    # Gamma(shape=df/2, rate=1/2)
    from ..utils.stats import log_gamma_pdf

    logpdf_u = log_gamma_pdf(u, shape=df / 2.0, rate=0.5)

    return logpdf_u + jnp.log(2.0 * gamma)
