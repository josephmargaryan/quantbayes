# quantbayes/pkdiffusion/analytic_scores.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.parameterizations import vp_alpha_sigma

Array = jax.Array


@dataclass
class AnalyticGaussianVPSScore(eqx.Module):
    """Analytic score for VP forward marginals when x0 ~ N(mean0, cov0)."""

    mean0: Array  # (d,)
    cov0: Array  # (d,d)
    int_beta_fn: Callable = eqx.field(static=True)

    def __call__(
        self,
        t: Array,
        x: Array,
        *,
        key: Array | None = None,
        train: bool | None = None,
    ) -> Array:
        x = jnp.asarray(x)
        mean0 = jnp.asarray(self.mean0, dtype=x.dtype)
        cov0 = jnp.asarray(self.cov0, dtype=x.dtype)

        a, s = vp_alpha_sigma(self.int_beta_fn, t)
        a = jnp.asarray(a, dtype=x.dtype)
        s = jnp.asarray(s, dtype=x.dtype)

        d = x.shape[-1]
        I = jnp.eye(d, dtype=x.dtype)

        # VP marginal: x_t ~ N(a*mean0, a^2*cov0 + s^2*I)
        Sigma = (a * a) * cov0 + (s * s) * I
        mu_t = a * mean0

        # score = ∇_x log N(x; mu_t, Sigma) = - Sigma^{-1}(x - mu_t)
        L = jnp.linalg.cholesky(Sigma)
        y = jax.scipy.linalg.cho_solve((L, True), (x - mu_t))
        return -y
