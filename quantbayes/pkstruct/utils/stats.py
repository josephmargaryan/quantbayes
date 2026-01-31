# quantbayes/pkstruct/utils/stats.py
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..typing import Array


def log_vonmises(theta: Array, mu: Array, kappa: Array) -> Array:
    """
    Log pdf of Von Mises:
      f(θ) = exp(κ cos(θ-μ)) / (2π I0(κ))

    Uses stable log(I0(κ)) via i0e: I0(κ) = i0e(κ) * exp(|κ|).
    """
    theta = jnp.asarray(theta)
    mu = jnp.asarray(mu, dtype=theta.dtype)
    kappa = jnp.asarray(kappa, dtype=theta.dtype)

    # i0e is exp(-|k|) I0(k); for kappa>=0: log I0(k) = log i0e(k) + k
    logI0 = jnp.log(jax.scipy.special.i0e(kappa)) + kappa
    return kappa * jnp.cos(theta - mu) - jnp.log(2.0 * jnp.pi) - logI0


def log_beta_pdf(x: Array, alpha: Array, beta: Array) -> Array:
    """
    Log pdf of Beta(alpha, beta) on x in (0,1).
    """
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha, dtype=x.dtype)
    beta = jnp.asarray(beta, dtype=x.dtype)

    return (
        (alpha - 1.0) * jnp.log(x)
        + (beta - 1.0) * jnp.log1p(-x)
        - jax.scipy.special.betaln(alpha, beta)
    )


def log_scaled_beta_pdf(
    r: Array, alpha: Array, beta: Array, N: int, eps: float = 1e-12
) -> Array:
    """
    If x ~ Beta(alpha,beta) on (0,1) and r = N x, then:
      p(r) = (1/N) p(x=r/N)
    """
    r = jnp.asarray(r)
    Nf = jnp.asarray(float(N), dtype=r.dtype)
    r = jnp.clip(r, eps, Nf - eps)
    x = r / Nf
    return log_beta_pdf(x, alpha, beta) - jnp.log(Nf)


def log_gamma_pdf(x: Array, shape: Array, rate: Array) -> Array:
    """
    Log pdf of Gamma(shape=a, rate=b):
      b^a / Γ(a) * x^(a-1) exp(-b x)
    """
    x = jnp.asarray(x)
    shape = jnp.asarray(shape, dtype=x.dtype)
    rate = jnp.asarray(rate, dtype=x.dtype)

    return (
        shape * jnp.log(rate)
        - jax.scipy.special.gammaln(shape)
        + (shape - 1.0) * jnp.log(x)
        - rate * x
    )
