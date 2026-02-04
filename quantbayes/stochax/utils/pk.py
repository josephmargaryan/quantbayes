# quantbayes/stochax/utils/pk.py
"""
Probability kinematics utilities (JAX-friendly).

We operate on samples of a low-dimensional observable Y=F(X).
Given:
  - samples y_i ~ p_Y  (induced by your "prior" over X)
  - a target evidence density q(y) (logpdf provided)
we compute PK weights:
  w_i ∝ q(y_i) / p_Y(y_i)

We estimate p_Y with a simple 1D Gaussian KDE (optionally leave-one-out at samples).
This is intentionally small and production-friendly for low-d evidence spaces.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

Array = jnp.ndarray

_LOG_2PI = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))


def normal_logpdf(y: Array, mu: Array, sigma: Array) -> Array:
    """Log N(mu, sigma^2) evaluated at y. Broadcasts."""
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    mu = jnp.asarray(mu, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    z = (y - mu) / (sigma + 1e-12)
    return -0.5 * (z * z) - jnp.log(sigma + 1e-12) - 0.5 * _LOG_2PI


def silverman_bandwidth_1d(y: Array, min_bw: float = 1e-3) -> Array:
    """
    Silverman's rule of thumb bandwidth for 1D KDE.

    h = 0.9 * min(std, IQR/1.34) * n^{-1/5}
    """
    y = jnp.asarray(y, dtype=jnp.float32).reshape(-1)
    n = y.shape[0]
    std = jnp.std(y) + 1e-12
    q25, q75 = jnp.quantile(y, jnp.array([0.25, 0.75], dtype=jnp.float32))
    iqr = (q75 - q25) + 1e-12
    s = jnp.minimum(std, iqr / 1.34)
    h = 0.9 * s * (n ** (-0.2))
    return jnp.maximum(h, jnp.asarray(min_bw, dtype=jnp.float32))


def gaussian_kde_logpdf_1d(y_eval: Array, y_samples: Array, bw: Array) -> Array:
    """
    Standard 1D Gaussian KDE logpdf:
      p(y) = (1/N) Σ_i N(y | y_i, bw^2)

    y_eval: (M,)
    y_samples: (N,)
    bw: scalar
    returns: (M,)
    """
    y_eval = jnp.asarray(y_eval, dtype=jnp.float32).reshape(-1)
    y_samples = jnp.asarray(y_samples, dtype=jnp.float32).reshape(-1)
    bw = jnp.asarray(bw, dtype=jnp.float32)

    # (M, N)
    z = (y_eval[:, None] - y_samples[None, :]) / (bw + 1e-12)
    log_kernel = -0.5 * (z * z) - 0.5 * _LOG_2PI - jnp.log(bw + 1e-12)
    # log mean exp
    return logsumexp(log_kernel, axis=1) - jnp.log(y_samples.shape[0])


def gaussian_kde_logpdf_1d_loo_at_samples(y_samples: Array, bw: Array) -> Array:
    """
    Leave-one-out KDE logpdf evaluated at the sample locations:
      p_{-i}(y_i) = (1/(N-1)) Σ_{j≠i} N(y_i | y_j, bw^2)

    returns: (N,)
    """
    y = jnp.asarray(y_samples, dtype=jnp.float32).reshape(-1)
    n = y.shape[0]
    bw = jnp.asarray(bw, dtype=jnp.float32)

    # (N,N)
    z = (y[:, None] - y[None, :]) / (bw + 1e-12)
    log_kernel = -0.5 * (z * z) - 0.5 * _LOG_2PI - jnp.log(bw + 1e-12)

    # exclude diagonal
    log_kernel = log_kernel.at[jnp.arange(n), jnp.arange(n)].set(-jnp.inf)
    return logsumexp(log_kernel, axis=1) - jnp.log(
        jnp.asarray(n - 1, dtype=jnp.float32)
    )


def pk_weights_from_y_samples_1d(
    y_samples: Array,
    log_q: Callable[[Array], Array],
    *,
    bandwidth: Optional[float] = None,
    min_bw: float = 1e-3,
    leave_one_out: bool = True,
    max_logw: float = 60.0,
) -> Tuple[Array, Array, Array]:
    """
    Compute PK weights for samples y_i ~ p_Y using:
      w_i ∝ q(y_i)/p_Y(y_i)

    We estimate p_Y by 1D Gaussian KDE.

    Returns:
      weights: (N,) normalized
      logw: (N,) unnormalized log-weights
      logp_hat: (N,) KDE logpdf at samples (LOO or standard)
    """
    y = jnp.asarray(y_samples, dtype=jnp.float32).reshape(-1)

    bw = (
        silverman_bandwidth_1d(y, min_bw=min_bw)
        if bandwidth is None
        else jnp.asarray(bandwidth, jnp.float32)
    )
    logp_hat = (
        gaussian_kde_logpdf_1d_loo_at_samples(y, bw)
        if leave_one_out
        else gaussian_kde_logpdf_1d(y, y, bw)
    )

    logq = jnp.asarray(log_q(y), dtype=jnp.float32).reshape(-1)
    logw = logq - logp_hat

    # stabilize
    logw = jnp.clip(logw, a_min=-max_logw, a_max=max_logw)
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / (jnp.sum(w) + 1e-12)
    return w, logw, logp_hat


def effective_sample_size(weights: Array) -> Array:
    """ESS = 1 / Σ w_i^2 for normalized weights."""
    w = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    return 1.0 / (jnp.sum(w * w) + 1e-12)


def resample_indices(
    key: jr.PRNGKey,
    weights: Array,
    n: int,
    *,
    replace: bool = True,
) -> Array:
    """Resample indices according to weights."""
    w = jnp.asarray(weights, dtype=jnp.float32).reshape(-1)
    idx = jnp.arange(w.shape[0])
    return jr.choice(key, idx, shape=(int(n),), replace=replace, p=w)
