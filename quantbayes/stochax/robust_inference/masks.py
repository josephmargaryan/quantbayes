# quantbayes/stochax/robust_inference/masks.py
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln


def choose_m_probs(n: int | jnp.ndarray, f: int | jnp.ndarray) -> jnp.ndarray:
    """
    Return p(m) ∝ C(n,m) for m=1..f as a probability vector (length f).
    JIT-safe (vectorized; no Python int conversions).
    """
    m_vals = jnp.arange(1, f + 1, dtype=jnp.float32)  # (f,)
    n_flt = jnp.asarray(n, dtype=jnp.float32)

    # log C(n,m) = log Gamma(n+1) - log Gamma(m+1) - log Gamma(n-m+1)
    logC = (
        gammaln(n_flt + 1.0) - gammaln(m_vals + 1.0) - gammaln((n_flt - m_vals) + 1.0)
    )
    return jax.nn.softmax(logC)  # (f,)


def sample_masks_paper(key: jr.PRNGKey, B: int, n: int, f: int) -> jnp.ndarray:
    """Sample a (B,n,1) mask: first sample m with p(m) ∝ C(n,m), then a subset of size m."""
    p_m = choose_m_probs(n, f)
    k_m, k_s = jr.split(key)
    ms = jr.categorical(k_m, jnp.log(p_m), shape=(B,)) + 1  # (B,)
    keys = jr.split(k_s, B)

    def one(k, m):
        perm = jr.permutation(k, n)  # (n,)
        base = (jnp.arange(n) < m).astype(jnp.float32)  # first m -> 1
        mask = jnp.zeros((n,), jnp.float32).at[perm].set(base)
        return mask[:, None]  # (n,1)

    return jax.vmap(one)(keys, ms)  # (B,n,1)


def sample_masks_fixed(key: jr.PRNGKey, B: int, n: int, f: int) -> jnp.ndarray:
    """Always attack exactly f rows per example; returns (B,n,1) mask."""
    keys = jr.split(key, B)

    def one(k):
        perm = jr.permutation(k, n)
        base = (jnp.arange(n) < f).astype(jnp.float32)
        return jnp.zeros((n,), jnp.float32).at[perm].set(base)[:, None]

    return jax.vmap(one)(keys)  # (B,n,1)
