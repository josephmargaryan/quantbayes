# losses.py
from __future__ import annotations
import jax.numpy as jnp


def kl_std_normal(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """KL(q(z|x)||N(0,1)) per-sample (sum over latent dims)."""
    return 0.5 * jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=-1)


def recon_bernoulli_logits(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """
    Bernoulli negative log-likelihood with logits (no sigmoid in module).
    Returns per-sample reconstruction term (sum over non-batch axes).
    """
    # Stable BCE with logits
    # log(1 + exp(-|logits|)) + max(logits,0) - logits * x
    axes = tuple(range(1, x.ndim))
    return jnp.sum(
        jnp.clip(logits, 0) - logits * x + jnp.log1p(jnp.exp(-jnp.abs(logits))),
        axis=axes,
    )


def recon_gaussian(
    x: jnp.ndarray, mean: jnp.ndarray, logvar: jnp.ndarray | None = None
) -> jnp.ndarray:
    """
    Gaussian negative log-likelihood. If logvar is None, reduces to (const + MSE).
    Returns per-sample reconstruction term (sum over non-batch axes).
    """
    axes = tuple(range(1, x.ndim))
    if logvar is None:
        return jnp.sum((x - mean) ** 2, axis=axes)
    inv = jnp.exp(-logvar)
    return 0.5 * jnp.sum((x - mean) ** 2 * inv + logvar, axis=axes)


def elbo(
    recon_term: jnp.ndarray,
    kl_term: jnp.ndarray,
    *,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> jnp.ndarray:
    """
    Combine recon + beta * KL with optional free-bits floor (applied per-sample).
    free_bits is expressed in nats (e.g., 0.02 * latent_dim).
    """
    if free_bits > 0:
        kl_term = jnp.maximum(kl_term, free_bits)
    return recon_term + beta * kl_term
