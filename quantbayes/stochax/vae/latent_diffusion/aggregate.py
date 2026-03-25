# quantbayes/stochax/vae/latent_diffusion/aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def _inference_copy(model):
    maybe = eqx.nn.inference_mode(model)
    enter = getattr(maybe, "__enter__", None)
    exit_ = getattr(maybe, "__exit__", None)
    if callable(enter) and callable(exit_):
        try:
            m = enter()
            return m
        finally:
            exit_(None, None, None)
    return maybe


def collect_latents_from_vae(
    vae,
    x_data: jnp.ndarray,
    *,
    key: jr.PRNGKey,
    batch_size: int = 256,
    num_samples: Optional[int] = None,
    use_mu: bool = False,
    logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
) -> jnp.ndarray:
    """
    Collect latents z ~ q(z|x) over dataset (aggregated posterior).
    Returns: (N, latent_dim)
    """
    vae_eval = _inference_copy(vae)

    N = int(
        x_data.shape[0] if num_samples is None else min(num_samples, x_data.shape[0])
    )
    x = x_data[:N]
    lo, hi = logvar_clamp

    zs = []
    for start in range(0, N, batch_size):
        xb = x[start : start + batch_size]
        key, k1, k2 = jr.split(key, 3)
        mu, logvar = vae_eval.encoder(xb, rng=k1, train=False)
        logvar = jnp.clip(logvar, lo, hi)
        if use_mu:
            z = mu
        else:
            eps = jr.normal(k2, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
        zs.append(z)

    return jnp.concatenate(zs, axis=0)


def collect_latents_with_labels_from_vae(
    vae,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray,
    *,
    key: jr.PRNGKey,
    batch_size: int = 256,
    num_samples: Optional[int] = None,
    use_mu: bool = False,
    logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
):
    """
    Returns (z, y) where z ~ q(z|x) aggregated over x_data and y are matching labels.
    """
    N = int(
        x_data.shape[0] if num_samples is None else min(num_samples, x_data.shape[0])
    )
    z = collect_latents_from_vae(
        vae,
        x_data[:N],
        key=key,
        batch_size=batch_size,
        num_samples=None,
        use_mu=use_mu,
        logvar_clamp=logvar_clamp,
    )
    y = y_data[:N].astype(jnp.int32)
    return z, y
