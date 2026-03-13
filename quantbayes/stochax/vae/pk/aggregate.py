# quantbayes/stochax/vae/pk/aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from .utils import inference_copy, clamp_logvar
from .features import FeatureMap


def collect_aggregate_latents(
    vae,
    x_data: jnp.ndarray,
    *,
    key: jr.PRNGKey,
    batch_size: int = 256,
    num_samples: Optional[int] = None,
    use_mu: bool = False,
    logvar_clamp_range: Tuple[float, float] = (-10.0, 10.0),
) -> jnp.ndarray:
    """
    Collect z samples from the aggregated posterior: z ~ q(z|x) over x_data.
    Returns (N,D). If num_samples is None, uses full dataset.
    """
    vae_eval = inference_copy(vae)
    N = (
        x_data.shape[0]
        if num_samples is None
        else int(min(num_samples, x_data.shape[0]))
    )
    x = x_data[:N]
    lo, hi = logvar_clamp_range

    zs = []
    for start in range(0, N, batch_size):
        xb = x[start : start + batch_size]
        key, k1, k2 = jr.split(key, 3)
        mu, logvar = vae_eval.encoder(xb, rng=k1, train=False)
        logvar = clamp_logvar(logvar, lo, hi)
        if use_mu:
            z = mu
        else:
            eps = jr.normal(k2, mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
        zs.append(z)
    return jnp.concatenate(zs, axis=0)


def collect_aggregate_features(
    vae,
    x_data: jnp.ndarray,
    feature_map: FeatureMap,
    *,
    key: jr.PRNGKey,
    batch_size: int = 256,
    num_samples: Optional[int] = None,
    use_mu: bool = False,
) -> jnp.ndarray:
    z = collect_aggregate_latents(
        vae,
        x_data,
        key=key,
        batch_size=batch_size,
        num_samples=num_samples,
        use_mu=use_mu,
    )
    return feature_map(z)
