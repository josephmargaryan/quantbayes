# quantbayes/stochax/vae/pk/utils.py
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def dataloader(
    dataset: jnp.ndarray, batch_size: int, *, key: jr.PRNGKey
) -> Iterator[jnp.ndarray]:
    """Infinite shuffled dataloader for a JAX array dataset."""
    n = dataset.shape[0]
    idx = jnp.arange(n)
    while True:
        key, sub = jr.split(key)
        perm = jr.permutation(sub, idx)
        for i in range(0, n, batch_size):
            bidx = perm[i : i + batch_size]
            yield dataset[bidx]


def inference_copy(model):
    """
    Return a copy of `model` with dropout/etc. set to inference mode.
    Works across Equinox versions.
    """
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


def clamp_logvar(
    logvar: jnp.ndarray, lo: float = -10.0, hi: float = 10.0
) -> jnp.ndarray:
    return jnp.clip(logvar, lo, hi)
