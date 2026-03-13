# quantbayes/stochax/robust_inference/models.py
from __future__ import annotations
from typing import Tuple
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx

Array = jnp.ndarray
PRNG = jax.Array


class ClientMLP(eqx.Module):
    """Single-sample Equinox MLP: (x, key, state) -> (logits, state)."""

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear
    k: int

    def __init__(self, d_in: int, width: int, k: int, key: PRNG):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = eqx.nn.Linear(d_in, width, key=k1)
        self.l2 = eqx.nn.Linear(width, width, key=k2)
        self.l3 = eqx.nn.Linear(width, k, key=k3)
        self.k = int(k)

    def __call__(self, x: Array, key: PRNG, state) -> Tuple[Array, None]:
        h = jax.nn.relu(self.l1(x))
        h = jax.nn.relu(self.l2(h))
        logits = self.l3(h)  # (K,)
        return logits, state  # no mutable state


def make_client_with_state(d_in: int, width: int, k: int, key: PRNG):
    """Construct via make_with_state; returns (model, state)."""
    return eqx.nn.make_with_state(lambda kk: ClientMLP(d_in, width, k, kk))(key)
