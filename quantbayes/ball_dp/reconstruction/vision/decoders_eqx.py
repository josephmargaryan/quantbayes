# quantbayes/ball_dp/reconstruction/vision/decoders_eqx.py
from __future__ import annotations

from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _static_field():
    if hasattr(eqx, "field"):
        return eqx.field(static=True)
    return eqx.static_field()


class DecoderMLP(eqx.Module):
    """
    Generic decoder: embedding -> flat image vector.
    Stochax interface: (x, key, state) -> (pred, state)
    """

    layers: Tuple[eqx.nn.Linear, ...]
    act: str = _static_field()
    out_act: str = _static_field()

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: Tuple[int, ...] = (1024, 1024),
        *,
        key: jr.PRNGKey,
        act: str = "relu",
        out_act: str = "sigmoid",
    ):
        keys = jr.split(key, len(hidden) + 1)
        ls: List[eqx.nn.Linear] = []
        last = int(d_in)
        for i, h in enumerate(hidden):
            ls.append(eqx.nn.Linear(last, int(h), key=keys[i]))
            last = int(h)
        ls.append(eqx.nn.Linear(last, int(d_out), key=keys[-1]))
        self.layers = tuple(ls)
        self.act = str(act).lower()
        self.out_act = str(out_act).lower()

    def _phi(self, x):
        if self.act == "gelu":
            return jax.nn.gelu(x)
        if self.act == "elu":
            return jax.nn.elu(x)
        return jax.nn.relu(x)

    def _out(self, x):
        if self.out_act == "sigmoid":
            return jax.nn.sigmoid(x)
        if self.out_act == "tanh":
            return jnp.tanh(x)
        return x

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, state):
        h = x
        for L in self.layers[:-1]:
            h = self._phi(L(h))
        out = self.layers[-1](h)
        out = self._out(out)
        return out, state


def decode_batch(decoder: DecoderMLP, E: jnp.ndarray) -> jnp.ndarray:
    """
    E: (N,d_in) -> (N,d_out)
    """
    keys = jr.split(jr.PRNGKey(0), E.shape[0])
    out, _ = jax.vmap(lambda e, k: decoder(e, k, None))(E, keys)
    return out


if __name__ == "__main__":
    import numpy as np

    E = jnp.asarray(np.random.default_rng(0).normal(size=(4, 16)).astype(np.float32))
    dec = DecoderMLP(16, 32, hidden=(64,), key=jr.PRNGKey(0))
    Y = decode_batch(dec, E)
    print("decoded shape:", Y.shape)
    assert Y.shape == (4, 32)
    print("[OK] DecoderMLP smoke.")
