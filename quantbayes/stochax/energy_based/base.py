# quantbayes/stochax/energy_based/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


class BaseEBM(eqx.Module, ABC):
    """
    Base class for EBMs.

    Convention:
      - energy(x) returns a scalar for single sample or (B,) for batch.
      - score(x) returns ∇x log p(x) ≈ -∇x E(x), same shape as x.
    """

    sample_ndim: int = eqx.static_field()

    def _as_batch(self, x: jnp.ndarray):
        x = jnp.asarray(x)
        if x.ndim == self.sample_ndim:
            return x[None, ...], True
        if x.ndim == self.sample_ndim + 1:
            return x, False
        raise ValueError(
            f"Expected x.ndim in {{{self.sample_ndim}, {self.sample_ndim+1}}}, got {x.ndim}"
        )

    @abstractmethod
    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        """x_batched: (B, ...) -> energies: (B,)"""
        raise NotImplementedError

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        xb, squeeze = self._as_batch(x)
        e = self.energy_batch(xb)
        return e[0] if squeeze else e

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.energy(x)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        ∇x log p(x) ≈ -∇x E(x).
        Uses grad of sum energies so we get a full batch gradient in one call.
        """
        xb, squeeze = self._as_batch(x)

        def sum_energy(z):
            return jnp.sum(self.energy_batch(z))

        grad_x = jax.grad(sum_energy)(xb)  # same shape as xb
        sc = -grad_x
        return sc[0] if squeeze else sc


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
class GlobalAvgPool2d(eqx.Module):
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x: (C,H,W)
        return jnp.mean(x, axis=(-2, -1))  # (C,)


# ------------------------------------------------------------
# MLP EBM
# ------------------------------------------------------------
class MLPBasedEBM(BaseEBM):
    sample_ndim: int = eqx.static_field()
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        depth: int,
        *,
        key: jr.PRNGKey,
        activation: Callable = jnn.relu,
    ):
        self.sample_ndim = 1
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )

    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        e = jax.vmap(self.mlp)(x_batched)  # (B,1)
        return jnp.squeeze(e, axis=-1)  # (B,)


# ------------------------------------------------------------
# Conv EBM (NCHW)
# ------------------------------------------------------------
class ConvEBM(BaseEBM):
    sample_ndim: int = eqx.static_field()
    net: eqx.nn.Sequential

    def __init__(
        self,
        *,
        key: jr.PRNGKey,
        in_channels: int = 1,
        hidden_channels: int = 64,
    ):
        self.sample_ndim = 3
        k1, k2, k3 = jr.split(key, 3)
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding="SAME",
                    key=k1,
                ),
                lambda x, **kw: jnn.silu(x),
                eqx.nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding="SAME",
                    key=k2,
                ),
                lambda x, **kw: jnn.silu(x),
                GlobalAvgPool2d(),
                eqx.nn.Linear(hidden_channels, 1, use_bias=True, key=k3),
            ]
        )

    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        # x_batched: (B,C,H,W); vmap over images -> (B,1)
        e = jax.vmap(self.net)(x_batched)
        return jnp.squeeze(e, axis=-1)  # (B,)


# ------------------------------------------------------------
# Sequence EBMs (GRU / LSTM) – batch-vmap outside scan (faster)
# ------------------------------------------------------------
class RNNBasedEBM(BaseEBM):
    sample_ndim: int = eqx.static_field()
    gru: eqx.nn.GRUCell
    proj: eqx.nn.Linear
    hidden_size: int = eqx.static_field()

    def __init__(self, input_size: int, hidden_size: int, *, key: jr.PRNGKey):
        self.sample_ndim = 2
        self.hidden_size = int(hidden_size)
        k1, k2 = jr.split(key, 2)
        self.gru = eqx.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, key=k1
        )
        self.proj = eqx.nn.Linear(hidden_size, 1, use_bias=True, key=k2)

    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        # x_batched: (B,T,D)
        def run_one(seq):
            h0 = jnp.zeros((self.hidden_size,), dtype=seq.dtype)

            def step(h, x_t):
                h = self.gru(x_t, h)
                return h, None

            hT, _ = jax.lax.scan(step, h0, seq)  # (hidden,)
            return hT

        h = jax.vmap(run_one)(x_batched)  # (B,H)
        e = jax.vmap(self.proj)(h)  # (B,1)
        return jnp.squeeze(e, axis=-1)


class LSTMBasedEBM(BaseEBM):
    sample_ndim: int = eqx.static_field()
    lstm: eqx.nn.LSTMCell
    proj: eqx.nn.Linear
    hidden_size: int = eqx.static_field()

    def __init__(self, input_size: int, hidden_size: int, *, key: jr.PRNGKey):
        self.sample_ndim = 2
        self.hidden_size = int(hidden_size)
        k1, k2 = jr.split(key, 2)
        self.lstm = eqx.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, key=k1
        )
        self.proj = eqx.nn.Linear(hidden_size, 1, use_bias=True, key=k2)

    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        # x_batched: (B,T,D)
        def run_one(seq):
            h0 = jnp.zeros((self.hidden_size,), dtype=seq.dtype)
            c0 = jnp.zeros((self.hidden_size,), dtype=seq.dtype)
            state0 = (h0, c0)

            def step(state, x_t):
                state = self.lstm(x_t, state)
                return state, None

            (hT, _), _ = jax.lax.scan(step, state0, seq)
            return hT

        h = jax.vmap(run_one)(x_batched)  # (B,H)
        e = jax.vmap(self.proj)(h)  # (B,1)
        return jnp.squeeze(e, axis=-1)


class AttentionBasedEBM(BaseEBM):
    sample_ndim: int = eqx.static_field()
    attn: eqx.nn.MultiheadAttention
    proj: eqx.nn.Linear
    pos_embed: jnp.ndarray
    max_seq_len: int = eqx.static_field()

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        max_seq_len: int,
        *,
        key: jr.PRNGKey,
    ):
        self.sample_ndim = 2
        self.max_seq_len = int(max_seq_len)
        k1, k2, k3 = jr.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=input_size, key=k1
        )
        self.proj = eqx.nn.Linear(input_size, 1, use_bias=True, key=k2)
        self.pos_embed = jr.normal(k3, (self.max_seq_len, input_size))

    def energy_batch(self, x_batched: jnp.ndarray) -> jnp.ndarray:
        # x_batched: (B,T,D)
        b, t, d = x_batched.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len {t} > max_seq_len {self.max_seq_len}")

        pe = self.pos_embed[:t][None, :, :]  # (1,T,D)
        x = x_batched + pe

        def one(seq):
            y = self.attn(seq, seq, seq)  # (T,D)
            pooled = jnp.mean(y, axis=0)  # (D,)
            return self.proj(pooled)[0]  # scalar

        return jax.vmap(one)(x)  # (B,)
