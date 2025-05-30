import equinox as eqx
from typing import Tuple
from jaxtyping import Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.layers import (
    SpectralGRUCell,
    SpectralLSTMCell,
    SpectralMultiheadAttention,
)

__all__ = [
    "GRU",
    "LSTM",
    "SpectralGRUModel",
    "SpectralLSTMModel",
    "SpectralAttentionModel",
]


class GRU(eqx.Module):
    cell: eqx.nn.GRUCell
    fc: eqx.nn.Linear

    def __init__(self, input_dim, hidden_size, key):
        k1, k2 = jr.split(key)
        self.cell = eqx.nn.GRUCell(
            input_size=input_dim, hidden_size=hidden_size, key=k1
        )
        self.fc = eqx.nn.Linear(hidden_size, 1, key=k2)

    def __call__(self, x, state=None, *, key=None):
        # x shape: [seq_len, input_dim]
        init_h = jnp.zeros(self.cell.hidden_size)

        def step(carry, xt):
            new_carry = self.cell(xt, carry)
            return new_carry, None

        final_h, _ = jax.lax.scan(step, init_h, x)
        return self.fc(final_h), state


class LSTM(eqx.Module):
    cell: eqx.nn.LSTMCell
    fc: eqx.nn.Linear

    def __init__(self, input_dim, hidden_size, key):
        k1, k2 = jr.split(key)
        self.cell = eqx.nn.LSTMCell(
            input_size=input_dim, hidden_size=hidden_size, key=k1
        )
        self.fc = eqx.nn.Linear(hidden_size, 1, key=k2)

    def __call__(self, x, state=None, *, key=None):
        # x shape: [seq_len, input_dim]
        # Initialize hidden state and cell state
        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        def step(carry, xt):
            h, c = carry
            new_state = self.cell(xt, (h, c))
            return new_state, None

        final_state, _ = jax.lax.scan(step, init_state, x)
        final_h, final_c = final_state
        return self.fc(final_h), state


class SpectralGRUModel(eqx.Module):
    """
    A simple time series prediction model using the SpectralGRUCell.
    It scans over a sequence and uses the final hidden state to predict the next value.
    """

    cell: SpectralGRUCell
    linear: eqx.nn.Linear  # maps hidden state to output

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        cell_key, lin_key = jr.split(key)
        self.cell = SpectralGRUCell(
            input_size=input_size, hidden_size=hidden_size, key=cell_key
        )
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lin_key)

    def __call__(self, inputs: Array, key, state) -> Array:
        """
        Args:
            inputs: array of shape (seq_len, input_size)
        Returns:
            prediction: scalar prediction (next time step)
        """
        seq_len = inputs.shape[0]
        # initial hidden state
        hidden = jnp.zeros((self.cell.hidden_size,))

        # scan over the sequence
        def step(hidden, x):
            new_hidden = self.cell(x, hidden)
            return new_hidden, new_hidden

        final_hidden, _ = jax.lax.scan(step, hidden, inputs)
        # Use the final hidden state to predict the next value
        pred = self.linear(final_hidden)
        return pred[0], state


class SpectralLSTMModel(eqx.Module):
    """
    A time series prediction model using the SpectralLSTMCell.
    It scans over a sequence and uses the final hidden state to predict the next value.
    """

    cell: SpectralLSTMCell
    linear: eqx.nn.Linear  # Maps hidden state to output

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        cell_key, lin_key = jr.split(key)
        self.cell = SpectralLSTMCell(
            input_size=input_size, hidden_size=hidden_size, key=cell_key
        )
        # Linear layer: from hidden state (hidden_size) to a scalar output.
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lin_key)

    def __call__(self, inputs: Array, key, state) -> Tuple[Array, any]:
        """
        Args:
            inputs: Array of shape (seq_len, input_size)
            key: Random key (ignored in this implementation)
            state: Additional state (passed through unchanged)

        Returns:
            A tuple of (prediction, state), where prediction is a scalar value.
        """
        seq_len = inputs.shape[0]
        # Initialize hidden state: h and c.
        h = jnp.zeros((self.cell.hidden_size,))
        c = jnp.zeros((self.cell.hidden_size,))
        hidden = (h, c)

        # Scan over the sequence.
        def step(hidden, x):
            new_hidden = self.cell(x, hidden)
            return new_hidden, new_hidden

        final_hidden, _ = jax.lax.scan(step, hidden, inputs)
        final_h, _ = final_hidden
        # Use the final hidden state to predict the next value.
        pred = self.linear(final_h)
        return pred[0], state


class SpectralAttentionModel(eqx.Module):
    """
    A simple model that uses the SpectralMultiheadAttention layer.
    It processes a sequence and outputs a scalar prediction based on an aggregated representation.
    """

    attn: SpectralMultiheadAttention
    linear: eqx.nn.Linear  # Maps aggregated representation to scalar output

    def __init__(self, in_features: int, num_heads: int, *, key: PRNGKeyArray):
        attn_key, lin_key = jr.split(key)
        self.attn = SpectralMultiheadAttention(in_features, num_heads, key=attn_key)
        self.linear = eqx.nn.Linear(in_features, 1, key=lin_key)

    def __call__(self, inputs: Array, key, state) -> Tuple[Array, any]:
        """
        Args:
            inputs: Array of shape (seq_len, in_features)
            key: Ignored in this model.
            state: Additional state (passed through unchanged)
        Returns:
            (prediction, state) where prediction is a scalar.
        """
        # Apply spectral attention.
        attn_out = self.attn(inputs)
        # Aggregate across the sequence (using mean).
        agg = jnp.mean(attn_out, axis=0)
        pred = self.linear(agg)
        return pred[0], state
