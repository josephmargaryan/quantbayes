import equinox as eqx
from typing import Tuple
from jaxtyping import Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import jax.random as jr

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
