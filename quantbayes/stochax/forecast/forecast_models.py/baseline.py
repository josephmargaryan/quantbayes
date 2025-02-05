import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

# -------------------------------------------------------------------
# GRU Baseline Model (single-sample version)
#
# Uses eqx.nn.GRUCell to process a sequence and maps the final hidden state
# to a scalar prediction.
# -------------------------------------------------------------------
class GRUBaseline(eqx.Module):
    cell: eqx.nn.GRUCell
    final_linear: eqx.nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, *, key):
        cell_key, linear_key = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_size, hidden_size, key=cell_key)
        self.final_linear = eqx.nn.Linear(hidden_size, 1, key=linear_key)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape (seq_len, input_size)
        init_state = jnp.zeros((self.hidden_size,))
        def step(state, x_t):
            new_state = self.cell(x_t, state)
            # new_state is a Jax array (the new hidden state)
            return new_state, new_state
        final_state, _ = jax.lax.scan(step, init_state, x)
        return self.final_linear(final_state)  # shape (1,)

# -------------------------------------------------------------------
# LSTM Baseline Model (single-sample version)
#
# Uses eqx.nn.LSTMCell to process a sequence and maps the final hidden state
# to a scalar prediction.
# -------------------------------------------------------------------
class LSTMBaseline(eqx.Module):
    cell: eqx.nn.LSTMCell
    final_linear: eqx.nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, *, key):
        cell_key, linear_key = jax.random.split(key)
        self.cell = eqx.nn.LSTMCell(input_size, hidden_size, key=cell_key)
        self.final_linear = eqx.nn.Linear(hidden_size, 1, key=linear_key)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape (seq_len, input_size)
        init_state = (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
        # Corrected scan function: the cell returns a single tuple (h, c),
        # and we output the hidden state (h).
        def step(state, x_t):
            new_state = self.cell(x_t, state)
            return new_state, new_state[0]  # new_state[0] is h
        _, h_seq = jax.lax.scan(step, init_state, x)
        final_hidden = h_seq[-1]
        return self.final_linear(final_hidden)  # shape (1,)

# -------------------------------------------------------------------
# Batch Wrappers
#
# These modules apply the single-sample models to each sample in the batch using jax.vmap.
# They accept input of shape [N, seq_len, D] and produce output [N, 1].
# -------------------------------------------------------------------
class GRUBaselineForecast(eqx.Module):
    model: GRUBaseline
    seq_len: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, seq_len: int, d: int, hidden_size: int, *, key):
        self.model = GRUBaseline(d, hidden_size, key=key)
        self.seq_len = seq_len
        self.d = d

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape [N, seq_len, d]
        batch_size = x.shape[0]
        if key is not None:
            batch_keys = jax.random.split(key, batch_size)
        else:
            batch_keys = [None] * batch_size
        return jax.vmap(lambda sample, k: self.model(sample, key=k))(x, batch_keys)

class LSTMBaselineForecast(eqx.Module):
    model: LSTMBaseline
    seq_len: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, seq_len: int, d: int, hidden_size: int, *, key):
        self.model = LSTMBaseline(d, hidden_size, key=key)
        self.seq_len = seq_len
        self.d = d

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape [N, seq_len, d]
        batch_size = x.shape[0]
        if key is not None:
            batch_keys = jax.random.split(key, batch_size)
        else:
            batch_keys = [None] * batch_size
        return jax.vmap(lambda sample, k: self.model(sample, key=k))(x, batch_keys)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == '__main__':
    # For example: A batch of 8 sequences, each of length 20, with 32 features.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))
    hidden_size = 64  # adjust as desired

    # Build and run the GRU baseline model.
    key_gru, key_lstm, key_run = jax.random.split(key, 3)
    gru_model = GRUBaselineForecast(seq_len, D, hidden_size, key=key_gru)
    gru_preds = gru_model(x, key=key_run)
    print("GRU Baseline predictions shape:", gru_preds.shape)  # Expected: (8, 1)

    # Build and run the LSTM baseline model.
    lstm_model = LSTMBaselineForecast(seq_len, D, hidden_size, key=key_lstm)
    lstm_preds = lstm_model(x, key=key_run)
    print("LSTM Baseline predictions shape:", lstm_preds.shape)  # Expected: (8, 1)
