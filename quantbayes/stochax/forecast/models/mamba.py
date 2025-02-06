import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


# --------------------------------------------------------------
# MambaStateSpaceCell
#
# A simple state update cell:
#    h[t+1] = activation( A @ h[t] + B @ x[t] + bias )
# where A and B are learnable matrices.
# --------------------------------------------------------------
class MambaStateSpaceCell(eqx.Module):
    A: jnp.ndarray  # shape (hidden_size, hidden_size)
    B: jnp.ndarray  # shape (hidden_size, input_dim)
    bias: jnp.ndarray  # shape (hidden_size,)
    activation: callable = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_size: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.A = jax.random.normal(k1, (hidden_size, hidden_size)) * 0.1
        self.B = jax.random.normal(k2, (hidden_size, input_dim)) * 0.1
        self.bias = jax.random.normal(k3, (hidden_size,)) * 0.1
        self.activation = jnn.tanh  # You could also try relu, gelu, etc.

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        # x: shape (input_dim,), h: shape (hidden_size,)
        # Compute the new state.
        h_new = self.activation(jnp.dot(self.A, h) + jnp.dot(self.B, x) + self.bias)
        return h_new


# --------------------------------------------------------------
# MambaStateSpaceModel
#
# This module applies the state-space recurrence over a sequence.
# Given an input sequence x (shape: [seq_len, D]), it scans over time with the cell.
# The final state is then mapped to a scalar forecast.
# --------------------------------------------------------------
class MambaStateSpaceModel(eqx.Module):
    cell: MambaStateSpaceCell
    out: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_size: int, *, key):
        # Split keys for the cell and the output layer.
        k1, k2 = jax.random.split(key)
        self.cell = MambaStateSpaceCell(input_dim, hidden_size, key=k1)
        # Map from final state (hidden_size) to scalar forecast.
        self.out = eqx.nn.Linear(hidden_size, 1, key=k2)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input sequence of shape (seq_len, input_dim)
        Returns:
          A scalar forecast (shape (1,)) computed from the final state.
        """
        seq_len = x.shape[0]
        # Initialize the state as zeros.
        h = jnp.zeros((self.hidden_size,))

        # Define the recurrence.
        def step(h, x_t):
            h_new = self.cell(x_t, h)
            return h_new, h_new

        # Use lax.scan to iterate over the sequence.
        _, hs = jax.lax.scan(step, h, x)
        final_state = hs[-1]  # shape: (hidden_size,)
        return self.out(final_state)  # shape: (1,)


# --------------------------------------------------------------
# MambaStateSpaceForecast
#
# This wrapper converts batched inputs to the form expected by the state space model.
# It expects inputs of shape [N, seq_len, D] and applies the state space model
# to each sample via jax.vmap, yielding outputs of shape [N, 1].
# --------------------------------------------------------------
class MambaStateSpaceForecast(eqx.Module):
    model: MambaStateSpaceModel
    seq_len: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, seq_len: int, d: int, hidden_size: int, *, key):
        # The state-space model processes inputs of shape (seq_len, d).
        self.model = MambaStateSpaceModel(d, hidden_size, key=key)
        self.seq_len = seq_len
        self.d = d

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, seq_len, D]
          key: Optional PRNG key.
        Returns:
          Forecasts of shape [N, 1] computed from the final state.
        """
        # We apply the state-space model to each sample in the batch.
        if key is not None:
            # Split the key into one per sample.
            batch_keys = jax.random.split(key, x.shape[0])
        else:
            batch_keys = [None] * x.shape[0]
        return jax.vmap(lambda x_sample, k: self.model(x_sample, key=k))(x, batch_keys)


# --------------------------------------------------------------
# Example usage
# --------------------------------------------------------------
if __name__ == "__main__":
    # For example: a batch of 8 sequences, each of length 20, with 32 features.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Define hyperparameters.
    hidden_size = 64  # Size of the state.

    # Create the MambaStateSpaceForecast model.
    model_key, run_key = jax.random.split(key)
    model = MambaStateSpaceForecast(seq_len, D, hidden_size, key=model_key)

    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected: (8, 1)
