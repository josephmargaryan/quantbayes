import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial


# -------------------------------------------------------------------
# A simple LSTM cell (adapted from the Equinox example)
# -------------------------------------------------------------------
class LSTMCell(eqx.Module):
    weight_ih: jnp.ndarray
    weight_hh: jnp.ndarray
    bias: jnp.ndarray
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self, input_size: int, hidden_size: int, *, key, use_bias: bool = True
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # For simplicity, we use a standard Glorot initialization.
        k1, k2, k3 = jax.random.split(key, 3)
        w_shape = (4 * hidden_size, input_size)
        self.weight_ih = jax.random.normal(k1, w_shape) * (1.0 / jnp.sqrt(input_size))
        self.weight_hh = jax.random.normal(k2, (4 * hidden_size, hidden_size)) * (
            1.0 / jnp.sqrt(hidden_size)
        )
        self.bias = (
            jax.random.normal(k3, (4 * hidden_size,))
            if use_bias
            else jnp.zeros((4 * hidden_size,))
        )

    def __call__(
        self, x: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        # x: shape (input_size,)
        # state: tuple of (hidden, cell) each shape (hidden_size,)
        h, c = state
        gates = self.weight_ih @ x + self.weight_hh @ h
        if self.use_bias:
            gates = gates + self.bias
        i, f, g, o = jnp.split(gates, 4)
        i = jnn.sigmoid(i)
        f = jnn.sigmoid(f)
        g = jnp.tanh(g)
        o = jnn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


# -------------------------------------------------------------------
# LSTM Encoder: processes a sequence and returns the sequence of hidden states.
# -------------------------------------------------------------------
class LSTMEncoder(eqx.Module):
    cell: LSTMCell

    def __init__(self, input_size: int, hidden_size: int, *, key):
        self.cell = LSTMCell(input_size, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          x: Input sequence of shape (seq_len, input_size)
        Returns:
          hs: Hidden states of shape (seq_len, hidden_size)
        """
        seq_len = x.shape[0]
        init_state = (
            jnp.zeros((self.cell.hidden_size,)),
            jnp.zeros((self.cell.hidden_size,)),
        )

        def step(state, x_t):
            new_state, h_t = self.cell(x_t, state)
            return new_state, h_t

        _, hs = jax.lax.scan(step, init_state, x)
        return hs  # shape (seq_len, hidden_size)


# -------------------------------------------------------------------
# Gating Module: computes a gate vector from the concatenation of two vectors.
# -------------------------------------------------------------------
class GatingModule(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_features: int, *, key):
        # Map concatenated vector (of size 2*hidden_size) to a gate vector (hidden_size)
        self.linear = eqx.nn.Linear(in_features, in_features // 2, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (2 * hidden_size,)
        return jnn.sigmoid(self.linear(x))  # output shape (hidden_size,)


# -------------------------------------------------------------------
# Temporal Fusion Transformer (simplified)
# -------------------------------------------------------------------
class TemporalFusionTransformer(eqx.Module):
    lstm_encoder: LSTMEncoder
    # Use Equinox’s MultiheadAttention (see our earlier MultiheadAttention module)
    attn: eqx.nn.MultiheadAttention
    gating: GatingModule
    final_linear: eqx.nn.Linear

    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,  # not used here, but could be for additional feedforward
        *,
        key
    ):
        # For simplicity, we set the LSTM’s hidden size to hidden_size.
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.lstm_encoder = LSTMEncoder(input_size, hidden_size, key=k1)
        # MultiheadAttention expects query shape (seq_len, q_dim) and keys/values of shape (seq_len, k_dim).
        # We will use the last hidden state as the query (shape (1, hidden_size)) and
        # the entire hidden sequence as keys/values.
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            key_size=hidden_size,
            value_size=hidden_size,
            output_size=hidden_size,
            dropout_p=0.0,
            inference=True,  # disable dropout for simplicity
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            key=k2,
        )
        # The gating module takes the concatenation of [last_hidden; attention_output] (size 2*hidden_size)
        self.gating = GatingModule(2 * hidden_size, key=k3)
        self.final_linear = eqx.nn.Linear(hidden_size, 1, key=k4)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input sequence of shape (seq_len, input_size)
          key: Optional PRNG key.
        Returns:
          A scalar prediction (shape (1,)) for the sample.
        """
        # 1. Process sequence with LSTM encoder.
        hs = self.lstm_encoder(x)  # shape (seq_len, hidden_size)
        last_hidden = hs[-1]  # shape (hidden_size,)
        # 2. Apply multihead attention: use last_hidden as query and hs as keys and values.
        # Note: MultiheadAttention expects inputs with shape (seq_len, feature_dim).
        # We expand the query to shape (1, hidden_size)
        attn_out = self.attn(last_hidden[None, :], hs, hs, mask=None, key=key)
        # attn_out: shape (1, hidden_size) → squeeze to (hidden_size,)
        attn_out = attn_out[0]
        # 3. Gated fusion: combine last_hidden and attn_out.
        concat = jnp.concatenate(
            [last_hidden, attn_out], axis=-1
        )  # shape (2*hidden_size,)
        gate = self.gating(concat)  # shape (hidden_size,)
        fused = gate * attn_out + (1 - gate) * last_hidden  # shape (hidden_size,)
        # 4. Final prediction.
        return self.final_linear(fused)  # shape (1,)


# -------------------------------------------------------------------
# Batch Wrapper: applies the model to each sample in the batch.
# -------------------------------------------------------------------
class TemporalFusionTransformerForecast(eqx.Module):
    model: TemporalFusionTransformer

    def __init__(self, input_size: int, hidden_size: int, num_heads: int, *, key):
        self.model = TemporalFusionTransformer(
            input_size, hidden_size, num_heads, key=key
        )

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, seq_len, input_size]
          key: Optional PRNG key.
        Returns:
          Predictions of shape [N, 1]
        """

        # We use vmap over the batch dimension.
        def process_sample(x_sample, key_sample):
            return self.model(x_sample, key=key_sample)

        if key is not None:
            batch_keys = jax.random.split(key, x.shape[0])
        else:
            batch_keys = [None] * x.shape[0]
        return jax.vmap(process_sample)(x, batch_keys)


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Batch of 8 sequences, each of length 20 with 32 features.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Create the Temporal Fusion Transformer Forecast model.
    # For simplicity, we set hidden_size equal to D.
    model_key, run_key = jax.random.split(key)
    tft_model = TemporalFusionTransformerForecast(
        input_size=D, hidden_size=D, num_heads=4, key=model_key
    )

    preds = tft_model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected: (8, 1)
