import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------
# LSTM Cell (adapted from Equinox examples)
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
        k1, k2, k3 = jr.split(key, 3)
        self.weight_ih = jr.normal(k1, (4 * hidden_size, input_size)) * (
            1.0 / jnp.sqrt(input_size)
        )
        self.weight_hh = jr.normal(k2, (4 * hidden_size, hidden_size)) * (
            1.0 / jnp.sqrt(hidden_size)
        )
        self.bias = (
            jr.normal(k3, (4 * hidden_size,))
            if use_bias
            else jnp.zeros((4 * hidden_size,))
        )

    def __call__(
        self, x: jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        # Ensure x is at least 1D. For input_size==1, x should be shape (1,)
        x = jnp.atleast_1d(x)
        h, c = state
        h = jnp.atleast_1d(h)
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
# LSTM Encoder: processes a sequence and returns hidden states.
# -------------------------------------------------------------------
class LSTMEncoder(eqx.Module):
    cell: LSTMCell

    def __init__(self, input_size: int, hidden_size: int, *, key):
        self.cell = LSTMCell(input_size, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (seq_len, input_size)
        seq_len = x.shape[0]
        init_state = (
            jnp.zeros((self.cell.hidden_size,)),
            jnp.zeros((self.cell.hidden_size,)),
        )

        def step(state, x_t):
            new_state, h_t = self.cell(x_t, state)
            return new_state, h_t

        _, hs = jax.lax.scan(step, init_state, x)
        return hs  # shape: (seq_len, hidden_size)


# -------------------------------------------------------------------
# Gating Module: computes a gate vector for fusion.
# -------------------------------------------------------------------
class GatingModule(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_features: int, *, key):
        # Map concatenated vector (2*hidden_size) to a gate vector (hidden_size)
        self.linear = eqx.nn.Linear(in_features, in_features // 2, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnn.sigmoid(self.linear(x))


# -------------------------------------------------------------------
# Temporal Fusion Transformer (simplified)
# -------------------------------------------------------------------
class TemporalFusionTransformer(eqx.Module):
    lstm_encoder: LSTMEncoder
    attn: eqx.nn.MultiheadAttention
    gating: GatingModule
    final_linear: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, num_heads: int, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.lstm_encoder = LSTMEncoder(input_size, hidden_size, key=k1)
        # MultiheadAttention: use the last hidden state as query and full sequence as keys/values.
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            key_size=hidden_size,
            value_size=hidden_size,
            output_size=hidden_size,
            dropout_p=0.0,
            inference=True,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            key=k2,
        )
        self.gating = GatingModule(2 * hidden_size, key=k3)
        self.final_linear = eqx.nn.Linear(hidden_size, 1, key=k4)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: shape (seq_len, input_size)
        hs = self.lstm_encoder(x)  # shape: (seq_len, hidden_size)
        last_hidden = hs[-1]  # shape: (hidden_size,)
        attn_out = self.attn(
            last_hidden[None, :], hs, hs, mask=None, key=key
        )  # shape: (1, hidden_size)
        attn_out = attn_out[0]  # shape: (hidden_size,)
        concat = jnp.concatenate(
            [last_hidden, attn_out], axis=-1
        )  # shape: (2*hidden_size,)
        gate = self.gating(concat)  # shape: (hidden_size,)
        fused = gate * attn_out + (1 - gate) * last_hidden  # shape: (hidden_size,)
        return self.final_linear(fused)  # shape: (1,)


# -------------------------------------------------------------------
# Batch Wrapper: TemporalFusionTransformerForecast
# -------------------------------------------------------------------
class TemporalFusionTransformerForecast(eqx.Module):
    model: TemporalFusionTransformer

    def __init__(self, input_size: int, hidden_size: int, num_heads: int, *, key):
        self.model = TemporalFusionTransformer(
            input_size, hidden_size, num_heads, key=key
        )

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        # x: single sample of shape [seq_len, input_size]
        return self.model(x, key=key), state


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Ensure input shape is [N, seq_len, 1] and targets are [N, 1]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(TemporalFusionTransformerForecast)(
        input_size=1, hidden_size=24, num_heads=4, key=key
    )
    trainer = ForecastingModel(lr=1e-3)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=500,
        patience=100,
        key=jr.PRNGKey(42),
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
