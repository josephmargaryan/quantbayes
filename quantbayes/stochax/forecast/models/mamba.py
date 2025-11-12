import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------
# MambaStateSpaceCell
#
# A simple state update cell:
#   h[t+1] = activation( A @ h[t] + B @ x[t] + bias )
# where A and B are learnable matrices.
# -------------------------------------------------------
class MambaStateSpaceCell(eqx.Module):
    A: jnp.ndarray  # shape: (hidden_size, hidden_size)
    B: jnp.ndarray  # shape: (hidden_size, input_dim)
    bias: jnp.ndarray  # shape: (hidden_size,)
    activation: callable = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_size: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.A = jr.normal(k1, (hidden_size, hidden_size)) * 0.1
        self.B = jr.normal(k2, (hidden_size, input_dim)) * 0.1
        self.bias = jr.normal(k3, (hidden_size,)) * 0.1
        self.activation = jnn.tanh

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        # Ensure x and h are flattened to 1D vectors.
        x = jnp.reshape(x, (-1,))
        h = jnp.reshape(h, (-1,))
        h_new = self.activation(jnp.dot(self.A, h) + jnp.dot(self.B, x) + self.bias)
        return h_new


# -------------------------------------------------------
# MambaStateSpaceModel
#
# Applies the state-space recurrence over a sequence.
# Given an input sequence x (shape: [seq_len, D]), it scans over time.
# The final state is mapped to a scalar forecast.
# -------------------------------------------------------
class MambaStateSpaceModel(eqx.Module):
    cell: MambaStateSpaceCell
    out: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_size: int, *, key):
        k1, k2 = jr.split(key)
        self.cell = MambaStateSpaceCell(input_dim, hidden_size, key=k1)
        self.out = eqx.nn.Linear(hidden_size, 1, key=k2)
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
            x: Input sequence of shape (seq_len, input_dim)
        Returns:
            Scalar forecast of shape (1,)
        """
        seq_len = x.shape[0]
        h = jnp.zeros((self.hidden_size,))

        def step(h, x_t):
            h_new = self.cell(x_t, h)
            return h_new, h_new

        _, hs = jax.lax.scan(step, h, x)
        final_state = hs[-1]
        return self.out(final_state)


# -------------------------------------------------------
# MambaStateSpaceForecast Wrapper
#
# This wrapper makes the model compatible with the forecasting
# training pipeline. It assumes that the input x is a single sample
# of shape (seq_len, D) and passes through a dummy state.
# The training wrapper will vmap over samples.
# -------------------------------------------------------
class MambaStateSpaceForecast(eqx.Module):
    model: MambaStateSpaceModel
    seq_len: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, seq_len: int, d: int, hidden_size: int, *, key):
        self.model = MambaStateSpaceModel(d, hidden_size, key=key)
        self.seq_len = seq_len
        self.d = d

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        """
        Args:
            x: A single sample of shape (seq_len, d)
            state: Unused state (passed through for consistency)
            key: Random key for any stochastic operations (unused here)
        Returns:
            Tuple of (forecast, state) where forecast is of shape (1,)
        """
        preds = self.model(x, key=key)
        return preds, state


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Reshape raw input to [N, seq_len, D] with D == 1.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    # Suggested hyperparameters for a univariate time series:
    # seq_len = 10, d = 1, hidden_size = 12.
    model, state = eqx.nn.make_with_state(MambaStateSpaceForecast)(
        seq_len=10, d=1, hidden_size=12, key=key
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
    print(f"preds shape: {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
