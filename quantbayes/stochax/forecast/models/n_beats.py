from typing import List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


# -------------------------------------------------------
# NBeatsBlock
# -------------------------------------------------------
class NBeatsBlock(eqx.Module):
    fc_layers: List[eqx.nn.Linear]
    forecast_layer: eqx.nn.Linear
    backcast_layer: eqx.nn.Linear
    activation: callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        forecast_dim: int,
        num_fc_layers: int,
        *,
        key,
    ):
        keys = jax.random.split(key, num_fc_layers + 2)
        self.fc_layers = []
        for i in range(num_fc_layers):
            input_dim = in_features if i == 0 else hidden_features
            self.fc_layers.append(
                eqx.nn.Linear(input_dim, hidden_features, key=keys[i])
            )
        self.forecast_layer = eqx.nn.Linear(
            hidden_features, forecast_dim, key=keys[num_fc_layers]
        )
        self.backcast_layer = eqx.nn.Linear(
            hidden_features, in_features, key=keys[num_fc_layers + 1]
        )
        self.activation = jnn.relu

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: shape (in_features,)
        h = x
        for fc in self.fc_layers:
            h = self.activation(fc(h))
        forecast = self.forecast_layer(h)  # shape (forecast_dim,)
        backcast = self.backcast_layer(h)  # shape (in_features,)
        return forecast, backcast


# -------------------------------------------------------
# NBeats
# -------------------------------------------------------
class NBeats(eqx.Module):
    blocks: List[NBeatsBlock]
    num_blocks: int = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    forecast_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        forecast_dim: int,
        num_blocks: int,
        hidden_features: int,
        num_fc_layers: int,
        *,
        key,
    ):
        keys = jax.random.split(key, num_blocks)
        self.blocks = [
            NBeatsBlock(
                in_features, hidden_features, forecast_dim, num_fc_layers, key=k
            )
            for k in keys
        ]
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.forecast_dim = forecast_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (in_features,)
        forecast_sum = 0.0
        residual = x
        for block in self.blocks:
            f, b = block(residual)
            forecast_sum = forecast_sum + f
            residual = residual - b
        return forecast_sum


# -------------------------------------------------------
# NBeatsForecast Wrapper
# -------------------------------------------------------
class NBeatsForecast(eqx.Module):
    model: NBeats
    seq_len: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(
        self,
        seq_len: int,
        d: int,
        num_blocks: int,
        hidden_features: int,
        num_fc_layers: int,
        *,
        key,
    ):
        # For a sample of shape (seq_len, d), the flattened input has length seq_len*d.
        in_features = seq_len * d
        forecast_dim = 1
        self.model = NBeats(
            in_features,
            forecast_dim,
            num_blocks,
            hidden_features,
            num_fc_layers,
            key=key,
        )
        self.seq_len = seq_len
        self.d = d

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        """
        Args:
          x: Input tensor of shape [N, seq_len, d]
        Returns:
          Forecasts of shape [N, 1]
        """
        # If our training framework vmaps over samples, each call receives a single sample of shape (seq_len, d).
        # So, we flatten it to a vector of length (seq_len * d).
        x_flat = x.reshape(self.seq_len * self.d)
        forecast = self.model(x_flat)  # shape (1,)
        return forecast, state


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Reshape raw input to [N, seq_len, d] with d == 1.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    # Suggested hyperparameters:
    #   seq_len = 10, d = 1, num_blocks = 4, hidden_features = 12, num_fc_layers = 3.
    model, state = eqx.nn.make_with_state(NBeatsForecast)(
        seq_len=10, d=1, num_blocks=4, hidden_features=12, num_fc_layers=3, key=key
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
