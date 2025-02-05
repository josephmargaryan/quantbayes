import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from typing import List


# -------------------------------------------------------
# NBeatsBlock
#
# A simplified block that maps an input vector x (of dimension in_features)
# through several fully connected layers (with ReLU activation) to produce:
#   - a forecast vector (of dimension forecast_dim, here 1), and
#   - a backcast vector (of the same dimension as x).
#
# In the full N‑BEATS architecture the backcast is used to “explain away”
# the input; the forecast contributions are summed across blocks.
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
        key
    ):
        keys = jax.random.split(key, num_fc_layers + 2)
        # Create a list of fully-connected layers.
        self.fc_layers = []
        for i in range(num_fc_layers):
            # For the first layer, input dimension is in_features; then hidden_features.
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
#
# The overall model is a stack of NBeatsBlocks.
# For a given block, its forecast is added to the cumulative forecast,
# and its backcast is subtracted from the input before the next block.
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
        key
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
        # Initialize cumulative forecast.
        forecast_sum = 0.0
        residual = x
        for block in self.blocks:
            f, b = block(residual)
            forecast_sum = forecast_sum + f
            residual = residual - b
        return forecast_sum


# -------------------------------------------------------
# NBeatsForecast
#
# A wrapper that:
#   1. Accepts inputs of shape [N, seq_len, D].
#   2. Flattens each sample to shape [seq_len * D].
#   3. Applies the NBeats model to produce a forecast (of shape [1]).
#   4. vmaps over the batch dimension so that the output is [N, 1].
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
        key
    ):
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, seq_len, D]
        Returns:
          Forecasts of shape [N, 1]
        """
        # Flatten each sample along the time and feature dimensions.
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, self.seq_len * self.d)
        # Apply the model to each sample via vmap.
        forecasts = jax.vmap(self.model)(x_flat)  # shape [N, 1]
        return forecasts


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------
if __name__ == "__main__":
    # For example, a batch of 8 sequences, each of length 20, with 32 features.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Hyperparameters for the NBeats network.
    num_blocks = 3  # number of blocks to stack
    hidden_features = 128  # hidden layer size within each block
    num_fc_layers = 3  # number of fully connected layers per block

    model_key, run_key = jax.random.split(key)
    model = NBeatsForecast(
        seq_len, D, num_blocks, hidden_features, num_fc_layers, key=model_key
    )

    preds = model(x)
    print("Predictions shape:", preds.shape)  # Expected: (8, 1)
