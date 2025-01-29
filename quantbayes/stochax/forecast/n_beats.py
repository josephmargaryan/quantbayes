import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax
import math
from typing import Any, Callable, Optional
import functools

# -------------------------------
# 1. NBeatsBlock
# -------------------------------

class NBeatsBlock(nn.Module):
    """
    A single N-BEATS block in "generic" mode:
      - MLP => two parameter vectors (theta_b, theta_f)
      - backcast = linear_b(theta_b)
      - forecast = linear_f(theta_f)
    """
    input_size: int         # L * input_dim (flattened size)
    hidden_dim: int = 256
    n_layers: int = 4
    basis: str = 'generic'  # 'generic', 'trend', 'seasonal'

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Shape (batch_size, input_size)
        Returns:
            backcast: Shape (batch_size, input_size)
            forecast: Shape (batch_size, 1)
        """
        # 1. MLP for theta
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        # 2. Final linear layer to get theta
        theta = nn.Dense(features=self.input_size + 1)(x)  # Shape: (batch_size, input_size + 1)

        # 3. Split theta into backcast and forecast
        theta_b = theta[:, :self.input_size]  # Shape: (batch_size, input_size)
        theta_f = theta[:, self.input_size:]  # Shape: (batch_size, 1)

        # 4. Mapping to backcast and forecast (generic basis uses identity)
        backcast = theta_b  # Identity mapping
        forecast = theta_f  # Identity mapping

        return backcast, forecast

# -------------------------------
# 2. NBeatsStack
# -------------------------------

class NBeatsStack(nn.Module):
    """
    A stack of multiple N-BEATS blocks. The final forecast is the sum of each block's forecast.
    The residual for block i+1 is x_{i+1} = x_i - backcast_i.
    """
    input_size: int         # L * input_dim
    num_blocks: int = 3
    block_hidden: int = 256
    n_layers: int = 4
    basis: str = 'generic'

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Shape (batch_size, input_size)
        Returns:
            forecast_final: Shape (batch_size, 1)
        """
        forecast_final = jnp.zeros((x.shape[0], 1), dtype=x.dtype)  # Initialize forecast
        residual = x  # Initialize residual

        for _ in range(self.num_blocks):
            block = NBeatsBlock(
                input_size=self.input_size,
                hidden_dim=self.block_hidden,
                n_layers=self.n_layers,
                basis=self.basis
            )
            backcast, forecast = block(residual)  # Each of shape (batch_size, input_size) and (batch_size, 1)
            residual = residual - backcast
            forecast_final += forecast

        return forecast_final

# -------------------------------
# 3. NBeats Model
# -------------------------------

class NBeats(nn.Module):
    """
    A simplified N-BEATS model for single-step forecasting,
    with "generic basis" blocks, no future covariates.

    Input shape:  (batch_size, L, input_dim)
    Output shape: (batch_size, 1)
    """
    seq_len: int            # L
    input_dim: int = 1
    num_blocks: int = 3
    block_hidden: int = 256
    n_layers: int = 4
    basis: str = 'generic'

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Shape (batch_size, L, input_dim)
        Returns:
            forecast: Shape (batch_size, 1)
        """
        batch_size, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got L={L}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got C={C}"

        # 1. Flatten the input: (batch_size, L, C) -> (batch_size, L*C)
        x_flat = x.reshape(batch_size, -1)

        # 2. Pass through NBeatsStack
        forecast = NBeatsStack(
            input_size=self.seq_len * self.input_dim,
            num_blocks=self.num_blocks,
            block_hidden=self.block_hidden,
            n_layers=self.n_layers,
            basis=self.basis
        )(x_flat)  # Shape: (batch_size, 1)

        return forecast

# -------------------------------
# 4. Simple Forward Pass Test
# -------------------------------

def test_nbeats():
    # Hyperparameters
    batch_size = 16
    seq_len = 24
    input_dim = 2  # e.g., 2 features

    # Initialize model
    rng = jax.random.PRNGKey(0)
    model = NBeats(
        seq_len=seq_len,
        input_dim=input_dim,
        num_blocks=3,
        block_hidden=128,
        n_layers=4,
        basis='generic'
    )

    # Create dummy input tensor
    dummy_input = jax.random.normal(rng, (batch_size, seq_len, input_dim))  # Shape: (batch_size, seq_len, input_dim)

    # Initialize model parameters
    variables = model.init(rng, dummy_input)  # 'params' are stored under 'params'

    # Perform a forward pass
    forecast = model.apply(variables, dummy_input)  # Shape: (batch_size, 1)

    # Print the output shape
    print("Forecast shape:", forecast.shape)  # Expected: (batch_size, 1)

if __name__ == "__main__":
    test_nbeats()
