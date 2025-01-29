import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax
import math
from typing import Any, Callable, Optional
import functools

# -------------------------------
# 1. Helper Functions
# -------------------------------

def trend_matrix(L: int, degree: int) -> jnp.ndarray:
    """
    Create a (L x degree) polynomial matrix, e.g., for i in [0..L-1],
    columns are [1, i, i^2, ..., i^(degree-1)] (scaled).

    Args:
        L: Length of the sequence.
        degree: Degree of the polynomial.

    Returns:
        A (L, degree) matrix.
    """
    t = jnp.linspace(0, 1, L).reshape(L, 1)  # Shape: (L, 1)
    powers = jnp.concatenate([t**p for p in range(degree)], axis=1)  # Shape: (L, degree)
    return powers  # Shape: (L, degree)

def seasonal_matrix(L: int, harmonics: int) -> jnp.ndarray:
    """
    Create a (L x 2*harmonics) matrix with columns of sin/cos expansions:
       [cos(2 pi k t), sin(2 pi k t), ...] for k=1..harmonics

    Args:
        L: Length of the sequence.
        harmonics: Number of harmonics.

    Returns:
        A (L, 2*harmonics) matrix.
    """
    t = jnp.linspace(0, 1, L).reshape(L, 1)  # Shape: (L, 1)
    mats = []
    for k in range(1, harmonics + 1):
        mats.append(jnp.cos(2 * jnp.pi * k * t))  # Shape: (L, 1)
        mats.append(jnp.sin(2 * jnp.pi * k * t))  # Shape: (L, 1)
    mat = jnp.concatenate(mats, axis=1)  # Shape: (L, 2*harmonics)
    return mat  # Shape: (L, 2*harmonics)

# -------------------------------
# 2. NBeatsBlock
# -------------------------------

class NBeatsBlock(nn.Module):
    input_size: int         # Flattened size = L * input_dim
    backcast_length: int    # L (time steps)
    input_dim: int = 1      # Number of features (must be 1 if trend/seasonal)
    basis: str = 'generic'  # 'generic', 'trend', 'seasonal'
    hidden_dim: int = 256
    n_layers: int = 4
    degree_of_polynomial: int = 2
    n_harmonics: int = 2

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]

        # Determine backcast and forecast sizes
        if self.basis == 'generic':
            backcast_size = self.input_size
            forecast_size = 1
        elif self.basis == 'trend':
            backcast_size = self.backcast_length * self.degree_of_polynomial
            forecast_size = self.degree_of_polynomial
        elif self.basis == 'seasonal':
            backcast_size = self.backcast_length * 2 * self.n_harmonics
            forecast_size = 2 * self.n_harmonics
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")

        # MLP for theta
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        # Final linear layer to get theta
        theta = nn.Dense(features=backcast_size + forecast_size)(x)

        # Split theta into backcast and forecast
        theta_b = theta[:, :backcast_size]
        theta_f = theta[:, backcast_size:]

        if self.basis == 'generic':
            # Identity mapping
            backcast = theta_b
            forecast = theta_f

        elif self.basis == 'trend':
            # Reshape theta_b for trend basis
            theta_b = theta_b.reshape(B, self.backcast_length, self.degree_of_polynomial)
            t_back = trend_matrix(self.backcast_length, self.degree_of_polynomial)

            # Backcast: (B, L, degree) * (L, degree) -> (B, L)
            backcast = jnp.einsum('bld,ld->bl', theta_b, t_back)

            # Forecast: (B, 1, degree) * (1, degree) -> (B, 1)
            t_fore = trend_matrix(1, self.degree_of_polynomial)
            forecast = jnp.einsum('bd,ld->bl', theta_f, t_fore)

        elif self.basis == 'seasonal':
            # Reshape theta_b for seasonal basis
            theta_b = theta_b.reshape(B, self.backcast_length, 2 * self.n_harmonics)
            s_back = seasonal_matrix(self.backcast_length, self.n_harmonics)

            # Backcast: (B, L, 2*harmonics) * (L, 2*harmonics) -> (B, L)
            backcast = jnp.einsum('blh,lh->bl', theta_b, s_back)

            # Forecast: (B, 1, 2*harmonics) * (1, 2*harmonics) -> (B, 1)
            s_fore = seasonal_matrix(1, self.n_harmonics)
            forecast = jnp.einsum('bh,lh->bl', theta_f, s_fore)

        return backcast, forecast


# -------------------------------
# 3. NBeatsStack
# -------------------------------

class NBeatsStack(nn.Module):
    """
    A stack of multiple N-BEATS blocks. The final forecast is the sum of each block's forecast.
    The residual for block i+1 is x_{i+1} = x_i - backcast_i.
    """
    input_size: int         # Flattened size = L * input_dim or L for 'trend'/'seasonal'
    backcast_length: int    # L (time steps)
    input_dim: int = 1      # Number of features (must be 1 if trend/seasonal)
    num_blocks: int = 3
    block_hidden: int = 256
    n_layers: int = 4
    basis: str = 'generic'  # 'generic', 'trend', 'seasonal'
    degree_of_polynomial: int = 2
    n_harmonics: int = 2

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Shape (batch_size, input_size) for 'generic' or (batch_size, L) for 'trend'/'seasonal'
        Returns:
            forecast_final: Shape (batch_size, 1)
        """
        forecast_final = jnp.zeros((x.shape[0], 1), dtype=x.dtype)  # Initialize forecast
        residual = x  # Initialize residual

        for _ in range(self.num_blocks):
            block = NBeatsBlock(
                input_size=self.input_size,
                backcast_length=self.backcast_length,
                input_dim=self.input_dim,
                basis=self.basis,
                hidden_dim=self.block_hidden,
                n_layers=self.n_layers,
                degree_of_polynomial=self.degree_of_polynomial,
                n_harmonics=self.n_harmonics
            )
            backcast, forecast = block(residual)  # Each of shape (batch_size, input_size) or (batch_size, L) and (batch_size, 1)

            residual = residual - backcast  # Update residual
            forecast_final = forecast_final + forecast  # Accumulate forecast

        return forecast_final  # Shape: (batch_size, 1)

# -------------------------------
# 4. NBeats2 Model
# -------------------------------

class NBeats2(nn.Module):
    """
    N-BEATS model for single-step forecasting, with selectable basis:
      - 'generic' (can handle input_dim>1)
      - 'trend'/'seasonal' (assumes univariate => input_dim=1)

    Input shape:  (batch_size, L, input_dim)
    Output shape: (batch_size, 1)
    """
    seq_len: int            # L
    input_dim: int = 1
    num_blocks: int = 3
    block_hidden: int = 256
    n_layers: int = 4
    basis: str = 'generic'  # 'generic', 'trend', 'seasonal'
    degree_of_polynomial: int = 2
    n_harmonics: int = 2

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Shape (batch_size, L, input_dim)
        Returns:
            forecast: Shape (batch_size, 1)
        """
        B, L, C = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got L={L}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got C={C}"

        # Check basis and input_dim constraints
        if self.basis in ['trend', 'seasonal'] and self.input_dim != 1:
            raise ValueError(
                f"Basis='{self.basis}' is only implemented for univariate series (input_dim=1). "
                f"For multivariate, use basis='generic'."
            )

        # 1. Flatten the input if 'generic'; else, keep as (B, L)
        if self.basis == 'generic':
            x_flat = x.reshape(B, -1)  # Shape: (B, L * input_dim)
        else:
            x_flat = x[..., 0]  # Shape: (B, L)

        # 2. Pass through NBeatsStack
        forecast = NBeatsStack(
            input_size=(self.seq_len * self.input_dim if self.basis == 'generic' else self.seq_len),
            backcast_length=self.seq_len,
            input_dim=self.input_dim if self.basis == 'generic' else 1,
            num_blocks=self.num_blocks,
            block_hidden=self.block_hidden,
            n_layers=self.n_layers,
            basis=self.basis,
            degree_of_polynomial=self.degree_of_polynomial,
            n_harmonics=self.n_harmonics
        )(x_flat)  # Shape: (B, 1)

        return forecast  # Shape: (B, 1)

# -------------------------------
# 5. Simple Forward Pass Test
# -------------------------------

def test_nbeats():
    # Hyperparameters
    batch_size = 4
    seq_len = 16

    # ------------------------
    # 5.1 Generic + Multi-variate
    # ------------------------
    input_dim_generic = 2
    model_generic = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim_generic,
        num_blocks=2,
        block_hidden=128,
        n_layers=4,
        basis='generic'
    )
    rng = jax.random.PRNGKey(0)
    dummy_input_generic = jax.random.normal(rng, (batch_size, seq_len, input_dim_generic))  # Shape: (B, L, C)
    variables_generic = model_generic.init(rng, dummy_input_generic)  # Initialize parameters
    forecast_generic = model_generic.apply(variables_generic, dummy_input_generic)  # Forward pass
    print("Generic =>", forecast_generic.shape)  # Expected: (B, 1)

    # ------------------------
    # 5.2 Trend + Univariate
    # ------------------------
    input_dim_trend = 1
    model_trend = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim_trend,
        num_blocks=2,
        block_hidden=128,
        n_layers=4,
        basis='trend',
        degree_of_polynomial=3
    )
    rng, rng_trend = jax.random.split(rng)
    dummy_input_trend = jax.random.normal(rng_trend, (batch_size, seq_len, input_dim_trend))  # Shape: (B, L, 1)
    variables_trend = model_trend.init(rng_trend, dummy_input_trend)  # Initialize parameters
    forecast_trend = model_trend.apply(variables_trend, dummy_input_trend)  # Forward pass
    print("Trend =>", forecast_trend.shape)  # Expected: (B, 1)

    # ------------------------
    # 5.3 Seasonal + Univariate
    # ------------------------
    input_dim_seasonal = 1
    model_seasonal = NBeats2(
        seq_len=seq_len,
        input_dim=input_dim_seasonal,
        num_blocks=2,
        block_hidden=128,
        n_layers=4,
        basis='seasonal',
        n_harmonics=5
    )
    rng, rng_seasonal = jax.random.split(rng)
    dummy_input_seasonal = jax.random.normal(rng_seasonal, (batch_size, seq_len, input_dim_seasonal))  # Shape: (B, L, 1)
    variables_seasonal = model_seasonal.init(rng_seasonal, dummy_input_seasonal)  # Initialize parameters
    forecast_seasonal = model_seasonal.apply(variables_seasonal, dummy_input_seasonal)  # Forward pass
    print("Seasonal =>", forecast_seasonal.shape)  # Expected: (B, 1)

if __name__ == "__main__":
    test_nbeats()
