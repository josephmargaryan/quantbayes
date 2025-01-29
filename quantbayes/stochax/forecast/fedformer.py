from flax import linen as nn
import jax.numpy as jnp
from typing import Tuple


class SeriesDecomposition(nn.Module):
    kernel_size: int

    def setup(self):
        # Create the fixed moving average kernel
        # Shape: (kernel_size, 1, 1) for depthwise convolution
        self.kernel = jnp.ones((self.kernel_size, 1, 1)) / self.kernel_size

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (batch, length, channels)
        Returns:
            trend: Smoothed trend component of shape (batch, length, channels)
            seasonal: Residual seasonal component of shape (batch, length, channels)
        """
        # Ensure the input has the expected shape
        assert len(x.shape) == 3, "Input must be of shape (batch, length, channels)"

        batch, length, channels = x.shape

        # Expand kernel to match the number of input channels
        kernel = jnp.tile(
            self.kernel, (1, 1, channels)
        )  # Shape: (kernel_size, 1, channels)

        # Apply convolution along the time dimension
        trend = jax.lax.conv_general_dilated(
            x,  # (batch, length, channels)
            kernel,  # (kernel_size, 1, channels)
            window_strides=(1,),  # Stride 1
            padding=((self.kernel_size // 2, self.kernel_size // 2),),  # "SAME" padding
            dimension_numbers=("NWC", "WIO", "NWC"),  # Channels last format
            feature_group_count=channels,  # One kernel per channel
        )

        # Compute the seasonal component
        seasonal = x - trend

        return trend, seasonal


class FourierBlock(nn.Module):
    """A simple frequency block that uses 1D FFT -> processing -> iFFT."""

    top_k_freq: int

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (batch, length, channels), real-valued
        Returns:
            Processed tensor of shape (batch, length, channels), real-valued
        """
        # Ensure the input has the expected shape
        assert len(x.shape) == 3, "Input must be of shape (batch, length, channels)"
        batch, length, channels = x.shape

        # 1) FFT
        freq_x = jnp.fft.rfft(
            x, axis=1
        )  # shape: (batch, length // 2 + 1, channels), complex

        # 2) Select top-K largest frequencies based on magnitude
        mag = jnp.abs(freq_x)  # shape: (batch, length // 2 + 1, channels)
        freq_x_filtered = jnp.zeros_like(freq_x)  # Initialize filtered frequency tensor

        def filter_frequencies(freq_x_b, mag_b):
            # Filter frequencies for a single batch
            def filter_channel(freq_x_bc, mag_bc):
                # Get top-k indices for this channel
                top_k_indices = jnp.argsort(mag_bc)[-self.top_k_freq :]
                # Create a mask to retain only the top-k frequencies
                mask = jnp.zeros_like(mag_bc)
                mask = mask.at[top_k_indices].set(1)
                # Apply the mask to the frequency domain
                return freq_x_bc * mask

            # Apply the filtering channel-wise
            return jax.vmap(filter_channel, in_axes=(1, 1))(freq_x_b, mag_b)

        # Apply the filtering batch-wise
        freq_x_filtered = jax.vmap(filter_frequencies, in_axes=(0, 0))(freq_x, mag)

        # 3) iFFT
        out = jnp.fft.irfft(
            freq_x_filtered, n=length, axis=1
        )  # shape: (batch, length, channels), real
        return out


class FedformerEncoderLayer(nn.Module):
    """
    A layer that applies:
      - A FourierBlock to the seasonal component
      - A feed-forward network with dropout and layer normalization
      - Optionally merges or passes the trend component unmodified
    """

    d_model: int = 64
    top_k_freq: int = 16
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self, seasonal: jnp.ndarray, trend: jnp.ndarray, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            seasonal: [batch, length, d_model] - Seasonal component
            trend: [batch, length, d_model] - Trend component
            deterministic: Whether to apply dropout during inference
        Returns:
            processed_seasonal: [batch, length, d_model]
            trend: [batch, length, d_model] - Passed unmodified
        """
        # 1) Frequency modeling of seasonal
        freq_seasonal = FourierBlock(top_k_freq=self.top_k_freq)(
            seasonal
        )  # [batch, length, d_model]

        # 2) Feed-forward on frequency-transformed seasonal
        out = nn.Dense(self.d_model)(freq_seasonal)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        # Add skip connection and layer normalization
        processed_seasonal = nn.LayerNorm()(seasonal + out)

        # Pass trend forward unmodified
        return processed_seasonal, trend


class FedformerEncoder(nn.Module):
    """
    Stack multiple FedformerEncoderLayer with series decomposition each time.
    """

    d_model: int = 64
    layer_num: int = 2
    decomp_ks: int = 25
    top_k_freq: int = 16
    dropout: float = 0.1

    def setup(self):
        # Series decomposition module
        self.decomp = SeriesDecomposition(kernel_size=self.decomp_ks)
        # Stack of FedformerEncoderLayer
        self.layers = [
            FedformerEncoderLayer(
                d_model=self.d_model, top_k_freq=self.top_k_freq, dropout=self.dropout
            )
            for _ in range(self.layer_num)
        ]
        # Layer normalization
        self.layernorm = nn.LayerNorm()

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [batch, length, d_model] - Input time series data
            deterministic: Whether to apply dropout in deterministic mode (default=True)
        Returns:
            out: [batch, length, d_model] - Final combined output
        """
        # Initial decomposition into seasonal and trend components
        seasonal, trend = self.decomp(x)  # [batch, length, d_model] each

        # Pass through each layer
        for layer in self.layers:
            seasonal, trend = layer(seasonal, trend, deterministic=deterministic)

        # Combine seasonal and trend components
        out = seasonal + trend
        # Apply final layer normalization
        out = self.layernorm(out)
        return out


class Fedformer(nn.Module):
    """
    A simplified Fedformer-like model with:
      - Input linear embedding
      - FedformerEncoder
      - Output projection (single-step forecast -> take last step's output)

    Input shape: (batch, length, input_dim)
    Output shape: (batch, 1)
    """

    input_dim: int
    d_model: int = 64
    layer_num: int = 2
    top_k_freq: int = 16
    dropout: float = 0.1
    kernel_size: int = 25

    def setup(self):
        # 1) Input embedding
        self.embedding = nn.Dense(self.d_model)
        # 2) Fedformer encoder
        self.encoder = FedformerEncoder(
            d_model=self.d_model,
            layer_num=self.layer_num,
            decomp_ks=self.kernel_size,
            top_k_freq=self.top_k_freq,
            dropout=self.dropout,
        )
        # 3) Final projection
        self.projection = nn.Dense(1)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, length, input_dim)
            deterministic: Whether to apply dropout in deterministic mode
        Returns:
            out: Output tensor of shape (batch, 1)
        """
        # 1) Embed input
        x = self.embedding(x)  # [batch, length, d_model]
        # 2) Apply Fedformer encoder
        enc_out = self.encoder(
            x, deterministic=deterministic
        )  # [batch, length, d_model]
        # 3) Take the last time step
        last_token = enc_out[:, -1, :]  # [batch, d_model]
        # 4) Project to a single value
        out = self.projection(last_token)  # [batch, 1]
        return out


# Testing the full model
if __name__ == "__main__":
    import numpy as np
    import jax
    import jax.numpy as jnp

    # Initialize RNG
    rng = jax.random.PRNGKey(0)

    # Sample data
    batch_size = 4
    sequence_length = 32
    input_dim = 10

    # Random input time series
    x = jnp.array(
        np.random.rand(batch_size, sequence_length, input_dim), dtype=jnp.float32
    )

    # Initialize the model
    model = Fedformer(
        input_dim=input_dim,
        d_model=64,
        layer_num=2,
        top_k_freq=16,
        dropout=0.1,
        kernel_size=25,
    )

    # Initialize parameters
    variables = model.init(rng, x)

    # Apply the model
    out = model.apply(variables, x, deterministic=True)

    # Print the shapes of the results
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
