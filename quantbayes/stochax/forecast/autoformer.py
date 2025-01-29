import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


# -----------------------------------------------------------------------------
# 1) Series Decomposition
# -----------------------------------------------------------------------------
class SeriesDecomposition(nn.Module):
    """Decompose a time series into trend + seasonal using a simple moving average."""
    kernel_size: int = 25  # Size of the moving average kernel

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Input tensor of shape [B, L, C]
        Returns:
            Tuple (trend, seasonal), each of shape [B, L, C]
        """
        B, L, C = x.shape
        pad = self.kernel_size // 2

        # Pad in the sequence length (L) dimension with reflection padding
        x_padded = jnp.pad(
            x,
            pad_width=((0, 0), (pad, pad), (0, 0)),  # Only pad the L dimension
            mode="reflect"
        )  # [B, L + 2*pad, C]

        # Create moving average kernel with output channels matching C
        kernel = jnp.ones((self.kernel_size, 1, C)) / self.kernel_size  # [kernel_size, 1, C]

        # Perform depthwise convolution
        trend = jax.lax.conv_general_dilated(
            lhs=x_padded,                   # [B, L + 2*pad, C]
            rhs=kernel,                     # [kernel_size, 1, C]
            window_strides=(1,),            # Stride of 1
            padding='VALID',                # Padding is already applied
            dimension_numbers=('NWC', 'WIO', 'NWC'),  # Corrected dimension numbers
            feature_group_count=C           # Perform depthwise convolution
        )  # [B, L, C]

        # Seasonal component
        seasonal = x - trend

        return trend, seasonal


# -----------------------------------------------------------------------------
# 2) AutoCorrelation Block
# -----------------------------------------------------------------------------
class AutoCorrelationBlock(nn.Module):
    """
    A simplified version of the "Auto-Correlation" mechanism:
      - Transform to frequency domain via FFT
      - Multiply Q * conj(K) => correlation in freq domain
      - iFFT -> get correlation in time domain
      - Pick top-k correlated time-lags
      - Reconstruct an output
    """
    top_k: int = 16  # Number of top correlations to retain

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x (jnp.ndarray): Input tensor of shape [B, L, C].

        Returns:
            jnp.ndarray: Output tensor of shape [B, L, C].
        """
        B, L, C = x.shape

        # 1) FFT
        X_freq = jnp.fft.rfft(x, axis=1)  # [B, L//2+1, C], complex

        # 2) AutoCorrelation: X_freq * conj(X_freq) = power spectrum
        AC_freq = X_freq * jnp.conj(X_freq)  # [B, L//2+1, C]

        # 3) iFFT -> correlation in time domain
        corr_time = jnp.fft.irfft(AC_freq, n=L, axis=1)  # [B, L, C], real

        # 4) Select top-k correlations in the time dimension
        mag = jnp.abs(corr_time)  # [B, L, C]

        # Transpose to [B, C, L] to perform top_k along the last axis (time)
        mag_transposed = jnp.transpose(mag, (0, 2, 1))  # [B, C, L]

        # Use lax.top_k to get top-k indices along the time axis for each [B, C]
        topk_values, topk_indices = jax.lax.top_k(mag_transposed, self.top_k)  # [B, C, top_k]

        # Create one-hot encodings for the top-k indices
        one_hot = jax.nn.one_hot(topk_indices, num_classes=L)  # [B, C, top_k, L]

        # Sum along the top_k dimension to create the mask
        mask = jnp.sum(one_hot, axis=2)  # [B, C, L]

        # Transpose back to [B, L, C]
        mask = jnp.transpose(mask, (0, 2, 1))  # [B, L, C]

        # Apply the mask to retain only top-k correlations
        out = corr_time * mask  # [B, L, C]

        return out


# -----------------------------------------------------------------------------
# 3) Autoformer Encoder Layer
# -----------------------------------------------------------------------------
class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer Encoder Layer:
      - Applies AutoCorrelationBlock to the seasonal component
      - Uses a feed-forward layer, skip connections, and layer normalization
    """
    d_model: int = 64
    top_k: int = 16
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        seasonal: jnp.ndarray,
        trend: jnp.ndarray,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            seasonal (jnp.ndarray): Seasonal component [B, L, d_model].
            trend (jnp.ndarray): Trend component [B, L, d_model].
            train (bool): Flag indicating training mode (affects dropout).
            rngs: Random number generators for operations like dropout.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Processed seasonal and trend components.
        """
        # 1) AutoCorrelation on seasonal
        auto_seasonal = AutoCorrelationBlock(top_k=self.top_k)(seasonal)  # [B, L, d_model]

        # 2) Feed-forward network with skip connection
        ff_output = nn.Dense(self.d_model, name="feed_forward")(auto_seasonal)  # [B, L, d_model]
        ff_output = nn.Dropout(rate=self.dropout)(
            ff_output,
            deterministic=not train,
            rng=rngs.get("dropout") if rngs else None
        )

        # Add skip connection and apply layer normalization
        seasonal_out = nn.LayerNorm(name="layer_norm")(seasonal + ff_output)  # [B, L, d_model]

        # The trend component is passed through unchanged
        return seasonal_out, trend


# -----------------------------------------------------------------------------
# 4) Autoformer Encoder
# -----------------------------------------------------------------------------
class AutoformerEncoder(nn.Module):
    """
    Repeated decomposition and auto-correlation across multiple layers.
    """
    d_model: int = 64
    layer_num: int = 2
    kernel_size: int = 25
    top_k: int = 16
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, d_model]
            train: Training flag
            rngs: Random number generators

        Returns:
            jnp.ndarray: Encoded output [B, L, d_model]
        """
        # 1) Decompose into trend + seasonal
        trend, seasonal = SeriesDecomposition(kernel_size=self.kernel_size)(x)  # [B, L, d_model] each

        # 2) Pass through stacked encoder layers
        for i in range(self.layer_num):
            seasonal, trend = AutoformerEncoderLayer(
                d_model=self.d_model,
                top_k=self.top_k,
                dropout=self.dropout,
                name=f"encoder_layer_{i}"
            )(seasonal, trend, train=train, rngs=rngs)

        # 3) Combine seasonal and trend
        out = seasonal + trend

        # 4) Final layer normalization
        out = nn.LayerNorm(name="final_layer_norm")(out)

        return out


# -----------------------------------------------------------------------------
# 5) Autoformer Model
# -----------------------------------------------------------------------------
class Autoformer(nn.Module):
    """
    A simplified Autoformer for single-step forecasting:
      - Input linear embedding
      - AutoformerEncoder (with auto-correlation blocks)
      - Final projection to 1 dimension
      - Return last time step's forecast
    """
    input_dim: int
    d_model: int = 64
    layer_num: int = 2
    top_k: int = 16
    dropout: float = 0.1
    kernel_size: int = 25

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = True,
        rngs: Optional[dict] = None
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, L, input_dim]
            train: Training flag
            rngs: Random number generators

        Returns:
            jnp.ndarray: Forecasted value [B, 1]
        """
        # 1) Input embedding
        embedding = nn.Dense(self.d_model, name="embedding")(x)  # [B, L, d_model]

        # 2) Autoformer encoder
        encoder = AutoformerEncoder(
            d_model=self.d_model,
            layer_num=self.layer_num,
            kernel_size=self.kernel_size,
            top_k=self.top_k,
            dropout=self.dropout,
            name="autoformer_encoder"
        )
        enc_out = encoder(embedding, train=train, rngs=rngs)  # [B, L, d_model]

        # 3) Take the last time step
        last_token = enc_out[:, -1, :]  # [B, d_model]

        # 4) Final projection to 1 dimension
        out = nn.Dense(1, name="projection")(last_token)  # [B, 1]

        return out


# -----------------------------------------------------------------------------
# 6) Full Test
# -----------------------------------------------------------------------------
def test_autoformer():
    """
    Test the complete Autoformer model with a sample input.
    """
    # Define parameters
    batch_size = 4
    seq_len = 16
    input_dim = 8
    d_model = 32
    layer_num = 2
    top_k = 8
    dropout = 0.1
    kernel_size = 7

    # Generate random input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, seq_len, input_dim))

    # Initialize the model
    model = Autoformer(
        input_dim=input_dim,
        d_model=d_model,
        layer_num=layer_num,
        top_k=top_k,
        dropout=dropout,
        kernel_size=kernel_size
    )

    # Initialize parameters
    variables = model.init(jax.random.PRNGKey(1), x, train=True)

    # Apply the model
    y = model.apply(variables, x, train=False)

    # Print output shape
    print("Output shape:", y.shape)  # Expected: (4, 1)
    print("Output:", y)


if __name__ == "__main__":
    test_autoformer()
