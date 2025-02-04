import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial

# --- Helper: Simple series decomposition ---
def decompose(x: jnp.ndarray, kernel_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decompose an input series x into seasonal and trend components.
    Args:
      x: A JAX array of shape [seq_len, D].
      kernel_size: Size of the moving average filter.
    Returns:
      seasonal: x - trend, same shape as x.
      trend: Moving average of x (computed channelwise), same shape as x.
    """
    # x: [seq_len, D]. For each channel, compute the moving average.
    # We use a simple convolution with an average kernel.
    kernel = jnp.ones((kernel_size,)) / kernel_size

    # For each channel (i.e. for each column), perform 1D convolution.
    # We use jax.vmap over the channel dimension. (x.T has shape [D, seq_len])
    trend = jax.vmap(lambda v: jnp.convolve(v, kernel, mode='same'))(x.T).T
    seasonal = x - trend
    return seasonal, trend

# --- Reuse building blocks from before ---
# MLP: a simple feed-forward block with one hidden layer, activation and dropout.
class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(self, in_features: int, hidden_features: int, dropout_p: float, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=key2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.relu  # you might experiment with jnn.gelu

    def __call__(self, x, *, key=None):
        y = self.fc1(x)
        y = self.activation(y)
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1 = key2 = None
        y = self.dropout(y, key=key1)
        y = self.fc2(y)
        y = self.dropout(y, key=key2)
        return y

# TransformerBlock: similar to a standard transformer encoder block.
# (We reuse the same ideas as before but note that here it will be used only on the seasonal component.)
class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout_attn: eqx.nn.Dropout

    norm2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 dropout_p: float = 0.1,
                 *,
                 key):
        key_norm1, key_attn, key_dropout, key_norm2, key_mlp = jax.random.split(key, 5)
        self.norm1 = eqx.nn.LayerNorm(embed_dim, eps=1e-6, use_weight=True, use_bias=True)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout_p,
            inference=False,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            key=key_attn,
        )
        self.dropout_attn = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.norm2 = eqx.nn.LayerNorm(embed_dim, eps=1e-6, use_weight=True, use_bias=True)
        hidden_mlp = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_mlp, dropout_p=dropout_p, key=key_mlp)

    def __call__(self, x, *, key=None):
        # x: [seq_len, embed_dim]
        if key is not None:
            key_attn, key_mlp = jax.random.split(key)
        else:
            key_attn = key_mlp = None

        # Apply layer norm tokenwise.
        y = jax.vmap(self.norm1)(x)
        attn_out = self.attn(y, y, y, key=key_attn)
        attn_out = self.dropout_attn(attn_out, key=key_attn)
        x = x + attn_out  # residual connection

        # Feed-forward (MLP) block. Apply layer norm tokenwise.
        y = jax.vmap(self.norm2)(x)
        if key_mlp is not None:
            keys_mlp = jax.random.split(key_mlp, y.shape[0])
        else:
            keys_mlp = None
        mlp_out = jax.vmap(self.mlp)(y, key=keys_mlp)
        x = x + mlp_out  # residual connection
        return x

# --- The Fedformer Module ---
class Fedformer(eqx.Module):
    kernel_size: int
    seasonal_block: TransformerBlock
    trend_net: MLP
    final_linear: eqx.nn.Linear

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout_p: float = 0.1,
                 kernel_size: int = 3,
                 *,
                 key):
        # Split keys for each submodule.
        key_seasonal, key_trend, key_final = jax.random.split(key, 3)
        self.kernel_size = kernel_size
        self.seasonal_block = TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=key_seasonal)
        # The trend branch: a simple MLP (applied tokenwise) to capture the smooth trend.
        self.trend_net = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout_p, key=key_trend)
        # Final linear layer takes concatenated seasonal and trend features from the last time step.
        self.final_linear = eqx.nn.Linear(embed_dim * 2, 1, key=key_final)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input array of shape [seq_len, embed_dim] (one sample).
          key: Optional PRNG key.
        Returns:
          A prediction (array of shape [1]) computed from the last time step.
        """
        # Decompose the input series into seasonal and trend parts.
        seasonal, trend = decompose(x, self.kernel_size)
        # Process the seasonal component with the transformer block.
        if key is not None:
            key_seasonal, key_trend = jax.random.split(key)
        else:
            key_seasonal = key_trend = None
        seasonal_out = self.seasonal_block(seasonal, key=key_seasonal)  # shape: [seq_len, embed_dim]
        # Process the trend component with the MLP (applied tokenwise).
        if key_trend is not None:
            keys_trend = jax.random.split(key_trend, trend.shape[0])
        else:
            keys_trend = None
        trend_out = jax.vmap(self.trend_net)(trend, key=keys_trend)  # shape: [seq_len, embed_dim]
        # Select the last time step features from both branches.
        last_seasonal = seasonal_out[-1]  # shape: [embed_dim]
        last_trend = trend_out[-1]        # shape: [embed_dim]
        # Concatenate features and produce final forecast.
        final_feat = jnp.concatenate([last_seasonal, last_trend], axis=-1)
        return self.final_linear(final_feat)

# --- A wrapper to process batched inputs ---
class FedformerForecast(eqx.Module):
    model: Fedformer

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout_p: float = 0.1,
                 kernel_size: int = 3,
                 *,
                 key):
        self.model = Fedformer(embed_dim, num_heads, mlp_ratio, dropout_p, kernel_size, key=key)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input array of shape [N, seq_len, embed_dim].
          key: Optional PRNG key.
        Returns:
          An array of shape [N, 1] with forecasts for the last time step.
        """
        def process_sample(x_sample, key_sample):
            return self.model(x_sample, key=key_sample)
        if key is not None:
            batch_keys = jax.random.split(key, x.shape[0])
        else:
            batch_keys = [None] * x.shape[0]
        return jax.vmap(process_sample)(x, batch_keys)

# --- Example usage ---
if __name__ == '__main__':
    # Suppose we have a batch of 8 time series, each of length 20, with embedding dimension 32.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Create our FedformerForecast model.
    model_key, run_key = jax.random.split(key)
    model = FedformerForecast(embed_dim=D, num_heads=4, dropout_p=0.1, kernel_size=3, key=model_key)

    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected output: (8, 1)
