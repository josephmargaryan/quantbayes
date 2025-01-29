import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
import math
from typing import Callable, Optional
import functools

# -------------------------------
# 1. Learned Positional Encoding
# -------------------------------

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for each position in the sequence.
    """
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional embeddings added
        """
        seq_len = x.shape[1]
        # Define an embedding layer for positions
        pos_embedding = nn.Embed(
            num_embeddings=self.max_len,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0)
        )
        # Create position indices
        positions = jnp.arange(seq_len)
        # Embed positions
        pos_emb = pos_embedding(positions)  # Shape: (seq_len, d_model)
        # Expand to batch and add
        return x + pos_emb  # Broadcasting over batch dimension

# -------------------------------
# 2. Mask Generation Utilities
# -------------------------------

def make_causal_mask(seq_len: int, dtype=jnp.float32):
    """
    Create a causal mask for self-attention.

    Args:
        seq_len: Length of the sequence
        dtype: Data type of the mask

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
    """
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1).astype(dtype)
    # Expand dimensions to match attention mask shape: (batch, num_heads, seq_len, seq_len)
    return mask[None, None, :, :]  # Shape: (1, 1, seq_len, seq_len)

# -------------------------------
# 3. Decoder Block
# -------------------------------

class DecoderBlock(nn.Module):
    """
    A single decoder block consisting of multi-head self-attention and a feed-forward network.
    """
    d_model: int
    nhead: int
    dim_feedforward: int = 256

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, num_heads, seq_len, seq_len)
            deterministic: Whether or not to apply dropout
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-Head Self-Attention
        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            dtype=jnp.float32,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.1,
            deterministic=deterministic
        )
        attn_output = self_attn(x, x, x, mask=mask)  # Shape: (batch, seq_len, d_model)
        x = x + attn_output  # Residual connection
        x = nn.LayerNorm()(x)  # Layer normalization

        # Feed-Forward Network
        ff = nn.Sequential([
            nn.Dense(features=self.dim_feedforward),
            nn.relu,
            nn.Dense(features=self.d_model)
        ])
        ff_output = ff(x)  # Shape: (batch, seq_len, d_model)
        x = x + ff_output  # Residual connection
        x = nn.LayerNorm()(x)  # Layer normalization

        return x

# -------------------------------
# 4. TimeGTP Model
# -------------------------------

class TimeGTP(nn.Module):
    """
    A GPT-style Transformer model for time series forecasting.
    """
    input_dim: int        # Dimension of each time step's features
    d_model: int = 64     # Embedding (hidden) dimension
    nhead: int = 4
    num_layers: int = 2
    seq_len: int = 30
    dim_feedforward: int = 256
    max_len: int = 5000

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Attention mask of shape (1, 1, seq_len, seq_len)
            deterministic: Whether or not to apply dropout
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # 1. Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = nn.Dense(features=self.d_model)(x)

        # 2. Add positional encodings
        pos_encoding = LearnedPositionalEncoding(d_model=self.d_model, max_len=self.max_len)
        x = pos_encoding(x)  # Shape: (batch, seq_len, d_model)

        # 3. Stacked Decoder Blocks
        for _ in range(self.num_layers):
            x = DecoderBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward
            )(x, mask=mask, deterministic=deterministic)  # Shape: (batch, seq_len, d_model)

        # 4. Final Projection: Take the last time step and project to output
        x_last = x[:, -1, :]  # Shape: (batch, d_model)
        out = nn.Dense(features=1)(x_last)  # Shape: (batch, 1)
        return out


def test_time_gtp():
    # Hyperparameters
    input_dim = 1
    d_model = 64
    nhead = 4
    num_layers = 2
    seq_len = 30
    batch_size = 8

    # Initialize model
    rng = jax.random.PRNGKey(0)
    model = TimeGTP(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len
    )

    # Create dummy input tensor
    dummy_input = jax.random.normal(rng, (batch_size, seq_len, input_dim))  # Shape: (batch_size, seq_len, input_dim)

    # Initialize model parameters
    variables = model.init(rng, dummy_input)

    # Perform a forward pass
    output = model.apply(variables, dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)

# Run the test
test_time_gtp()
