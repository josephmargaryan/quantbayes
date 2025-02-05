import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial


# -------------------------------------------------------------------
# A simple MLP block used inside the GPT block.
# -------------------------------------------------------------------
class SimpleMLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(
        self, in_features: int, hidden_features: int, dropout_p: float, *, key
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=k2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.gelu  # or jnn.relu

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


# -------------------------------------------------------------------
# TimeGPT Block: a transformer-style (decoder) block with causal self-attention.
# -------------------------------------------------------------------
class TimeGPTBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout1: eqx.nn.Dropout

    norm2: eqx.nn.LayerNorm
    mlp: SimpleMLP
    dropout2: eqx.nn.Dropout

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout_p: float, *, key
    ):
        # Split keys for submodules.
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.norm1 = eqx.nn.LayerNorm(
            embed_dim, eps=1e-6, use_weight=True, use_bias=True
        )
        # MultiheadAttention expects inputs of shape (seq_len, embed_dim) and supports a mask.
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
            key=k2,
        )
        self.dropout1 = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.norm2 = eqx.nn.LayerNorm(
            embed_dim, eps=1e-6, use_weight=True, use_bias=True
        )
        hidden_mlp = int(embed_dim * mlp_ratio)
        self.mlp = SimpleMLP(embed_dim, hidden_mlp, dropout_p, key=k3)
        self.dropout2 = eqx.nn.Dropout(p=dropout_p, inference=False)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape (seq_len, embed_dim)
          key: Optional PRNG key.
        Returns:
          Tensor of shape (seq_len, embed_dim)
        """
        seq_len = x.shape[0]
        # Compute a causal mask: allow each time step to attend only to itself and earlier time steps.
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

        # Pre-norm before attention.
        # (We apply LayerNorm tokenwise.)
        normed = jax.vmap(self.norm1)(x)
        # Apply multihead self-attention.
        # (attn expects inputs of shape (seq_len, embed_dim).)
        # Pass the causal mask so that future tokens are masked out.
        attn_out = self.attn(normed, normed, normed, mask=causal_mask, key=key)
        # Apply dropout and add residual.
        if key is not None:
            key_drop, key_block = jax.random.split(key)
        else:
            key_drop = key_block = None
        x = x + self.dropout1(attn_out, key=key_drop)
        # Second sub-layer: pre-norm and feedforward.
        normed2 = jax.vmap(self.norm2)(x)
        # For the MLP, we want to apply it tokenwise.
        # If a key is provided, split it for each token.
        if key_block is not None:
            keys_mlp = jax.random.split(key_block, seq_len)
        else:
            keys_mlp = None
        mlp_out = jax.vmap(self.mlp)(normed2, key=keys_mlp)
        x = x + self.dropout2(mlp_out, key=key_drop)
        return x


# -------------------------------------------------------------------
# Positional Embedding: learnable positional embeddings added to the input.
# -------------------------------------------------------------------
class PositionalEmbedding(eqx.Module):
    pe: jnp.ndarray  # shape (max_len, embed_dim)

    def __init__(self, max_len: int, embed_dim: int, *, key):
        # Initialize positional embeddings (for simplicity, random initialization).
        self.pe = jax.random.normal(key, (max_len, embed_dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape (seq_len, embed_dim)
        Returns:
          Tensor of shape (seq_len, embed_dim) with positional embeddings added.
        """
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]


# -------------------------------------------------------------------
# TimeGPT Model: stacks several TimeGPT blocks, adds positional embeddings,
# and predicts a scalar from the last time step.
# -------------------------------------------------------------------
class TimeGPT(eqx.Module):
    pos_emb: PositionalEmbedding
    blocks: list[TimeGPTBlock]
    final_linear: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        max_len: int,
        *,
        key
    ):
        keys = jax.random.split(key, num_layers + 2)
        self.embed_dim = embed_dim
        self.pos_emb = PositionalEmbedding(max_len, embed_dim, key=keys[0])
        self.blocks = [
            TimeGPTBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=k)
            for k in keys[1:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape (seq_len, embed_dim) for one sample.
          key: Optional PRNG key.
        Returns:
          A scalar prediction (shape (1,)) for the sample.
        """
        # x is assumed to already have dimension D == embed_dim.
        x = self.pos_emb(x)
        # For each block, if a key is provided, split one.
        if key is not None:
            block_keys = jax.random.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, k in zip(self.blocks, block_keys):
            x = block(x, key=k)
        # Take the representation for the last time step.
        last = x[-1]  # shape (embed_dim,)
        return self.final_linear(last)  # shape (1,)


# -------------------------------------------------------------------
# Batch Wrapper: applies TimeGPT to each sample in the batch.
# -------------------------------------------------------------------
class TimeGPTForecast(eqx.Module):
    model: TimeGPT

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        max_len: int,
        *,
        key
    ):
        self.model = TimeGPT(
            num_layers, embed_dim, num_heads, mlp_ratio, dropout_p, max_len, key=key
        )

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape [N, seq_len, embed_dim]
          key: Optional PRNG key.
        Returns:
          Predictions of shape [N, 1]
        """

        def process_sample(x_sample, key_sample):
            return self.model(x_sample, key=key_sample)

        if key is not None:
            batch_keys = jax.random.split(key, x.shape[0])
        else:
            batch_keys = [None] * x.shape[0]
        return jax.vmap(process_sample)(x, batch_keys)


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Batch of 8 sequences, each of length 20, with embedding dimension 32.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Create the TimeGPTForecast model.
    # Here we set number of layers to 4, use 4 attention heads, mlp_ratio 4, dropout 0.1,
    # and max_len at least as long as seq_len.
    model_key, run_key = jax.random.split(key)
    model = TimeGPTForecast(
        num_layers=4,
        embed_dim=D,
        num_heads=4,
        mlp_ratio=4.0,
        dropout_p=0.1,
        max_len=50,
        key=model_key,
    )

    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected: (8, 1)
