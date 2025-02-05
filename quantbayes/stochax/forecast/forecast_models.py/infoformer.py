import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial


# --- Helper: Standard Transformer Block ---
# (This block is essentially the same as used in the previous examples.)
class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout_attn: eqx.nn.Dropout

    norm2: eqx.nn.LayerNorm
    mlp: eqx.Module  # our simple MLP block

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        *,
        key
    ):
        key_norm1, key_attn, key_dropout, key_norm2, key_mlp = jax.random.split(key, 5)
        self.norm1 = eqx.nn.LayerNorm(
            embed_dim, eps=1e-6, use_weight=True, use_bias=True
        )
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
        self.norm2 = eqx.nn.LayerNorm(
            embed_dim, eps=1e-6, use_weight=True, use_bias=True
        )
        hidden_mlp = int(embed_dim * mlp_ratio)
        # Define a simple MLP block as used before.
        self.mlp = SimpleMLP(
            in_features=embed_dim,
            hidden_features=hidden_mlp,
            dropout_p=dropout_p,
            key=key_mlp,
        )

    def __call__(self, x, *, key=None):
        # x: [seq_len, embed_dim]
        if key is not None:
            key_attn, key_mlp = jax.random.split(key)
        else:
            key_attn = key_mlp = None

        # Pre-norm and attention (apply norm tokenwise)
        y = jax.vmap(self.norm1)(x)
        attn_out = self.attn(y, y, y, key=key_attn)
        attn_out = self.dropout_attn(attn_out, key=key_attn)
        x = x + attn_out  # residual connection

        # Feed-forward (MLP) block with pre-norm (applied tokenwise)
        y = jax.vmap(self.norm2)(x)
        # Split key per token if provided.
        if key_mlp is not None:
            keys_mlp = jax.random.split(key_mlp, y.shape[0])
        else:
            keys_mlp = None
        mlp_out = jax.vmap(self.mlp)(y, key=keys_mlp)
        x = x + mlp_out  # residual connection
        return x


# --- A simple MLP block (used both in TransformerBlock and InfoBottleneck) ---
class SimpleMLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(
        self, in_features: int, hidden_features: int, dropout_p: float, *, key
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=key2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.relu  # or use jnn.gelu

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


# --- The Information Bottleneck Module ---
# This module forces a token's representation through a lower-dimensional bottleneck.
class InfoBottleneck(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    activation: callable

    def __init__(self, in_features: int, bottleneck_dim: int, *, key):
        key1, key2 = jax.random.split(key, 2)
        self.fc1 = eqx.nn.Linear(in_features, bottleneck_dim, key=key1)
        self.fc2 = eqx.nn.Linear(bottleneck_dim, in_features, key=key2)
        self.activation = jnn.relu

    def __call__(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y


# --- The InfoFormer Block ---
# This block first applies a transformer block then passes the output
# through an information bottleneck (applied tokenwise) and adds a residual.
class InfoFormerBlock(eqx.Module):
    transformer: TransformerBlock
    bottleneck: InfoBottleneck

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        bottleneck_dim: int = None,
        *,
        key
    ):
        # If bottleneck_dim is not specified, choose half of embed_dim.
        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 2
        key_trans, key_bottle = jax.random.split(key)
        self.transformer = TransformerBlock(
            embed_dim, num_heads, mlp_ratio, dropout_p, key=key_trans
        )
        self.bottleneck = InfoBottleneck(embed_dim, bottleneck_dim, key=key_bottle)

    def __call__(self, x, *, key=None):
        # x: [seq_len, embed_dim]
        if key is not None:
            key_trans, key_bottle = jax.random.split(key)
        else:
            key_trans = key_bottle = None
        x_trans = self.transformer(x, key=key_trans)
        # Apply bottleneck tokenwise:
        bottleneck_out = jax.vmap(self.bottleneck)(x_trans)
        # Combine with a residual connection:
        return x_trans + bottleneck_out


# --- The Full InfoFormer Model ---
# This model stacks several InfoFormerBlocks and then predicts from the last time step.
class InfoFormer(eqx.Module):
    layers: list[InfoFormerBlock]
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        bottleneck_dim: int = None,
        *,
        key
    ):
        # Split keys for each block plus one for final linear.
        keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            InfoFormerBlock(
                embed_dim, num_heads, mlp_ratio, dropout_p, bottleneck_dim, key=k
            )
            for k in keys[:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x, *, key=None):
        """
        Args:
          x: Input tensor of shape [seq_len, embed_dim] (one sample).
          key: Optional PRNG key.
        Returns:
          A prediction of shape [1] for the last time step.
        """
        # For each layer we optionally split the key.
        if key is not None:
            layer_keys = jax.random.split(key, len(self.layers))
        else:
            layer_keys = [None] * len(self.layers)
        y = x  # shape: [seq_len, embed_dim]
        for layer, k in zip(self.layers, layer_keys):
            y = layer(y, key=k)
        # Use the representation from the final time step.
        last_token = y[-1]
        return self.final_linear(last_token)


# --- A wrapper to handle batched inputs ---
class InfoFormerForecast(eqx.Module):
    model: InfoFormer

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        bottleneck_dim: int = None,
        *,
        key
    ):
        self.model = InfoFormer(
            num_layers,
            embed_dim,
            num_heads,
            mlp_ratio,
            dropout_p,
            bottleneck_dim,
            key=key,
        )

    def __call__(self, x, *, key=None):
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


# --- Example usage ---
if __name__ == "__main__":
    # Suppose we have a batch of 8 sequences, each with length 20 and embedding dimension 32.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    # Create our InfoFormerForecast model. For example, using 3 blocks and 4 attention heads.
    model_key, run_key = jax.random.split(key)
    model = InfoFormerForecast(
        num_layers=3,
        embed_dim=D,
        num_heads=4,
        dropout_p=0.1,
        bottleneck_dim=D // 2,
        key=model_key,
    )

    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected output: (8, 1)
