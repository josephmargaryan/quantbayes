import math
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from functools import partial


# A simple MLP block with one hidden layer, dropout, and an activation.
class MLP(eqx.Module):
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
        self.activation = jnn.relu  # or jnn.gelu

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


# A transformer-style block that uses multihead self-attention and an MLP.
class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout_attn: eqx.nn.Dropout

    norm2: eqx.nn.LayerNorm
    mlp: MLP

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
        self.mlp = MLP(
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

        # Pre-norm for the MLP block (again tokenwise)
        y = jax.vmap(self.norm2)(x)
        # Now apply the MLP tokenwise; since self.mlp expects input shape (embed_dim,)
        if key_mlp is not None:
            keys_mlp = jax.random.split(key_mlp, y.shape[0])
        else:
            keys_mlp = None
        mlp_out = jax.vmap(self.mlp)(y, key=keys_mlp)
        x = x + mlp_out  # residual connection
        return x


# The full simplified Autoformer model.
# It takes inputs of shape [N, seq_len, D] and outputs a prediction for the last time step: [N, 1].
class Autoformer(eqx.Module):
    layers: list[TransformerBlock]
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        *,
        key
    ):
        keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=k)
            for k in keys[:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x, *, key=None):
        """
        Args:
          x: Input array of shape [N, seq_len, D]
          key: Optional PRNG key. If provided, we split it for the transformer blocks.
        Returns:
          Output array of shape [N, 1] computed from the last time step.
        """

        def process_sample(x_sample, key_sample):
            keys = (
                jax.random.split(key_sample, len(self.layers))
                if key_sample is not None
                else [None] * len(self.layers)
            )
            y = x_sample  # shape: [seq_len, D]
            for layer, k in zip(self.layers, keys):
                y = layer(y, key=k)
            last_token = y[-1]  # shape: [D]
            return self.final_linear(last_token)

        if key is not None:
            batch_keys = jax.random.split(key, x.shape[0])
        else:
            batch_keys = [None] * x.shape[0]
        out = jax.vmap(process_sample)(x, batch_keys)
        return out


# === Example usage ===
if __name__ == "__main__":
    # For example: a batch of 8 sequences, each with length 20 and embedding dimension 32.
    N, seq_len, D = 8, 20, 32
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (N, seq_len, D))

    model_key, run_key = jax.random.split(key)
    model = Autoformer(
        num_layers=3, embed_dim=D, num_heads=4, dropout_p=0.1, key=model_key
    )

    preds = model(x, key=run_key)
    print("Predictions shape:", preds.shape)  # Expected output: (8, 1)
