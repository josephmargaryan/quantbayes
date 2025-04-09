import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------
# Batched LayerNorm Wrapper (to avoid per-token Python loops)
# -------------------------------------------------------------------
class BatchedLayerNorm(eqx.Module):
    ln: eqx.nn.LayerNorm

    def __init__(
        self,
        feature_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = True,
    ):
        self.ln = eqx.nn.LayerNorm(
            feature_dim, eps=eps, use_weight=use_weight, use_bias=use_bias
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (..., feature_dim)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        y_flat = jax.vmap(self.ln)(x_flat)
        return y_flat.reshape(orig_shape)


# -------------------------------------------------------------------
# Input Projection: maps raw input to embed_dim.
# -------------------------------------------------------------------
class InputProjection(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, input_dim: int, embed_dim: int, *, key):
        self.linear = eqx.nn.Linear(input_dim, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use einsum to handle batched inputs.
        return jnp.einsum("...i,oi->...o", x, self.linear.weight) + self.linear.bias


# -------------------------------------------------------------------
# Positional Embedding: learnable positional embeddings.
# -------------------------------------------------------------------
class PositionalEmbedding(eqx.Module):
    pe: jnp.ndarray  # shape (max_len, embed_dim)

    def __init__(self, max_len: int, embed_dim: int, *, key):
        self.pe = jr.normal(key, (max_len, embed_dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]


# -------------------------------------------------------------------
# Simple MLP Block (used inside the GPT block)
# -------------------------------------------------------------------
class SimpleMLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(
        self, in_features: int, hidden_features: int, dropout_p: float, *, key
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=k2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.gelu  # or jnn.relu

    def __call__(self, x, *, key=None):
        y = self.fc1(x)
        y = self.activation(y)
        if key is not None:
            key1, key2 = jr.split(key)
        else:
            key1 = key2 = None
        y = self.dropout(y, key=key1)
        y = self.fc2(y)
        y = self.dropout(y, key=key2)
        return y


# -------------------------------------------------------------------
# TimeGPT Block: a transformer-style decoder block with causal self-attention.
# -------------------------------------------------------------------
class TimeGPTBlock(eqx.Module):
    norm1: BatchedLayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout1: eqx.nn.Dropout
    norm2: BatchedLayerNorm
    mlp: SimpleMLP
    dropout2: eqx.nn.Dropout

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout_p: float, *, key
    ):
        # Split keys once.
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.norm1 = BatchedLayerNorm(embed_dim, eps=1e-6)
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
        self.norm2 = BatchedLayerNorm(embed_dim, eps=1e-6)
        hidden_mlp = int(embed_dim * mlp_ratio)
        self.mlp = SimpleMLP(
            in_features=embed_dim,
            hidden_features=hidden_mlp,
            dropout_p=dropout_p,
            key=k3,
        )
        self.dropout2 = eqx.nn.Dropout(p=dropout_p, inference=False)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        seq_len = x.shape[0]
        # Create a causal mask so that token i can only attend to tokens ≤ i.
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        # Pre-norm using our batched LayerNorm.
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed, mask=causal_mask, key=key)
        if key is not None:
            key_drop, key_rest = jr.split(key)
        else:
            key_drop = key_rest = None
        x = x + self.dropout1(attn_out, key=key_drop)
        # Apply norm2 tokenwise.
        normed2 = self.norm2(x)
        # Here we apply the MLP tokenwise.
        mlp_out = jax.vmap(self.mlp)(
            normed2, key=(None if key_rest is None else jr.split(key_rest, seq_len))
        )
        x = x + self.dropout2(mlp_out, key=key_drop)
        return x


# -------------------------------------------------------------------
# TimeGPT Model: stacks TimeGPT blocks, adds positional embeddings, and outputs a forecast.
# -------------------------------------------------------------------
class TimeGPT(eqx.Module):
    input_proj: InputProjection  # Projects raw input to embed_dim.
    pos_emb: PositionalEmbedding
    blocks: list[TimeGPTBlock]
    final_linear: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        max_len: int,
        *,
        key,
    ):
        keys = jr.split(key, num_layers + 3)
        self.embed_dim = embed_dim
        self.input_proj = InputProjection(input_dim, embed_dim, key=keys[0])
        self.pos_emb = PositionalEmbedding(max_len, embed_dim, key=keys[1])
        self.blocks = [
            TimeGPTBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=k)
            for k in keys[2:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        Args:
          x: Input tensor of shape (seq_len, input_dim) for one sample.
          key: Optional PRNG key.
        Returns:
          A scalar prediction (shape (1,)) for the sample.
        """
        x = self.input_proj(x)  # shape: (seq_len, embed_dim)
        x = self.pos_emb(x)
        if key is not None:
            block_keys = jr.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, k in zip(self.blocks, block_keys):
            x = block(x, key=k)
        last = x[-1]  # shape: (embed_dim,)
        return self.final_linear(last)


# -------------------------------------------------------------------
# Batch Wrapper: TimeGPTForecast
# -------------------------------------------------------------------
class TimeGPTForecast(eqx.Module):
    model: TimeGPT

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_p: float,
        max_len: int,
        *,
        key,
    ):
        self.model = TimeGPT(
            input_dim,
            embed_dim,
            num_layers,
            num_heads,
            mlp_ratio,
            dropout_p,
            max_len,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, any]:
        """
        Args:
          x: Input tensor of shape [N, seq_len, input_dim]
          key: Optional PRNG key.
        Returns:
          A tuple (predictions, state) where predictions has shape [N, 1].
        """
        # Our training framework already vmaps over samples; x here is one sample.
        return self.model(x, key=key), state


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    # Raw input shape: [N, seq_len, input_dim] with input_dim == 1.
    # The InputProjection will map it internally to embed_dim (here, 16).
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(TimeGPTForecast)(
        input_dim=1,
        embed_dim=16,
        num_layers=4,
        num_heads=2,
        mlp_ratio=4,
        dropout_p=0.1,
        max_len=10,
        key=key,
    )
    trainer = ForecastingModel(lr=1e-3)
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=500,
        patience=100,
        key=jr.PRNGKey(42),
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
