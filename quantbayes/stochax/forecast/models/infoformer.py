import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


# --- Helper: Batched Linear ---
def apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    # Uses einsum to allow extra dimensions.
    return jnp.einsum("...i,oi->...o", x, linear.weight) + linear.bias


# --- Helper: Simple series decomposition ---
def decompose(x: jnp.ndarray, kernel_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    # x: [seq_len, D]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    kernel = jnp.ones((kernel_size,)) / kernel_size
    trend = jax.vmap(lambda v: jnp.convolve(v, kernel, mode="same"))(x.T).T
    seasonal = x - trend
    return seasonal, trend


# --- Batched LayerNorm Wrapper (from Autoformer) ---
class BatchedLayerNorm(eqx.Module):
    ln: eqx.nn.LayerNorm

    def __init__(
        self,
        feature_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = True,
    ):
        # Underlying layer norm expects input shape exactly equal to (feature_dim,)
        self.ln = eqx.nn.LayerNorm(
            feature_dim, eps=eps, use_weight=use_weight, use_bias=use_bias
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x may have extra dimensions before the last one; we apply layer norm tokenwise.
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        # Use vmap so that self.ln is applied to each token (row) independently.
        y_flat = jax.vmap(self.ln)(x_flat)
        return y_flat.reshape(orig_shape)


# --- Simple MLP (used for both TransformerBlock and InfoBottleneck) ---
class SimpleMLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(
        self, in_features: int, hidden_features: int, dropout_p: float, *, key
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=key2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.relu  # or use jnn.gelu

    def __call__(self, x, *, key=None):
        y = apply_linear(self.fc1, x)
        y = self.activation(y)
        y = self.dropout(y, key=key)
        y = apply_linear(self.fc2, y)
        return y


# --- Transformer Block ---
class TransformerBlock(eqx.Module):
    norm1: BatchedLayerNorm
    attn: eqx.nn.MultiheadAttention
    dropout_attn: eqx.nn.Dropout
    norm2: BatchedLayerNorm
    mlp: SimpleMLP

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        *,
        key,
    ):
        keys = jr.split(key, 5)
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
            key=keys[1],
        )
        self.dropout_attn = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.norm2 = BatchedLayerNorm(embed_dim, eps=1e-6)
        hidden_mlp = int(embed_dim * mlp_ratio)
        self.mlp = SimpleMLP(
            in_features=embed_dim,
            hidden_features=hidden_mlp,
            dropout_p=dropout_p,
            key=keys[4],
        )

    def __call__(self, x, *, key=None):
        # x: [seq_len, embed_dim]
        y = self.norm1(x)
        attn_out = self.attn(y, y, y, key=key)
        attn_out = self.dropout_attn(attn_out, key=key)
        x = x + attn_out  # residual connection
        y = self.norm2(x)
        mlp_out = self.mlp(y, key=key)
        x = x + mlp_out  # residual connection
        return x


# --- Info Bottleneck Module ---
class InfoBottleneck(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    activation: callable

    def __init__(self, in_features: int, bottleneck_dim: int, *, key):
        key1, key2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(in_features, bottleneck_dim, key=key1)
        self.fc2 = eqx.nn.Linear(bottleneck_dim, in_features, key=key2)
        self.activation = jnn.relu

    def __call__(self, x):
        y = apply_linear(self.fc1, x)
        y = self.activation(y)
        y = apply_linear(self.fc2, y)
        return y


# --- InfoFormer Block ---
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
        key,
    ):
        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 2
        key_trans, key_bottle = jr.split(key)
        self.transformer = TransformerBlock(
            embed_dim, num_heads, mlp_ratio, dropout_p, key=key_trans
        )
        self.bottleneck = InfoBottleneck(embed_dim, bottleneck_dim, key=key_bottle)

    def __call__(self, x, *, key=None):
        # x: [seq_len, embed_dim]
        if key is not None:
            key_trans, key_bottle = jr.split(key)
        else:
            key_trans = key_bottle = None
        x_trans = self.transformer(x, key=key_trans)
        # Apply the bottleneck on the entire sequence.
        bottleneck_out = self.bottleneck(x_trans)
        return x_trans + bottleneck_out


# --- Full InfoFormer Model ---
class InfoFormer(eqx.Module):
    input_proj: eqx.nn.Linear  # Projects raw input to embed_dim.
    layers: list[InfoFormerBlock]
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        bottleneck_dim: int = None,
        *,
        key,
    ):
        keys = jr.split(key, num_layers + 2)
        self.input_proj = eqx.nn.Linear(input_dim, embed_dim, key=keys[0])
        self.layers = [
            InfoFormerBlock(
                embed_dim, num_heads, mlp_ratio, dropout_p, bottleneck_dim, key=k
            )
            for k in keys[1:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x, state=None, *, key=None) -> tuple[jnp.ndarray, any]:
        # x: [seq_len, input_dim]
        x_proj = apply_linear(self.input_proj, x)  # [seq_len, embed_dim]
        if key is not None:
            layer_keys = jr.split(key, len(self.layers))
        else:
            layer_keys = [None] * len(self.layers)
        y = x_proj
        for layer, k in zip(self.layers, layer_keys):
            y = layer(y, key=k)
        # Use the last time step for final prediction.
        last_token = y[-1]
        return apply_linear(self.final_linear, last_token), state


# --- A wrapper to process batched inputs ---
class InfoFormerForecast(eqx.Module):
    model: InfoFormer

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        bottleneck_dim: int = None,
        *,
        key,
    ):
        self.model = InfoFormer(
            input_dim,
            embed_dim,
            num_layers,
            num_heads,
            mlp_ratio,
            dropout_p,
            bottleneck_dim,
            key=key,
        )

    def __call__(self, x, key, state) -> tuple[jnp.ndarray, any]:
        return self.model(x, state=state, key=key)


# --- Example usage ---
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(InfoFormerForecast)(
        input_dim=X_train.shape[-1],
        embed_dim=12,
        num_layers=2,
        num_heads=2,
        mlp_ratio=4,
        dropout_p=0.1,
        bottleneck_dim=16,  # explicitly set bottleneck_dim to half of embed_dim
        key=key,
    )
    trainer = ForecastingModel(lr=1e-3)
    trainer.fit(
        model, state, X_train, y_train, X_val, y_val, num_epochs=500, patience=100
    )
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"preds shape {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
