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
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    kernel = jnp.ones((kernel_size,)) / kernel_size
    trend = jax.vmap(
        lambda v: jnp.convolve(jnp.reshape(v, (-1,)), kernel, mode="same")
    )(x.T).T
    seasonal = x - trend
    return seasonal, trend


# --- MLP ---
class MLP(eqx.Module):
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
        self.activation = jnn.relu

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


# --- TransformerBlock ---
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
        key,
    ):
        key_norm1, key_attn, key_dropout, key_norm2, key_mlp = jr.split(key, 5)
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
        if key is not None:
            key_attn, key_mlp = jr.split(key)
        else:
            key_attn = key_mlp = None
        y = jax.vmap(self.norm1)(x)
        attn_out = self.attn(y, y, y, key=key_attn)
        attn_out = self.dropout_attn(attn_out, key=key_attn)
        x = x + attn_out
        y = jax.vmap(self.norm2)(x)
        mlp_out = jax.vmap(self.mlp)(
            y, key=None if key_mlp is None else jr.split(key_mlp, x.shape[0])
        )
        x = x + mlp_out
        return x


# --- Fedformer ---
class Fedformer(eqx.Module):
    input_proj: eqx.nn.Linear
    kernel_size: int
    seasonal_block: TransformerBlock
    trend_net: MLP
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        kernel_size: int = 3,
        *,
        key,
    ):
        key_proj, key_seasonal, key_trend, key_final = jr.split(key, 4)
        self.input_proj = eqx.nn.Linear(input_dim, embed_dim, key=key_proj)
        self.kernel_size = kernel_size
        self.seasonal_block = TransformerBlock(
            embed_dim, num_heads, mlp_ratio, dropout_p, key=key_seasonal
        )
        self.trend_net = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout_p=dropout_p,
            key=key_trend,
        )
        self.final_linear = eqx.nn.Linear(embed_dim * 2, 1, key=key_final)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # x: [seq_len, D]
        # Use our helper to project x
        x_proj = apply_linear(self.input_proj, x)  # [seq_len, embed_dim]
        seasonal, trend = decompose(
            x_proj, self.kernel_size
        )  # both [seq_len, embed_dim]
        if key is not None:
            key_seasonal, key_trend = jr.split(key)
        else:
            key_seasonal = key_trend = None
        seasonal_out = self.seasonal_block(
            seasonal, key=key_seasonal
        )  # [seq_len, embed_dim]
        trend_out = self.trend_net(trend[-1], key=key_trend)  # [embed_dim]
        last_seasonal = seasonal_out[-1]  # [embed_dim]
        final_feat = jnp.concatenate(
            [last_seasonal, trend_out], axis=-1
        )  # [2*embed_dim]
        return apply_linear(self.final_linear, final_feat)  # [1]


# --- A wrapper for batched inputs ---
# Our training framework vmaps over samples.
class FedformerForecast(eqx.Module):
    model: Fedformer

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        kernel_size: int = 3,
        *,
        key,
    ):
        self.model = Fedformer(
            input_dim, embed_dim, num_heads, mlp_ratio, dropout_p, kernel_size, key=key
        )

    def __call__(self, x: jnp.ndarray, key, state) -> tuple[jnp.ndarray, eqx.nn.State]:
        # Return prediction and pass through the state.
        return self.model(x, key=key), state


# --- Example usage ---
if __name__ == "__main__":
    import jax.random as jr

    from quantbayes.fake_data import create_synthetic_time_series
    from quantbayes.stochax.forecast import ForecastingModel  # your training wrapper

    # Create synthetic data.
    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Increase model capacity a bit for diagnosis.
    key = jr.PRNGKey(0)
    model, state = eqx.nn.make_with_state(FedformerForecast)(
        input_dim=X_train.shape[-1],
        embed_dim=16,  # Increased embed_dim for more capacity.
        num_heads=2,  # Using 2 heads.
        mlp_ratio=4,
        dropout_p=0.1,
        kernel_size=3,
        key=key,
    )

    trainer = ForecastingModel(lr=1e-3)
    # Train for 100 epochs with early stopping patience of 20.
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        patience=20,
        key=jr.PRNGKey(42),
    )

    # Predict on validation set.
    preds = trainer.predict(model, state, X_val, key=jr.PRNGKey(123))
    print(f"Predictions shape: {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
