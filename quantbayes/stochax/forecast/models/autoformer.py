import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt


# --- Batched LayerNorm Wrapper ---
class BatchedLayerNorm(eqx.Module):
    """
    A wrapper around eqx.nn.LayerNorm that applies normalization over the
    last dimension, regardless of any extra batch dimensions.
    """

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
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        y_flat = jax.vmap(self.ln)(x_flat)
        return y_flat.reshape(orig_shape)


# --- Helper: Batched Linear ---
def apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...i,oi->...o", x, linear.weight) + linear.bias


# --- Utility: Series Decomposition ---
class SeriesDecomposition(eqx.Module):
    """
    A simple moving average based decomposition module.
    Computes a trend component using a fixed kernel via depthwise convolution.
    """

    kernel_size: int

    def __init__(self, kernel_size: int = 25):
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (B, L, D)
        D = x.shape[-1]
        # Create a normalized averaging kernel of shape (kernel_size, 1, 1)
        kernel = jnp.ones((self.kernel_size, 1, 1)) / self.kernel_size
        # Tile kernel to shape (kernel_size, 1, D) so that each channel gets its own filter.
        kernel = jnp.tile(kernel, (1, 1, D))
        # Transpose x to (B, D, L) for convolution.
        x_t = jnp.transpose(x, (0, 2, 1))
        trend = jax.lax.conv_general_dilated(
            x_t,
            kernel,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NCL", "LIO", "NCL"),
            feature_group_count=x_t.shape[
                1
            ],  # Depthwise convolution: one group per channel.
        )
        trend = jnp.transpose(trend, (0, 2, 1))
        return trend


# --- MLP Block ---
class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    activation: callable

    def __init__(
        self, in_features: int, hidden_features: int, dropout_p: float, *, key
    ):
        key1, key2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_features, in_features, key=key2)
        self.dropout = eqx.nn.Dropout(p=dropout_p, inference=False)
        self.activation = jnn.gelu

    def __call__(self, x: jnp.ndarray, *, key):
        x = apply_linear(self.fc1, x)
        x = self.activation(x)
        x = self.dropout(x, key=key)
        x = apply_linear(self.fc2, x)
        return x


# --- Auto-Correlation Attention ---
def auto_correlation(q: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    q_fft = jnp.fft.rfft(q, axis=2)
    k_fft = jnp.fft.rfft(k, axis=2)
    corr = q_fft * jnp.conjugate(k_fft)
    corr_time = jnp.fft.irfft(corr, n=q.shape[2], axis=2)
    corr_avg = jnp.mean(corr_time, axis=-1)
    return corr_avg


class AutoCorrelationAttention(eqx.Module):
    embed_dim: int
    num_heads: int
    head_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, *, key):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        keys = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, L, _ = x.shape
        Q = apply_linear(self.q_proj, x)
        K = apply_linear(self.k_proj, x)
        V = apply_linear(self.v_proj, x)
        Q = Q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        corr = auto_correlation(Q, K)
        attn_weights = jnn.softmax(corr, axis=-1)
        attn_weights = attn_weights[..., None]
        out = jnp.sum(attn_weights * V, axis=2)
        out = out.transpose(0, 2, 1).reshape(B, self.embed_dim)
        out = apply_linear(self.out_proj, out)
        out = jnp.repeat(out[:, None, :], L, axis=1)
        return out


# --- Transformer Block with Autoformer Elements ---
class TransformerBlock(eqx.Module):
    norm1: BatchedLayerNorm
    attn: AutoCorrelationAttention
    norm2: BatchedLayerNorm
    mlp: MLP
    decomposition: SeriesDecomposition

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        *,
        key,
    ):
        keys = jr.split(key, 4)
        self.norm1 = BatchedLayerNorm(embed_dim, eps=1e-6)
        self.attn = AutoCorrelationAttention(embed_dim, num_heads, key=keys[0])
        self.norm2 = BatchedLayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout_p, key=keys[1])
        self.decomposition = SeriesDecomposition(kernel_size=25)

    def __call__(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        y = self.norm1(x)
        y = self.attn(y)
        x = x + y
        trend = self.decomposition(x)
        x = x - trend
        y = self.norm2(x)
        key_mlp, _ = jr.split(key)
        y = self.mlp(y, key=key_mlp)
        x = x + y
        return x


# --- Autoformer Model ---
class Autoformer(eqx.Module):
    input_proj: eqx.nn.Linear
    layers: list[TransformerBlock]
    final_linear: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.1,
        *,
        key,
    ):
        keys = jr.split(key, num_layers + 2)
        self.input_proj = eqx.nn.Linear(input_dim, embed_dim, key=keys[0])
        self.layers = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_p, key=k)
            for k in keys[1:-1]
        ]
        self.final_linear = eqx.nn.Linear(embed_dim, 1, key=keys[-1])

    def __call__(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        B, L, _ = x.shape
        x = apply_linear(self.input_proj, x)
        for layer in self.layers:
            key, subkey = jr.split(key)
            x = layer(x, key=subkey)
        last_token = x[:, -1, :]
        out = apply_linear(self.final_linear, last_token)
        return out


# --- Forecasting Training Wrapper ---
class ForecastingModel:
    def __init__(self, lr=1e-3, loss_fn=None):
        self.lr = lr
        self.loss_fn = loss_fn if loss_fn is not None else self.mse_loss
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def mse_loss(preds, Y):
        return jnp.mean((preds - Y) ** 2)

    def _batch_forward_train(self, model, X, key):
        keys = jr.split(key, X.shape[0])

        def single_forward(x, key):
            preds = model(x[None, ...], key=key)
            return preds[0]

        preds = jax.vmap(single_forward)(X, keys)
        return preds

    def _batch_forward_inference(self, model, X, key):
        keys = jr.split(key, X.shape[0])

        def single_forward(x, key):
            preds = model(x[None, ...], key=key)
            return preds[0]

        preds = jax.vmap(single_forward)(X, keys)
        return preds

    @eqx.filter_jit
    def _train_step(self, model, X, Y, key):
        def loss_wrapper(m):
            preds = self._batch_forward_train(m, X, key)
            return self.loss_fn(preds, Y)

        loss_val, grads = eqx.filter_value_and_grad(loss_wrapper)(model)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss_val, new_model

    def _eval_step(self, model, X, Y, key):
        preds = self._batch_forward_inference(model, X, key)
        return self.loss_fn(preds, Y)

    def fit(
        self,
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        num_epochs=50,
        patience=10,
        key=jr.PRNGKey(0),
    ):
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            key, subkey = jr.split(key)
            loss_val, model = self._train_step(model, X_train, Y_train, subkey)
            self.train_losses.append(float(loss_val))
            key, subkey = jr.split(key)
            val_loss = self._eval_step(model, X_val, Y_val, subkey)
            self.val_losses.append(float(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: Train Loss = {loss_val:.4f}, Val Loss = {val_loss:.4f}"
                )
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

        return model

    def predict(self, model, X, key=jr.PRNGKey(123)):
        preds = self._batch_forward_inference(model, X, key)
        return preds

    def visualize(self, y_true, y_pred, title="Forecast vs. Ground Truth"):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, marker="o", label="Ground Truth")
        plt.plot(y_pred, marker="x", label="Predictions")
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.legend()
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    from quantbayes.fake_data import create_synthetic_time_series

    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    y_train, y_val = y_train.reshape(y_train.shape[0], -1), y_val.reshape(
        y_val.shape[0], -1
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {y_train.shape}")

    key = jr.PRNGKey(0)
    model = Autoformer(
        input_dim=10,
        num_layers=4,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=4,
        dropout_p=0.1,
        key=key,
    )
    trainer = ForecastingModel(lr=1e-3)
    model = trainer.fit(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=50,
        patience=10,
        key=jr.PRNGKey(42),
    )
    preds = trainer.predict(model, X_val, key=jr.PRNGKey(123))
    print(f"preds shape: {preds.shape}")
    trainer.visualize(y_val, preds, title="Forecast vs. Ground Truth")
