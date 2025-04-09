import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax


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

    def __call__(self, x: jnp.ndarray, key, state) -> jnp.ndarray:
        B, L, _ = x.shape
        x = apply_linear(self.input_proj, x)
        for layer in self.layers:
            key, subkey = jr.split(key)
            x = layer(x, key=subkey)
        last_token = x[:, -1, :]
        out = apply_linear(self.final_linear, last_token)
        return out, state
