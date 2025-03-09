from typing import Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

# Import your circulant layer.
from quantbayes.stochax.layers import JVPCirculantProcess


# -------------------------------------------------------------------
# Drop Path (Stochastic Depth) Function
# -------------------------------------------------------------------
def drop_path(
    x: jnp.ndarray, drop_prob: float, key: PRNGKeyArray, training: bool
) -> jnp.ndarray:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = jr.uniform(key, shape)
    binary_tensor = jnp.floor(random_tensor + keep_prob)
    return x / keep_prob * binary_tensor


# -------------------------------------------------------------------
# Patch Embedding Module
# -------------------------------------------------------------------
class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(
        self, input_channels: int, output_shape: int, patch_size: int, key: PRNGKeyArray
    ):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels, output_shape, key=key
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)
        return x


# -------------------------------------------------------------------
# Spectral Multihead Attention (adapted from your friend’s ideas)
# -------------------------------------------------------------------
class SpectralMultiheadAttention(eqx.Module):
    num_heads: int
    head_dim: int
    embed_dim: int
    seq_len: int = eqx.static_field()
    adaptive: bool
    dropout: eqx.nn.Dropout
    pre_norm: eqx.nn.LayerNorm

    # Base spectral parameters.
    base_filter: jnp.ndarray  # shape: (num_heads, freq_bins, 1)
    base_bias: jnp.ndarray  # shape: (num_heads, freq_bins, 1)
    adaptive_mlp: Optional[eqx.nn.MLP]

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        adaptive: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        self.adaptive = adaptive
        freq_bins = seq_len // 2 + 1

        key, key_filter, key_bias, key_mlp, key_dropout = jr.split(key, 5)
        self.base_filter = jr.normal(key_filter, (num_heads, freq_bins, 1))
        self.base_bias = jr.normal(key_bias, (num_heads, freq_bins, 1)) - 0.1

        if adaptive:
            self.adaptive_mlp = eqx.nn.MLP(
                in_size=embed_dim,
                out_size=num_heads * freq_bins * 2,
                width_size=embed_dim,
                depth=2,
                key=key_mlp,
            )
        else:
            self.adaptive_mlp = None

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.pre_norm = eqx.nn.LayerNorm(embed_dim)

    def complex_activation(self, z: jnp.ndarray) -> jnp.ndarray:
        mag = jnp.abs(z)
        mag_act = jax.nn.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def __call__(
        self, x: jnp.ndarray, training: bool, key: PRNGKeyArray
    ) -> jnp.ndarray:
        # x: (seq_len, embed_dim) or (B, seq_len, embed_dim)
        single_example = False
        if x.ndim == 2:
            single_example = True
            x = x[None, ...]  # add batch dim

        B, N, D = x.shape
        # Apply layer normalization over each token.
        x_norm = jax.vmap(jax.vmap(self.pre_norm))(x)
        # Reshape to separate heads.
        x_heads = x_norm.reshape(B, N, self.num_heads, self.head_dim)
        x_heads = jnp.transpose(
            x_heads, (0, 2, 1, 3)
        )  # (B, num_heads, seq_len, head_dim)
        F_fft = jnp.fft.rfft(x_heads, axis=2)

        if self.adaptive and (self.adaptive_mlp is not None):
            # Global context from all tokens.
            context = x_norm.mean(axis=1)  # shape: (B, embed_dim)
            # Vmap the MLP so that it processes each sample individually.
            adapt_params = jax.vmap(self.adaptive_mlp)(
                context
            )  # shape: (B, num_heads*freq_bins*2)
            freq_bins = F_fft.shape[2]
            adapt_params = adapt_params.reshape(B, self.num_heads, freq_bins, 2)
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            adaptive_scale = jnp.zeros((B, self.num_heads, F_fft.shape[2], 1))
            adaptive_bias = jnp.zeros((B, self.num_heads, F_fft.shape[2], 1))

        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias
        F_fft_mod = F_fft * effective_filter + effective_bias
        F_fft_nl = self.complex_activation(F_fft_mod)
        x_filtered = jnp.fft.irfft(F_fft_nl, n=self.seq_len, axis=2)
        x_filtered = jnp.transpose(x_filtered, (0, 2, 1, 3))
        x_filtered = x_filtered.reshape(B, self.seq_len, self.embed_dim)
        key, subkey = jr.split(key)
        x_filtered = self.dropout(x_filtered, inference=not training, key=subkey)
        if single_example:
            return x_filtered[0]
        return x_filtered


# -------------------------------------------------------------------
# Spectral Attention Block (Fusion of spectral attention and spectral circulant MLP)
# -------------------------------------------------------------------
class SpectralAttentionBlock(eqx.Module):
    # Branch for spectral attention.
    layer_norm1: eqx.nn.LayerNorm
    spectral_attn: SpectralMultiheadAttention

    # MLP branch: first projection replaced by the circulant layer.
    layer_norm2: eqx.nn.LayerNorm
    circulant: JVPCirculantProcess
    linear2: eqx.nn.Linear

    dropout_attn: eqx.nn.Dropout
    dropout_mlp: eqx.nn.Dropout

    # Drop path rate for stochastic depth.
    drop_path_rate: float = eqx.static_field()

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int,
        dropout_rate: float,
        drop_path_rate: float,
        *,
        key: PRNGKeyArray,
        use_spectral: bool = True,
    ):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.spectral_attn = SpectralMultiheadAttention(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            adaptive=True,
            key=key1,
        )
        self.dropout_attn = eqx.nn.Dropout(dropout_rate)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.circulant = JVPCirculantProcess(in_features=embed_dim, key=key2)
        self.linear2 = eqx.nn.Linear(embed_dim, embed_dim, key=key3)
        self.dropout_mlp = eqx.nn.Dropout(dropout_rate)
        self.drop_path_rate = drop_path_rate

    def __call__(
        self, x: jnp.ndarray, training: bool, key: PRNGKeyArray
    ) -> jnp.ndarray:
        key_attn, key_dp1, key_mlp, key_dp2 = jr.split(key, 4)
        # Apply layer norm over each token.
        norm1_out = jax.vmap(self.layer_norm1)(x)
        attn_out = self.spectral_attn(norm1_out, training, key_attn)
        attn_out = self.dropout_attn(attn_out, inference=not training, key=key_dp1)
        attn_out = drop_path(attn_out, self.drop_path_rate, key_dp1, training)
        x = x + attn_out

        norm2_out = jax.vmap(self.layer_norm2)(x)
        mlp_out = self.circulant(norm2_out)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = self.dropout_mlp(mlp_out, inference=not training, key=key_mlp)
        mlp_out = jax.vmap(self.linear2)(mlp_out)
        mlp_out = self.dropout_mlp(mlp_out, inference=not training, key=key_dp2)
        mlp_out = drop_path(mlp_out, self.drop_path_rate, key_dp2, training)
        x = x + mlp_out
        return x


# -------------------------------------------------------------------
# Vision Transformer with Fused Spectral Blocks
# -------------------------------------------------------------------
class SpectralVisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    blocks: list[SpectralAttentionBlock]
    dropout: eqx.nn.Dropout
    mlp_head: eqx.nn.Sequential
    num_layers: int

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        drop_path_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        seq_len: int,
        key: PRNGKeyArray,
        channels: int,
    ):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)
        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(key3, (1, embedding_dim))
        self.num_layers = num_layers
        self.blocks = [
            SpectralAttentionBlock(
                embed_dim=embedding_dim,
                seq_len=seq_len,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                drop_path_rate=drop_path_rate,
                key=jr.fold_in(key4, i),
            )
            for i in range(num_layers)
        ]
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.mlp_head = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=jr.fold_in(key4, 999)),
            ]
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        training: bool,
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_classes"]:
        x = self.patch_embedding(x)
        x = jnp.concatenate((self.cls_token, x), axis=0)
        x += self.positional_embedding[: x.shape[0]]
        key, subkey = jr.split(key)
        x = self.dropout(x, inference=not training, key=subkey)
        for i, block in enumerate(self.blocks):
            key, subkey = jr.split(key)
            x = block(x, training, subkey)
        x = x[0]
        x = self.mlp_head(x)
        return x, state


# -------------------------------------------------------------------
# Test Function
# -------------------------------------------------------------------
def test_vision_transformer_output_shape():
    # Configuration:
    embedding_dim = 768
    num_heads = 12
    num_layers = 12
    dropout_rate = 0.1
    drop_path_rate = 0.1
    patch_size = 16
    img_height = 224
    img_width = 224
    channels = 3
    num_patches = (img_height // patch_size) * (img_width // patch_size)
    seq_len = num_patches + 1  # including class token
    num_classes = 1000

    key = jr.PRNGKey(0)
    key, model_key, run_key, input_key = jr.split(key, 4)

    vit = SpectralVisionTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        seq_len=seq_len,
        key=model_key,
        channels=channels,
    )

    x = jr.normal(input_key, (channels, img_height, img_width))
    logits = vit(x, training=False, key=run_key)
    assert logits.shape == (
        num_classes,
    ), f"Expected output shape ({num_classes},), got {logits.shape}"
    print("Test passed!")


if __name__ == "__main__":
    test_vision_transformer_output_shape()
