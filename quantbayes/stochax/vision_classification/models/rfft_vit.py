from __future__ import annotations
from typing import Tuple, Optional, Any

import math
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from quantbayes.stochax.layers import RFFTCirculant1D


def make_linear_or_spectral(
    in_features: int,
    out_features: int,
    *,
    use_spectral: bool,
    key: PRNGKeyArray,
) -> eqx.Module:
    """
    Return eqx.nn.Linear(in_features, out_features) unless
    use_spectral=True and in_features == out_features, in which case
    return an RFFTCirculant1D(in_features).
    """
    if use_spectral and (in_features == out_features):
        return RFFTCirculant1D(
            in_features=in_features,
            padded_dim=out_features,
            key=key,
        )
    else:
        return eqx.nn.Linear(in_features, out_features, key=key)


# ------------------------------------------------------------------------- #
#                             Original ViT, modified                       #
#          only in that square Linear layers can be spectralized           #
# ------------------------------------------------------------------------- #


# --------------------------- Patch Embedding --------------------------- #
class PatchEmbedding(eqx.Module):
    linear: eqx.Module  # may be Linear or RFFTCirculant1D
    patch_size: int
    in_ch: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        patch_size: int,
        key: PRNGKeyArray,
        *,
        use_spectral: bool = False,
    ):
        self.patch_size = patch_size
        self.in_ch = input_channels
        self.out_dim = output_dim

        in_dim = patch_size**2 * input_channels
        # Only becomes spectral if in_dim == output_dim and use_spectral=True.
        self.linear = make_linear_or_spectral(
            in_features=in_dim,
            out_features=output_dim,
            use_spectral=use_spectral,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        ps = self.patch_size
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=ps,
            pw=ps,
        )
        # Keep the exact original pattern: vmap over tokens
        x = jax.vmap(self.linear)(x)  # [N_patches, D]
        return x


# ------------------- Multi-Head Self-Attention (custom) ------------------- #
class MultiheadSelfAttention(eqx.Module):
    q_proj: eqx.Module
    k_proj: eqx.Module
    v_proj: eqx.Module
    out_proj: eqx.Module
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        use_spectral: bool = False,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko = jr.split(key, 4)

        # These are D -> D; with use_spectral=True they become RFFTCirculant1D
        self.q_proj = make_linear_or_spectral(
            embed_dim, embed_dim, use_spectral=use_spectral, key=kq
        )
        self.k_proj = make_linear_or_spectral(
            embed_dim, embed_dim, use_spectral=use_spectral, key=kk
        )
        self.v_proj = make_linear_or_spectral(
            embed_dim, embed_dim, use_spectral=use_spectral, key=kv
        )
        self.out_proj = make_linear_or_spectral(
            embed_dim, embed_dim, use_spectral=use_spectral, key=ko
        )

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N_tokens, D]
        D = self.embed_dim
        H = self.num_heads
        hd = self.head_dim

        # Projections: keep exact original syntax with vmap
        q = jax.vmap(self.q_proj)(x)  # [N, D]
        k = jax.vmap(self.k_proj)(x)  # [N, D]
        v = jax.vmap(self.v_proj)(x)  # [N, D]

        # Reshape to heads
        q = q.reshape(-1, H, hd).transpose(1, 0, 2)  # [H, N, hd]
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)  # [H, hd, N]
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)  # [H, N, hd]

        # Attention
        scale = 1.0 / math.sqrt(hd)
        scores = jnp.matmul(q * scale, k)  # [H, N, N]
        attn = jax.nn.softmax(scores, axis=-1)  # [H, N, N]
        ctx = jnp.matmul(attn, v)  # [H, N, hd]

        # Merge heads
        ctx = ctx.transpose(1, 0, 2).reshape(-1, D)  # [N, D]

        # Final projection, again keeping vmap
        out = jax.vmap(self.out_proj)(ctx)  # [N, D]
        return out


# --------------------------- Transformer Block --------------------------- #
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: MultiheadSelfAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
        *,
        use_spectral: bool = False,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.attention = MultiheadSelfAttention(
            embed_dim, num_heads, key=key1, use_spectral=use_spectral
        )

        # MLP: D -> hidden_dim -> D (not square in general), keep as dense Linear
        self.linear1 = eqx.nn.Linear(embed_dim, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, embed_dim, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jnp.ndarray, key: PRNGKeyArray) -> jnp.ndarray:
        # Self-attn branch
        x_norm = jax.vmap(self.layer_norm1)(x)  # [N, D]
        attn_out = self.attention(x_norm)  # [N, D]
        x = x + attn_out

        # MLP branch
        x_norm = jax.vmap(self.layer_norm2)(x)
        mlp_hidden = jax.vmap(self.linear1)(x_norm)
        mlp_hidden = jax.nn.gelu(mlp_hidden)

        k1, k2 = jr.split(key, 2)
        mlp_hidden = self.dropout1(mlp_hidden, key=k1)
        mlp_out = jax.vmap(self.linear2)(mlp_hidden)
        mlp_out = self.dropout2(mlp_out, key=k2)

        x = x + mlp_out
        return x


# ------------------------------- ViT Model ------------------------------- #
class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray  # [1+N, D]
    cls_token: jnp.ndarray  # [1, D]
    attention_blocks: Tuple[AttentionBlock, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    num_layers: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    use_spectral_proj: bool = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
        *,
        channels: int = 3,
        use_spectral_proj: bool = False,
    ):
        """
        use_spectral_proj=False → original ViT (all eqx.nn.Linear).
        use_spectral_proj=True  → replace all Linear with in_dim==out_dim by
                                  RFFTCirculant1D (Q/K/V/out proj, and patch
                                  embedding only if square).
        """
        k1, k2, k3, k4, k5 = jr.split(key, 5)

        self.patch_embedding = PatchEmbedding(
            channels,
            embedding_dim,
            patch_size,
            k1,
            use_spectral=use_spectral_proj,
        )

        # +1 for CLS token; keep an explicit [1+N, D] param for positional embedding
        self.positional_embedding = jr.normal(k2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(k3, (1, embedding_dim))
        self.num_layers = num_layers
        self.embed_dim = embedding_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channels = channels
        self.use_spectral_proj = bool(use_spectral_proj)

        blocks = []
        block_keys = jr.split(k4, num_layers)
        for kb in block_keys:
            blocks.append(
                AttentionBlock(
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    dropout_rate,
                    kb,
                    use_spectral=use_spectral_proj,
                )
            )
        self.attention_blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.head = eqx.nn.Linear(embedding_dim, num_classes, key=k5)

    def __call__(
        self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state
    ):
        # Embed patches
        x = self.patch_embedding(x)  # [N_patches, D]

        # Prepend CLS and add pos
        x = jnp.concatenate((self.cls_token, x), axis=0)  # [1+N, D]
        pos = self.positional_embedding[: x.shape[0]]  # safe slice
        x = x + pos

        # Transformer
        keys = jr.split(key, self.num_layers + 1)
        x = self.dropout(x, key=keys[0])
        for block, k in zip(self.attention_blocks, keys[1:]):
            x = block(x, key=k)

        # CLS -> norm -> head
        x_cls = self.norm(x[0])  # [D]
        logits = self.head(x_cls)  # [C]
        return logits, state


if __name__ == "__main__":
    import jax.random as jr
    import equinox as eqx

    IMAGE_SIZE = 32
    PATCH_SIZE = 4
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64

    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_HEADS = 4
    NUM_LAYERS = 6
    DROPOUT = 0.1
    NUM_CLASSES = 10

    key = jr.PRNGKey(0)
    k_dense, k_rfft = jr.split(key, 2)

    # Spectral ViT (RFFTCirculant1D on all D→D linears)
    model_rfft, state_rfft = eqx.nn.make_with_state(VisionTransformer)(
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=NUM_CLASSES,
        key=k_rfft,
        channels=3,
        use_spectral_proj=True,
    )

    # Baseline dense ViT
    model, state = eqx.nn.make_with_state(VisionTransformer)(
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=NUM_CLASSES,
        key=k_dense,
        channels=3,
        use_spectral_proj=False,
    )
