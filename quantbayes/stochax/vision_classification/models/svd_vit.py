from __future__ import annotations

from typing import Tuple, Literal

import math
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

try:
    from quantbayes.stochax.layers import SVDDense
except Exception:
    from quantbayes.stochax.layers.spectral_layers import SVDDense  # type: ignore


SVDMode = Literal["none", "attn_only", "attn_mlp", "all_linear"]


def _resolve_rank(
    in_features: int,
    out_features: int,
    *,
    rank: int | None,
    rank_ratio: float,
    min_rank: int,
    max_rank: int | None,
) -> int:
    full = min(int(in_features), int(out_features))
    if rank is not None:
        r = int(rank)
    else:
        r = max(int(min_rank), int(round(full * float(rank_ratio))))
    if max_rank is not None:
        r = min(r, int(max_rank))
    return max(1, min(full, r))


def make_linear_or_svd(
    in_features: int,
    out_features: int,
    *,
    use_svd: bool,
    key: PRNGKeyArray,
    rank: int | None = None,
    rank_ratio: float = 0.25,
    min_rank: int = 16,
    max_rank: int | None = None,
    alpha_init: float = 1.0,
) -> eqx.Module:
    dense = eqx.nn.Linear(in_features, out_features, key=key)
    if not use_svd:
        return dense
    r = _resolve_rank(
        in_features,
        out_features,
        rank=rank,
        rank_ratio=rank_ratio,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    return SVDDense.from_linear(dense, rank=r, alpha_init=alpha_init)


class PatchEmbedding(eqx.Module):
    linear: eqx.Module
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
        use_svd: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        self.patch_size = int(patch_size)
        self.in_ch = int(input_channels)
        self.out_dim = int(output_dim)
        in_dim = self.patch_size**2 * self.in_ch
        self.linear = make_linear_or_svd(
            in_dim,
            self.out_dim,
            use_svd=use_svd,
            key=key,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
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
        return jax.vmap(self.linear)(x)


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
        use_svd: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko = jr.split(key, 4)
        self.q_proj = make_linear_or_svd(
            embed_dim,
            embed_dim,
            use_svd=use_svd,
            key=kq,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.k_proj = make_linear_or_svd(
            embed_dim,
            embed_dim,
            use_svd=use_svd,
            key=kk,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.v_proj = make_linear_or_svd(
            embed_dim,
            embed_dim,
            use_svd=use_svd,
            key=kv,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.out_proj = make_linear_or_svd(
            embed_dim,
            embed_dim,
            use_svd=use_svd,
            key=ko,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.num_heads = int(num_heads)
        self.embed_dim = int(embed_dim)
        self.head_dim = self.embed_dim // self.num_heads

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D = self.embed_dim
        H = self.num_heads
        hd = self.head_dim

        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        q = q.reshape(-1, H, hd).transpose(1, 0, 2)
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)

        scale = 1.0 / math.sqrt(hd)
        scores = jnp.matmul(q * scale, k)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.matmul(attn, v)

        ctx = ctx.transpose(1, 0, 2).reshape(-1, D)
        return jax.vmap(self.out_proj)(ctx)


class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: MultiheadSelfAttention
    linear1: eqx.Module
    linear2: eqx.Module
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
        use_svd_attention: bool = False,
        use_svd_mlp: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.attention = MultiheadSelfAttention(
            embed_dim,
            num_heads,
            key=key1,
            use_svd=use_svd_attention,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.linear1 = make_linear_or_svd(
            embed_dim,
            hidden_dim,
            use_svd=use_svd_mlp,
            key=key2,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.linear2 = make_linear_or_svd(
            hidden_dim,
            embed_dim,
            use_svd=use_svd_mlp,
            key=key3,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jnp.ndarray, key: PRNGKeyArray) -> jnp.ndarray:
        x_norm = jax.vmap(self.layer_norm1)(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out

        x_norm = jax.vmap(self.layer_norm2)(x)
        mlp_hidden = jax.vmap(self.linear1)(x_norm)
        mlp_hidden = jax.nn.gelu(mlp_hidden)

        k1, k2 = jr.split(key, 2)
        mlp_hidden = self.dropout1(mlp_hidden, key=k1)
        mlp_out = jax.vmap(self.linear2)(mlp_hidden)
        mlp_out = self.dropout2(mlp_out, key=k2)

        return x + mlp_out


class SVDVisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: Tuple[AttentionBlock, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    head: eqx.Module

    num_layers: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    svd_mode: SVDMode = eqx.field(static=True)

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
        svd_mode: SVDMode = "attn_mlp",
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k1, k2, k3, k4, k5 = jr.split(key, 5)

        use_patch_svd = svd_mode == "all_linear"
        use_attn_svd = svd_mode in {"attn_only", "attn_mlp", "all_linear"}
        use_mlp_svd = svd_mode in {"attn_mlp", "all_linear"}
        use_head_svd = svd_mode == "all_linear"

        self.patch_embedding = PatchEmbedding(
            channels,
            embedding_dim,
            patch_size,
            k1,
            use_svd=use_patch_svd,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )

        self.positional_embedding = jr.normal(k2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(k3, (1, embedding_dim))
        self.num_layers = int(num_layers)
        self.embed_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.channels = int(channels)
        self.svd_mode = svd_mode

        block_keys = jr.split(k4, num_layers)
        self.attention_blocks = tuple(
            AttentionBlock(
                embedding_dim,
                hidden_dim,
                num_heads,
                dropout_rate,
                kb,
                use_svd_attention=use_attn_svd,
                use_svd_mlp=use_mlp_svd,
                svd_rank=svd_rank,
                svd_rank_ratio=svd_rank_ratio,
                svd_min_rank=svd_min_rank,
                svd_rank_cap=svd_rank_cap,
                alpha_init=alpha_init,
            )
            for kb in block_keys
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.head = make_linear_or_svd(
            embedding_dim,
            num_classes,
            use_svd=use_head_svd,
            key=k5,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )

    def forward_features(
        self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state
    ):
        x = self.patch_embedding(x)
        x = jnp.concatenate((self.cls_token, x), axis=0)
        x = x + self.positional_embedding[: x.shape[0]]

        keys = jr.split(key, self.num_layers + 1)
        x = self.dropout(x, key=keys[0])
        for block, k in zip(self.attention_blocks, keys[1:]):
            x = block(x, key=k)

        return self.norm(x[0]), state

    def forward_head(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.head(feat)

    def __call__(
        self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state
    ):
        feat, state = self.forward_features(x, key, state)
        logits = self.forward_head(feat)
        return logits, state
