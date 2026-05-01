from __future__ import annotations

from typing import Tuple, Literal

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.vision_backbones.dino.vit_dino_eqx import PatchEmbedding, Pool

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
    key,
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
        key,
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
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = make_linear_or_svd(
            self.embed_dim,
            self.embed_dim,
            use_svd=use_svd,
            key=kq,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.k_proj = make_linear_or_svd(
            self.embed_dim,
            self.embed_dim,
            use_svd=use_svd,
            key=kk,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.v_proj = make_linear_or_svd(
            self.embed_dim,
            self.embed_dim,
            use_svd=use_svd,
            key=kv,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.out_proj = make_linear_or_svd(
            self.embed_dim,
            self.embed_dim,
            use_svd=use_svd,
            key=ko,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D, H, hd = self.embed_dim, self.num_heads, self.head_dim
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        q = q.reshape(-1, H, hd).transpose(1, 0, 2)
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)

        scale = 1.0 / math.sqrt(hd)
        attn = jax.nn.softmax(jnp.matmul(q * scale, k), axis=-1)
        ctx = jnp.matmul(attn, v).transpose(1, 0, 2).reshape(-1, D)
        return jax.vmap(self.out_proj)(ctx)


class AttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: MultiheadSelfAttention
    norm2: eqx.nn.LayerNorm
    mlp1: eqx.Module
    mlp2: eqx.Module
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    ls1: jnp.ndarray | None
    ls2: jnp.ndarray | None

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        key,
        layer_scale_init: float | None = 1e-5,
        use_svd_attention: bool = False,
        use_svd_mlp: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k_attn, k_mlp1, k_mlp2 = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(
            embed_dim,
            num_heads,
            key=k_attn,
            use_svd=use_svd_attention,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.mlp1 = make_linear_or_svd(
            embed_dim,
            hidden_dim,
            use_svd=use_svd_mlp,
            key=k_mlp1,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.mlp2 = make_linear_or_svd(
            hidden_dim,
            embed_dim,
            use_svd=use_svd_mlp,
            key=k_mlp2,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)
        if layer_scale_init and layer_scale_init > 0:
            self.ls1 = jnp.ones((embed_dim,), dtype=jnp.float32) * float(
                layer_scale_init
            )
            self.ls2 = jnp.ones((embed_dim,), dtype=jnp.float32) * float(
                layer_scale_init
            )
        else:
            self.ls1 = None
            self.ls2 = None

    def __call__(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        k1, k2 = jr.split(key)
        a = self.attn(jax.vmap(self.norm1)(x))
        a = self.dropout1(a, key=k1)
        x = x + (a * self.ls1 if self.ls1 is not None else a)

        m = jax.vmap(self.mlp2)(
            jax.vmap(jax.nn.gelu)(jax.vmap(self.mlp1)(jax.vmap(self.norm2)(x)))
        )
        m = self.dropout2(m, key=k2)
        return x + (m * self.ls2 if self.ls2 is not None else m)


class SVDDinoVisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    register_tokens: jnp.ndarray | None
    attention_blocks: Tuple[AttentionBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.Module
    dropout: eqx.nn.Dropout

    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    n_register_tokens: int = eqx.field(static=True)
    pool: Pool = eqx.field(static=True)
    svd_mode: SVDMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        n_register_tokens: int = 0,
        dropout_rate: float = 0.0,
        layer_scale_init: float | None = 1e-5,
        pool: Pool = "cls",
        channels: int = 3,
        key,
        svd_mode: SVDMode = "attn_mlp",
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k_patch, k_misc = jr.split(key)
        self.embed_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.n_register_tokens = int(n_register_tokens)
        self.pool = pool
        self.svd_mode = svd_mode

        use_patch_svd = svd_mode == "all_linear"
        use_attn_svd = svd_mode in {"attn_only", "attn_mlp", "all_linear"}
        use_mlp_svd = svd_mode in {"attn_mlp", "all_linear"}
        use_head_svd = svd_mode == "all_linear"

        self.patch_embedding = PatchEmbedding(
            channels,
            embedding_dim,
            patch_size,
            key=k_patch,
        )
        if use_patch_svd:
            self.patch_embedding = eqx.tree_at(
                lambda m: m.linear,
                self.patch_embedding,
                make_linear_or_svd(
                    patch_size * patch_size * channels,
                    embedding_dim,
                    use_svd=True,
                    key=jr.fold_in(k_misc, 11),
                    rank=svd_rank,
                    rank_ratio=svd_rank_ratio,
                    min_rank=svd_min_rank,
                    max_rank=svd_rank_cap,
                    alpha_init=alpha_init,
                ),
            )

        self.positional_embedding = jr.normal(k_misc, (1 + num_patches, embedding_dim))
        self.cls_token = jr.normal(jr.fold_in(k_misc, 1), (1, embedding_dim))
        self.register_tokens = (
            jr.normal(jr.fold_in(k_misc, 2), (n_register_tokens, embedding_dim))
            if n_register_tokens > 0
            else None
        )

        block_keys = jr.split(jr.fold_in(k_misc, 3), num_layers)
        self.attention_blocks = tuple(
            AttentionBlock(
                embedding_dim,
                hidden_dim,
                num_heads,
                dropout_rate,
                key=kb,
                layer_scale_init=layer_scale_init,
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
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.head = make_linear_or_svd(
            embedding_dim,
            num_classes,
            use_svd=use_head_svd,
            key=jr.fold_in(k_misc, 4),
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def _tokens(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        ps = self.patch_size
        _, H, W = x.shape
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be multiples of patch size={ps}; got {(H, W)}.")
        patches = self.patch_embedding(x)
        seq = jnp.concatenate([self.cls_token, patches], axis=0)
        seq = seq + self.positional_embedding[: seq.shape[0]]
        if self.register_tokens is not None and self.register_tokens.shape[0] > 0:
            seq = jnp.concatenate([seq[:1], self.register_tokens, seq[1:]], axis=0)
        return self.dropout(seq, key=key)

    def forward_features(self, x: jnp.ndarray, key, state):
        ktoks, *krest = jr.split(key, self.num_layers + 1)
        x = self._tokens(x, key=ktoks)
        for block, kb in zip(self.attention_blocks, krest):
            x = block(x, key=kb)
        x = jax.vmap(self.norm)(x)
        if self.pool == "cls":
            feat = x[0]
        else:
            start = 1 + (
                self.register_tokens.shape[0] if self.register_tokens is not None else 0
            )
            feat = jnp.mean(x[start:], axis=0)
        return feat, state

    def forward_head(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.head(feat)

    def __call__(self, x: jnp.ndarray, key, state):
        feat, state = self.forward_features(x, key, state)
        logits = self.forward_head(feat)
        return logits, state
