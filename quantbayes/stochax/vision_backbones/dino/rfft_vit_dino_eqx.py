from __future__ import annotations
from typing import Tuple, Literal

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.layers.spectral_layers import RFFTCirculant1D


Pool = Literal["cls", "mean_patch"]


def make_linear_or_spectral(
    in_features: int,
    out_features: int,
    *,
    key,
    use_spectral: bool = True,
) -> eqx.Module:
    """Use RFFTCirculant1D only for square D->D maps; otherwise fall back to Linear."""
    if use_spectral and int(in_features) == int(out_features):
        return RFFTCirculant1D(
            in_features=int(in_features),
            padded_dim=int(out_features),
            key=key,
        )
    return eqx.nn.Linear(int(in_features), int(out_features), key=key)


class PatchEmbedding(eqx.Module):
    linear: eqx.Module
    patch_size: int
    in_ch: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        *,
        key,
        use_spectral_proj: bool = True,
    ):
        self.in_ch = int(in_channels)
        self.out_dim = int(embed_dim)
        self.patch_size = int(patch_size)

        in_dim = self.in_ch * self.patch_size * self.patch_size
        self.linear = make_linear_or_spectral(
            in_dim,
            self.out_dim,
            key=key,
            use_spectral=use_spectral_proj,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ps = self.patch_size
        C, H, W = x.shape
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be multiples of patch_size={ps}; got {(H, W)}.")
        nH, nW = H // ps, W // ps
        patches = x.reshape(C, nH, ps, nW, ps).transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(nH * nW, C * ps * ps)
        return jax.vmap(self.linear)(patches)


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
        use_spectral_proj: bool = True,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko = jr.split(key, 4)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = make_linear_or_spectral(
            self.embed_dim,
            self.embed_dim,
            key=kq,
            use_spectral=use_spectral_proj,
        )
        self.k_proj = make_linear_or_spectral(
            self.embed_dim,
            self.embed_dim,
            key=kk,
            use_spectral=use_spectral_proj,
        )
        self.v_proj = make_linear_or_spectral(
            self.embed_dim,
            self.embed_dim,
            key=kv,
            use_spectral=use_spectral_proj,
        )
        self.out_proj = make_linear_or_spectral(
            self.embed_dim,
            self.embed_dim,
            key=ko,
            use_spectral=use_spectral_proj,
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
    mlp1: eqx.nn.Linear
    mlp2: eqx.nn.Linear
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
        use_spectral_proj: bool = True,
        layer_scale_init: float | None = 1e-5,
    ):
        k_attn, k_mlp1, k_mlp2 = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(
            embed_dim,
            num_heads,
            key=k_attn,
            use_spectral_proj=use_spectral_proj,
        )
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.mlp1 = eqx.nn.Linear(embed_dim, hidden_dim, key=k_mlp1)
        self.mlp2 = eqx.nn.Linear(hidden_dim, embed_dim, key=k_mlp2)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

        if layer_scale_init is not None and layer_scale_init > 0:
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
        x = x + (m * self.ls2 if self.ls2 is not None else m)
        return x


class RFFTDinoVisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    register_tokens: jnp.ndarray | None
    attention_blocks: Tuple[AttentionBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    n_register_tokens: int = eqx.field(static=True)
    pool: Pool = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    use_spectral_proj: bool = eqx.field(static=True)

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
        use_spectral_proj: bool = True,
    ):
        k_patch, k_misc = jr.split(key)
        self.embed_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.n_register_tokens = int(n_register_tokens)
        self.pool = pool
        self.channels = int(channels)
        self.use_spectral_proj = bool(use_spectral_proj)

        self.patch_embedding = PatchEmbedding(
            self.channels,
            self.embed_dim,
            self.patch_size,
            key=k_patch,
            use_spectral_proj=self.use_spectral_proj,
        )

        self.positional_embedding = jr.normal(
            k_misc, (1 + self.num_patches, self.embed_dim)
        )
        self.cls_token = jr.normal(jr.fold_in(k_misc, 1), (1, self.embed_dim))
        self.register_tokens = (
            jr.normal(jr.fold_in(k_misc, 2), (self.n_register_tokens, self.embed_dim))
            if self.n_register_tokens > 0
            else None
        )

        block_keys = jr.split(jr.fold_in(k_misc, 3), self.num_layers)
        self.attention_blocks = tuple(
            AttentionBlock(
                self.embed_dim,
                hidden_dim,
                num_heads,
                dropout_rate,
                key=kb,
                use_spectral_proj=self.use_spectral_proj,
                layer_scale_init=layer_scale_init,
            )
            for kb in block_keys
        )
        self.norm = eqx.nn.LayerNorm(self.embed_dim)
        self.head = eqx.nn.Linear(
            self.embed_dim, int(num_classes), key=jr.fold_in(k_misc, 4)
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def _tokens(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        ps = self.patch_size
        C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"Expected {self.channels} input channels; got {C}.")
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be multiples of patch_size={ps}; got {(H, W)}.")

        patches = self.patch_embedding(x)
        seq = jnp.concatenate([self.cls_token, patches], axis=0)
        seq = seq + self.positional_embedding[: seq.shape[0]]

        if self.register_tokens is not None and self.register_tokens.shape[0] > 0:
            seq = jnp.concatenate([seq[:1], self.register_tokens, seq[1:]], axis=0)

        return self.dropout(seq, key=key)

    def forward_tokens(self, x: jnp.ndarray, key, state):
        keys = jr.split(key, self.num_layers + 1)
        seq = self._tokens(x, key=keys[0])
        for block, kb in zip(self.attention_blocks, keys[1:]):
            seq = block(seq, key=kb)
        seq = jax.vmap(self.norm)(seq)
        return seq, state

    def forward_features(self, x: jnp.ndarray, key, state):
        seq, state = self.forward_tokens(x, key, state)
        if self.pool == "cls":
            feat = seq[0]
        else:
            start = 1 + (
                self.register_tokens.shape[0] if self.register_tokens is not None else 0
            )
            feat = jnp.mean(seq[start:], axis=0)
        return feat, state

    def forward_head(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.head(feat)

    def __call__(self, x: jnp.ndarray, key, state):
        feat, state = self.forward_features(x, key, state)
        logits = self.forward_head(feat)
        return logits, state
