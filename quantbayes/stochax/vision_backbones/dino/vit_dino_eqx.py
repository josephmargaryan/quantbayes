# quantbayes/stochax/vision_backbones/dino/vit_dino_eqx.py
from __future__ import annotations
from typing import Tuple, Optional, Literal

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


Pool = Literal["cls", "mean_patch"]


class PatchEmbedding(eqx.Module):
    """(C,H,W) -> (N_patches, D).  DINO uses a conv in PyTorch; here we keep the
    same linearized equivalent so weight export is easy."""

    linear: eqx.nn.Linear
    patch_size: int
    in_ch: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, *, key):
        self.in_ch = int(in_channels)
        self.out_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.linear = eqx.nn.Linear(
            patch_size * patch_size * in_channels, embed_dim, key=key
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [C,H,W], H/W % patch_size == 0 (assert upstream)
        ps = self.patch_size
        C, H, W = x.shape
        assert H % ps == 0 and W % ps == 0, "H,W must be multiples of patch size"
        # unfold into (N_patches, C*ps*ps) and apply linear per patch
        # reshape → vmap(linear)
        nH, nW = H // ps, W // ps
        patches = x.reshape(C, nH, ps, nW, ps).transpose(
            1, 3, 0, 2, 4
        )  # [nH,nW,C,ps,ps]
        patches = patches.reshape(nH * nW, C * ps * ps)  # [N, C*ps*ps]
        return jax.vmap(self.linear)(patches)  # [N, D]


class MultiheadSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, num_heads: int, *, key):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko = jr.split(key, 4)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = eqx.nn.Linear(self.embed_dim, self.embed_dim, key=kq)
        self.k_proj = eqx.nn.Linear(self.embed_dim, self.embed_dim, key=kk)
        self.v_proj = eqx.nn.Linear(self.embed_dim, self.embed_dim, key=kv)
        self.out_proj = eqx.nn.Linear(self.embed_dim, self.embed_dim, key=ko)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        D, H, hd = self.embed_dim, self.num_heads, self.head_dim
        q = jax.vmap(self.q_proj)(x)  # [N,D]
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)
        # split heads
        q = q.reshape(-1, H, hd).transpose(1, 0, 2)  # [H,N,hd]
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)  # [H,hd,N]
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)  # [H,N,hd]
        scale = 1.0 / math.sqrt(hd)
        attn = jax.nn.softmax(jnp.matmul(q * scale, k), axis=-1)  # [H,N,N]
        ctx = jnp.matmul(attn, v).transpose(1, 0, 2).reshape(-1, D)  # [N,D]
        return jax.vmap(self.out_proj)(ctx)


class AttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: MultiheadSelfAttention
    norm2: eqx.nn.LayerNorm
    mlp1: eqx.nn.Linear
    mlp2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    # optional LayerScale (DINOv2 often uses tiny gamma)
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
    ):
        k_attn, k_mlp = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, key=k_attn)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.mlp1 = eqx.nn.Linear(embed_dim, hidden_dim, key=k_mlp)
        self.mlp2 = eqx.nn.Linear(hidden_dim, embed_dim, key=jr.split(k_mlp, 2)[1])
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
        # pre-norm
        a = self.attn(jax.vmap(self.norm1)(x))
        a = self.dropout1(a, key=k1)
        if self.ls1 is not None:
            x = x + a * self.ls1
        else:
            x = x + a

        m = jax.vmap(self.mlp2)(
            jax.vmap(jax.nn.gelu)(jax.vmap(self.mlp1)(jax.vmap(self.norm2)(x)))
        )
        m = self.dropout2(m, key=k2)
        if self.ls2 is not None:
            x = x + m * self.ls2
        else:
            x = x + m
        return x


class DinoVisionTransformer(eqx.Module):
    """DINO/DINOv2-style ViT with optional register tokens.

    Fields are named to match your loaders & surgery utils:
    - patch_embedding.linear.{weight,bias}
    - positional_embedding (learnable) for [CLS + patches] only
    - cls_token, register_tokens (R x D or None)
    - attention_blocks[i].{layer_norm1,attention,layer_norm2,linear1,linear2}
    - norm, head
    """

    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray  # [1 + N_patches, D]
    cls_token: jnp.ndarray  # [1, D]
    register_tokens: jnp.ndarray | None  # [R, D] or None
    attention_blocks: Tuple[AttentionBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    # statics
    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    n_register_tokens: int = eqx.field(static=True)
    pool: Pool = eqx.field(static=True)

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
    ):
        k_patch, k_misc = jr.split(key)
        self.embed_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.n_register_tokens = int(n_register_tokens)
        self.pool = pool

        self.patch_embedding = PatchEmbedding(
            channels, embedding_dim, patch_size, key=k_patch
        )
        self.positional_embedding = jr.normal(k_misc, (1 + num_patches, embedding_dim))
        self.cls_token = jr.normal(jr.fold_in(k_misc, 1), (1, embedding_dim))
        self.register_tokens = (
            jr.normal(jr.fold_in(k_misc, 2), (n_register_tokens, embedding_dim))
            if n_register_tokens > 0
            else None
        )

        blocks = []
        keys = jr.split(jr.fold_in(k_misc, 3), num_layers)
        for kb in keys:
            blocks.append(
                AttentionBlock(
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    dropout_rate,
                    key=kb,
                    layer_scale_init=layer_scale_init,
                )
            )
        self.attention_blocks = tuple(blocks)
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.head = eqx.nn.Linear(embedding_dim, num_classes, key=jr.fold_in(k_misc, 4))
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def _tokens(self, x: jnp.ndarray, *, key) -> jnp.ndarray:
        """Build token sequence: [CLS, (R registers), patches] + add pos to [CLS+patches]."""
        ps = self.patch_size
        C, H, W = x.shape
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be multiples of patch size={ps}; got {(H,W)}.")
        patches = self.patch_embedding(x)  # [N,D]
        seq = jnp.concatenate([self.cls_token, patches], axis=0)  # [1+N, D]
        seq = seq + self.positional_embedding[: seq.shape[0]]
        if self.register_tokens is not None and self.register_tokens.shape[0] > 0:
            seq = jnp.concatenate([seq[:1], self.register_tokens, seq[1:]], axis=0)
        return self.dropout(seq, key=key)

    def __call__(self, x: jnp.ndarray, key, state):
        # x: [C,H,W] single sample
        ktoks, *krest = jr.split(key, self.num_layers + 1)
        x = self._tokens(x, key=ktoks)  # [1+R+N, D]
        for block, kb in zip(self.attention_blocks, krest):
            x = block(x, key=kb)
        x = jax.vmap(self.norm)(x)  # [T,D]
        # pooling
        if self.pool == "cls":
            feat = x[0]
        else:  # mean over patches (exclude CLS + registers)
            start = 1 + (
                self.register_tokens.shape[0] if self.register_tokens is not None else 0
            )
            feat = jnp.mean(x[start:], axis=0)
        logits = self.head(feat)
        return logits, state
