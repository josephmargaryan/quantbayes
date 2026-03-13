# quantbayes/stochax/vision_segmentation/models/dino_seg.py
from __future__ import annotations
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image as jimg

from quantbayes.stochax.vision_backbones.dino.vit_dino_eqx import (
    PatchEmbedding,
    AttentionBlock,
)


class DinoSeg(eqx.Module):
    # backbone parts (DINO-style)
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray  # [1+N, D]
    cls_token: jnp.ndarray  # [1, D]
    register_tokens: jnp.ndarray | None  # [R, D] or None
    attention_blocks: Tuple[AttentionBlock, ...]
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    # segmentation head: 1x1 conv on patch-grid features
    seg_head: eqx.nn.Conv2d

    # statics
    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    n_register_tokens: int = eqx.field(static=True)
    out_ch: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        out_ch: int = 1,
        n_register_tokens: int = 0,
        dropout_rate: float = 0.0,
        layer_scale_init: float | None = 1e-5,
        channels: int = 3,
        key,
    ):
        k1, k2 = jr.split(key, 2)
        self.embed_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.patch_size = int(patch_size)
        self.n_register_tokens = int(n_register_tokens)
        self.out_ch = int(out_ch)

        # patch embed
        self.patch_embedding = PatchEmbedding(
            channels, embedding_dim, patch_size, key=k1
        )

        # dummy positional tokens (resized at load-time if you warm-start)
        # we don't know num_patches ahead of time; allocate a generous buffer and slice per-call
        # but to keep things static, we'll store a [0, D] placeholder and build per-call positions
        # (positions are added before blocks; not train-stateful)
        self.positional_embedding = jnp.zeros(
            (1, embedding_dim)
        )  # placeholder row; unused in practice
        self.cls_token = jr.normal(k2, (1, embedding_dim))
        self.register_tokens = (
            jr.normal(jr.fold_in(k2, 1), (n_register_tokens, embedding_dim))
            if n_register_tokens > 0
            else None
        )

        # blocks
        blocks = []
        keys = jr.split(jr.fold_in(k2, 2), num_layers)
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
        self.dropout = eqx.nn.Dropout(dropout_rate)

        # 1x1 conv head on [D, H/ps, W/ps]
        self.seg_head = eqx.nn.Conv2d(embedding_dim, out_ch, 1, key=jr.fold_in(k2, 3))

    def _positional(self, n_patches: int, dim: int) -> jnp.ndarray:
        # simple learnable pos table per call: [1+N, D], initialized zeros
        # you can swap to a parameter if you prefer; for synthetic smoke it's irrelevant
        return jnp.zeros((1 + n_patches, dim), dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray, key, state):
        # x: [C,H,W] single sample
        C, H, W = map(int, x.shape)
        ps = self.patch_size
        if (H % ps) or (W % ps):
            raise ValueError(f"H,W must be multiples of patch size {ps}; got {(H,W)}")

        nH, nW = H // ps, W // ps
        Np = nH * nW

        # tokens
        k_tok, *k_blk = jr.split(key, self.num_layers + 1)
        patches = self.patch_embedding(x)  # [Np, D]
        seq = jnp.concatenate([self.cls_token, patches], axis=0)  # [1+Np, D]
        pos = self._positional(Np, self.embed_dim)  # [1+Np, D]
        seq = seq + pos
        if self.register_tokens is not None and self.register_tokens.shape[0] > 0:
            seq = jnp.concatenate([seq[:1], self.register_tokens, seq[1:]], axis=0)

        seq = self.dropout(seq, key=k_tok)

        # transformer blocks
        for block, kb in zip(self.attention_blocks, k_blk):
            seq = block(seq, key=kb)

        seq = jax.vmap(self.norm)(seq)

        # drop CLS and registers → patch grid
        start = 1 + (
            self.register_tokens.shape[0] if self.register_tokens is not None else 0
        )
        patch_tokens = seq[start:]  # [Np, D]
        fmap = patch_tokens.reshape(nH, nW, self.embed_dim).transpose(
            2, 0, 1
        )  # [D, nH, nW]

        # 1x1 conv → [out_ch, nH, nW]
        (k_head,) = jr.split(jr.fold_in(key, 7), 1)
        logits = self.seg_head(fmap, key=k_head)

        # upsample to input size
        logits = jimg.resize(logits, (self.out_ch, H, W), method="bilinear")
        return logits, state
