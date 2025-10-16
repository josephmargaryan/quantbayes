# quantbayes/stochax/diffusion/models/spectral_dit.py
from __future__ import annotations

import math
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from quantbayes.stochax.layers.spectral_layers import SpectralDense


# ---------------- time / label / position ----------------
class SinusoidalTimeEmb(eqx.Module):
    emb: jnp.ndarray

    def __init__(self, dim: int):
        half_dim = dim // 2
        scale = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        e = x * self.emb
        return jnp.concatenate([jnp.sin(e), jnp.cos(e)], axis=-1)


class LearnablePositionalEmb(eqx.Module):
    pos_emb: jnp.ndarray

    def __init__(self, num_patches: int, embed_dim: int, *, key):
        self.pos_emb = jr.normal(key, (num_patches, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.pos_emb[: x.shape[0]]


class LabelEmbedder(eqx.Module):
    emb: eqx.nn.Embedding
    dropout_prob: float

    def __init__(self, num_classes: int, embed_dim: int, dropout_prob: float, *, key):
        self.emb = eqx.nn.Embedding(num_classes + 1, embed_dim, key=key)
        self.dropout_prob = dropout_prob

    def __call__(self, labels: jnp.ndarray, train: bool, *, key=None) -> jnp.ndarray:
        if train and self.dropout_prob > 0 and key is not None:
            drop = jr.uniform(key, labels.shape) < self.dropout_prob
            labels = jnp.where(drop, self.emb.num_embeddings - 1, labels)
        return self.emb(labels)


# ---------------- patch embed/unembed (spectral) ----------------
class PatchEmbedSpectral(eqx.Module):
    patch_size: int
    proj: SpectralDense
    num_patches: int
    embed_dim: int
    channels: int

    def __init__(
        self,
        channels: int,
        embed_dim: int,
        patch_size: int,
        img_size: tuple,
        *,
        key,
        svd_rank: int | None = None,
    ):
        self.channels = channels
        self.patch_size = patch_size
        c, h, w = img_size
        assert c == channels
        assert h % patch_size == 0 and w % patch_size == 0
        nh, nw = h // patch_size, w // patch_size
        self.num_patches = nh * nw
        self.embed_dim = embed_dim
        in_features = patch_size * patch_size * channels
        self.proj = SpectralDense(in_features, embed_dim, rank=svd_rank, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (C,H,W)
        ph = pw = self.patch_size
        patches = rearrange(x, "c (nh ph) (nw pw) -> (nh nw) (c ph pw)", ph=ph, pw=pw)
        return jax.vmap(self.proj)(patches)


class PatchUnembed(eqx.Module):
    patch_size: int
    num_patches: int
    channels: int
    h: int
    w: int

    def __init__(self, channels: int, patch_size: int, img_size: tuple):
        self.channels = channels
        self.patch_size = patch_size
        _, h, w = img_size
        assert h % patch_size == 0 and w % patch_size == 0
        self.h, self.w = h, w
        self.num_patches = (h // patch_size) * (w // patch_size)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        ph = pw = self.patch_size
        nh, nw = self.h // ph, self.w // pw
        return rearrange(
            tokens, "(nh nw) (c ph pw) -> c (nh ph) (nw pw)", nh=nh, nw=nw, ph=ph, pw=pw
        )


# ---------------- adaLN DiT block ----------------
def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale) + shift


class ZeroLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size, out_size):
        self.weight = jnp.zeros((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x):
        return x @ self.weight.T + self.bias


class DiTBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    ada_mod: eqx.nn.MLP
    ada_head: ZeroLinear
    dropout_rate: float

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        *,
        key,
    ):
        k_attn, k_mlp, k_ada = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm((embed_dim,))
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout_rate,
            inference=False,
            key=k_attn,
        )
        self.norm2 = eqx.nn.LayerNorm((embed_dim,))
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = eqx.nn.MLP(embed_dim, embed_dim, hidden, depth=2, key=k_mlp)
        self.ada_mod = eqx.nn.MLP(
            embed_dim, embed_dim * 2, width_size=embed_dim * 2, depth=1, key=k_ada
        )
        self.ada_head = ZeroLinear(embed_dim * 2, 6 * embed_dim)
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, key=None) -> jnp.ndarray:
        # adaLN params
        mod = self.ada_head(self.ada_mod(cond))
        s_attn, g_attn, ga_attn, s_mlp, g_mlp, ga_mlp = jnp.split(mod, 6, axis=-1)

        # split rng for attention
        k_attn, _ = (None, None) if key is None else jr.split(key, 2)

        # attn
        xn = jax.vmap(self.norm1)(x)
        xm = modulate(xn, s_attn, g_attn)
        aout = self.attn(xm, xm, xm, key=k_attn)
        x = x + ga_attn * aout

        # mlp
        xn2 = jax.vmap(self.norm2)(x)
        xm2 = modulate(xn2, s_mlp, g_mlp)
        mout = jax.vmap(self.mlp)(xm2)
        x = x + ga_mlp * mout
        return x


class FinalLayer(eqx.Module):
    norm: eqx.nn.LayerNorm
    proj: SpectralDense
    ada_mod: eqx.nn.MLP
    ada_head: ZeroLinear
    patch_size: int
    out_channels: int

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        out_channels: int,
        *,
        key,
        svd_rank: int | None = None,
    ):
        k_proj, k_ada = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm((embed_dim,))
        self.proj = SpectralDense(
            embed_dim, patch_size * patch_size * out_channels, rank=svd_rank, key=k_proj
        )
        self.ada_mod = eqx.nn.MLP(
            embed_dim, embed_dim, width_size=embed_dim, depth=1, key=k_ada
        )
        self.ada_head = ZeroLinear(embed_dim, 2 * embed_dim)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        mod = self.ada_head(self.ada_mod(cond))
        shift, scale = jnp.split(mod, 2, axis=-1)
        xn = jax.vmap(self.norm)(x)
        xm = modulate(xn, shift, scale)
        return jax.vmap(self.proj)(xm)


class SpectralDiT(eqx.Module):
    patch_embed: PatchEmbedSpectral
    pos_embed: LearnablePositionalEmb
    patch_unembed: PatchUnembed
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP
    label_embed: LabelEmbedder
    blocks: List[DiTBlock]
    final_layer: FinalLayer
    embed_dim: int
    num_patches: int
    patch_size: int
    out_channels: int

    def __init__(
        self,
        img_size: tuple[int, int, int],  # (C,H,W)
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        time_emb_dim: int,
        num_classes: int,
        learn_sigma: bool,
        *,
        key,
        svd_rank: int | None = None,
    ):
        c, h, w = img_size
        keys = jr.split(key, depth + 6)
        k_time, k_patch, k_pos, k_unembed, k_label, k_final, *k_blocks = keys

        self.patch_embed = PatchEmbedSpectral(
            c, embed_dim, patch_size, img_size, key=k_patch, svd_rank=svd_rank
        )
        effective_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_unembed = PatchUnembed(effective_channels, patch_size, img_size)
        self.pos_embed = LearnablePositionalEmb(
            self.patch_embed.num_patches, embed_dim, key=k_pos
        )

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            time_emb_dim, embed_dim, width_size=2 * embed_dim, depth=2, key=k_time
        )
        self.label_embed = LabelEmbedder(
            num_classes, embed_dim, dropout_rate, key=k_label
        )

        self.blocks = [
            DiTBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=k_blocks[i])
            for i in range(depth)
        ]
        self.final_layer = FinalLayer(
            embed_dim, patch_size, effective_channels, key=k_final, svd_rank=svd_rank
        )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_channels = effective_channels
        self.num_patches = self.patch_embed.num_patches

    # --- internal single sample ---
    def _forward(
        self, t: float, x: jnp.ndarray, label: jnp.ndarray, train: bool, *, key=None
    ) -> jnp.ndarray:
        tok = self.patch_embed(x)  # (P, D)
        tok = self.pos_embed(tok)

        t_emb = self.time_emb(jnp.array(t))
        t_emb = self.time_proj(t_emb)  # (D,)

        lk, key = (key, None) if key is None else jr.split(key)
        l_emb = self.label_embed(label, train, key=lk)
        cond = t_emb + l_emb

        # add cond to tokens (broadcast)
        tok = tok + jnp.broadcast_to(cond, tok.shape)

        for blk in self.blocks:
            if key is not None:
                key, sub = jr.split(key)
            else:
                sub = None
            tok = blk(tok, cond, key=sub)

        out_tokens = self.final_layer(tok, cond)  # (P, patch*c)
        return out_tokens

    def __call__(
        self, t: float, x: jnp.ndarray, label: jnp.ndarray, train: bool, *, key=None
    ) -> jnp.ndarray:
        if x.ndim == 4:

            def one(sample, lab, k):
                tokens = self._forward(t, sample, lab, train, key=k)
                return self.patch_unembed(tokens)

            if key is not None:
                keys = jr.split(key, x.shape[0])
                return jax.vmap(one)(x, label, keys)
            else:
                return jax.vmap(
                    lambda s, l: self.patch_unembed(
                        self._forward(t, s, l, train, key=None)
                    )
                )(x, label)
        else:
            tokens = self._forward(t, x, label, train, key=key)
            return self.patch_unembed(tokens)
