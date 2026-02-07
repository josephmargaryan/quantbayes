from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Any, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from quantbayes.stochax.layers.spectral_layers import RFFTCirculant1D


# ---------------------------
# Helpers
# ---------------------------
class SinusoidalTimeEmb(eqx.Module):
    freqs: jnp.ndarray

    def __init__(self, dim: int):
        half = max(dim // 2, 1)
        self.freqs = jnp.exp(jnp.arange(half) * -(jnp.log(10000.0) / max(half - 1, 1)))

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.asarray(t).reshape(())
        e = t * self.freqs
        emb = jnp.concatenate([jnp.sin(e), jnp.cos(e)], axis=-1)
        return emb[: (2 * self.freqs.shape[0])]  # safe


class ZeroLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size: int, out_size: int):
        self.weight = jnp.zeros((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x):
        return x @ self.weight.T + self.bias


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale) + shift


def make_linear_or_spectral(
    dim: int,
    *,
    use_spectral: bool,
    key: jr.PRNGKey,
) -> eqx.Module:
    # square only: dim->dim
    if use_spectral:
        return RFFTCirculant1D(in_features=dim, padded_dim=dim, key=key)
    return eqx.nn.Linear(dim, dim, key=key)


# ---------------------------
# Patch embedding/unembedding
# ---------------------------
class PatchEmbed(eqx.Module):
    patch_size: int
    proj: eqx.nn.Linear
    num_patches: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    h: int = eqx.field(static=True)
    w: int = eqx.field(static=True)

    def __init__(
        self, img_size: tuple[int, int, int], patch_size: int, embed_dim: int, *, key
    ):
        c, h, w = img_size
        assert h % patch_size == 0 and w % patch_size == 0
        self.channels = c
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.embed_dim = embed_dim
        in_features = patch_size * patch_size * c
        self.proj = eqx.nn.Linear(in_features, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ph = pw = self.patch_size
        patches = rearrange(x, "c (nh ph) (nw pw) -> (nh nw) (c ph pw)", ph=ph, pw=pw)
        return jax.vmap(self.proj)(patches)


class PatchUnembed(eqx.Module):
    patch_size: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    h: int = eqx.field(static=True)
    w: int = eqx.field(static=True)

    def __init__(
        self, img_size: tuple[int, int, int], patch_size: int, out_channels: int
    ):
        _, h, w = img_size
        self.h = h
        self.w = w
        self.channels = out_channels
        self.patch_size = patch_size

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        ph = pw = self.patch_size
        nh = self.h // ph
        nw = self.w // pw
        return rearrange(
            tokens, "(nh nw) (c ph pw) -> c (nh ph) (nw pw)", nh=nh, nw=nw, ph=ph, pw=pw
        )


class LearnablePositionalEmb(eqx.Module):
    pos_emb: jnp.ndarray  # (P,D)

    def __init__(self, num_patches: int, embed_dim: int, *, key):
        self.pos_emb = jr.normal(key, (num_patches, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.pos_emb[: x.shape[0]]


# ---------------------------
# CFG label embedding
# ---------------------------
class LabelEmbedder(eqx.Module):
    emb: eqx.nn.Embedding
    dropout_prob: float

    def __init__(self, num_classes: int, embed_dim: int, dropout_prob: float, *, key):
        self.emb = eqx.nn.Embedding(num_classes + 1, embed_dim, key=key)  # +1 null
        self.dropout_prob = float(dropout_prob)

    @property
    def null_label(self) -> int:
        return int(self.emb.num_embeddings - 1)

    def __call__(
        self, labels: jnp.ndarray, train: bool, *, key: Optional[jr.PRNGKey] = None
    ) -> jnp.ndarray:
        labels = jnp.asarray(labels, dtype=jnp.int32)
        if labels.ndim == 0:
            labels = labels[None]

        if train and self.dropout_prob > 0 and key is not None:
            drop = jr.uniform(key, labels.shape) < self.dropout_prob
            labels = jnp.where(drop, self.null_label, labels)

        labels = jnp.clip(labels, 0, self.null_label)
        return self.emb(labels)  # (B,D) or (1,D)


# ---------------------------
# Spectralizable MHSA
# ---------------------------
class SpectralMHSA(eqx.Module):
    q_proj: eqx.Module
    k_proj: eqx.Module
    v_proj: eqx.Module
    out_proj: eqx.Module
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    attn_dropout: eqx.nn.Dropout

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        use_spectral: bool,
        key,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko, kd = jr.split(key, 5)
        self.q_proj = make_linear_or_spectral(
            embed_dim, use_spectral=use_spectral, key=kq
        )
        self.k_proj = make_linear_or_spectral(
            embed_dim, use_spectral=use_spectral, key=kk
        )
        self.v_proj = make_linear_or_spectral(
            embed_dim, use_spectral=use_spectral, key=kv
        )
        self.out_proj = make_linear_or_spectral(
            embed_dim, use_spectral=use_spectral, key=ko
        )
        self.attn_dropout = eqx.nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

    def __call__(
        self, x: jnp.ndarray, *, train: bool, key: Optional[jr.PRNGKey]
    ) -> jnp.ndarray:
        # x: (P,D)
        H = self.num_heads
        hd = self.head_dim
        D = self.embed_dim

        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        q = q.reshape(-1, H, hd).transpose(1, 0, 2)  # (H,P,hd)
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)  # (H,hd,P)
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)  # (H,P,hd)

        scale = 1.0 / math.sqrt(hd)
        scores = jnp.matmul(q * scale, k)  # (H,P,P)
        attn = jax.nn.softmax(scores, axis=-1)

        if key is not None:
            attn = self.attn_dropout(attn, key=key, inference=(not train))

        ctx = jnp.matmul(attn, v)  # (H,P,hd)
        ctx = ctx.transpose(1, 0, 2).reshape(-1, D)  # (P,D)

        return jax.vmap(self.out_proj)(ctx)


# ---------------------------
# DiT block (AdaLN)
# ---------------------------
class DiTBlockRFFT(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: SpectralMHSA
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    adaLN_modulation: eqx.nn.MLP
    adaLN_head: ZeroLinear

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        *,
        use_spectral: bool,
        key,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm((embed_dim,))
        self.attn = SpectralMHSA(
            embed_dim, n_heads, dropout_rate, use_spectral=use_spectral, key=k1
        )
        self.norm2 = eqx.nn.LayerNorm((embed_dim,))
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=hidden_dim,
            depth=2,
            key=k2,
        )
        self.adaLN_modulation = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim * 2,
            width_size=embed_dim * 2,
            depth=1,
            key=k3,
        )
        self.adaLN_head = ZeroLinear(embed_dim * 2, 6 * embed_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        train: bool,
        key: Optional[jr.PRNGKey],
    ) -> jnp.ndarray:
        mod_params = self.adaLN_head(self.adaLN_modulation(cond))
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            mod_params, 6, axis=-1
        )

        k_attn, k_rest = (None, None) if key is None else jr.split(key, 2)

        x_norm = jax.vmap(self.norm1)(x)
        x_mod = modulate(x_norm, shift_attn, scale_attn)
        attn_out = self.attn(x_mod, train=train, key=k_attn)
        x = x + gate_attn * attn_out

        x_norm2 = jax.vmap(self.norm2)(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = jax.vmap(self.mlp)(x_mod2)
        x = x + gate_mlp * mlp_out
        return x


class DiTFinalLayer(eqx.Module):
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: eqx.nn.MLP
    adaLN_head: ZeroLinear
    patch_size: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm((embed_dim,))
        self.linear = eqx.nn.Linear(
            embed_dim, patch_size * patch_size * out_channels, key=k1
        )
        self.adaLN_modulation = eqx.nn.MLP(
            in_size=embed_dim, out_size=embed_dim, width_size=embed_dim, depth=1, key=k2
        )
        self.adaLN_head = ZeroLinear(embed_dim, 2 * embed_dim)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        shift, scale = jnp.split(
            self.adaLN_head(self.adaLN_modulation(cond)), 2, axis=-1
        )
        x_norm = jax.vmap(self.norm)(x)
        x_mod = modulate(x_norm, shift, scale)
        return jax.vmap(self.linear)(x_mod)


# ---------------------------
# Main DiT (EDM-compatible interface)
# ---------------------------
class RFFTAdaptiveDiT(eqx.Module):
    patch_embed: PatchEmbed
    pos_embed: LearnablePositionalEmb
    patch_unembed: PatchUnembed
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP
    label_embed: LabelEmbedder
    blocks: List[DiTBlockRFFT]
    final_layer: DiTFinalLayer

    img_size: tuple[int, int, int] = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    learn_sigma: bool = eqx.field(static=True)
    use_spectral_proj: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        img_size: tuple[int, int, int],
        patch_size: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        time_emb_dim: int,
        num_classes: int,
        learn_sigma: bool = False,
        use_spectral_proj: bool = True,
        key,
    ):
        c, h, w = img_size
        keys = jr.split(key, 6 + depth)
        k_time, k_patch, k_pos, k_label, k_final, k_blocks_base, *k_blocks = keys

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = int(num_classes)
        self.learn_sigma = bool(learn_sigma)
        self.use_spectral_proj = bool(use_spectral_proj)

        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, key=k_patch)

        eff_c = c * 2 if self.learn_sigma else c
        self.patch_unembed = PatchUnembed(img_size, patch_size, out_channels=eff_c)

        self.pos_embed = LearnablePositionalEmb(
            self.patch_embed.num_patches, embed_dim, key=k_pos
        )

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=embed_dim,
            width_size=2 * embed_dim,
            depth=2,
            key=k_time,
        )

        self.label_embed = LabelEmbedder(
            self.num_classes, embed_dim, dropout_rate, key=k_label
        )

        self.blocks = [
            DiTBlockRFFT(
                embed_dim,
                n_heads,
                mlp_ratio,
                dropout_rate,
                use_spectral=use_spectral_proj,
                key=k_blocks[i],
            )
            for i in range(depth)
        ]
        self.final_layer = DiTFinalLayer(embed_dim, patch_size, eff_c, key=k_final)

    def _forward(
        self,
        log_sigma: jnp.ndarray,
        x: jnp.ndarray,
        *,
        label: Optional[jnp.ndarray],
        train: bool,
        key: Optional[jr.PRNGKey],
    ):
        # x: (C,H,W) single
        tokens = self.patch_embed(x)
        tokens = self.pos_embed(tokens)

        # time cond (use log_sigma as in EDM)
        te = self.time_emb(jnp.asarray(log_sigma).reshape(()))
        cond_t = self.time_proj(te)  # (D,)

        # label cond (CFG style)
        if label is None:
            lab = jnp.asarray(self.label_embed.null_label, dtype=jnp.int32)
        else:
            lab = jnp.asarray(label, dtype=jnp.int32).reshape(())
        lab_key, k_rest = (None, None) if key is None else jr.split(key, 2)
        cond_y = self.label_embed(lab, train=train, key=lab_key)[0]  # (D,)

        cond = cond_t + cond_y

        # add cond to tokens
        tokens = tokens + cond[None, :]

        # transformer blocks
        k_loop = k_rest
        for blk in self.blocks:
            if k_loop is not None:
                k_loop, sub = jr.split(k_loop, 2)
            else:
                sub = None
            tokens = blk(tokens, cond, train=train, key=sub)

        out_tokens = self.final_layer(tokens, cond)
        return self.patch_unembed(out_tokens)

    def __call__(
        self,
        log_sigma: jnp.ndarray,
        x: jnp.ndarray,
        *,
        key=None,
        train: bool = False,
        label=None,
        **kwargs,
    ):
        # EDM expects this signature. Supports batch or single.
        x = jnp.asarray(x)
        if x.ndim == 4:
            b = x.shape[0]
            if label is None:
                labels = jnp.full((b,), self.label_embed.null_label, dtype=jnp.int32)
            else:
                lab = jnp.asarray(label, dtype=jnp.int32)
                labels = (
                    lab if lab.ndim == 1 else jnp.full((b,), int(lab), dtype=jnp.int32)
                )

            keys = None if key is None else jr.split(key, b)

            def one(xi, li, ki):
                return self._forward(log_sigma, xi, label=li, train=train, key=ki)

            if keys is None:
                return jax.vmap(lambda xi, li: one(xi, li, None))(x, labels)
            return jax.vmap(one)(x, labels, keys)

        return self._forward(log_sigma, x, label=label, train=train, key=key)
