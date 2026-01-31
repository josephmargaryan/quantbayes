# quantbayes/stochax/diffusion/models/timeseries_dit.py
from __future__ import annotations
from typing import List, Optional
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange


# ---------- sinusoidal time emb ----------
class SinusoidalTimeEmb(eqx.Module):
    emb: jnp.ndarray

    def __init__(self, dim: int):
        half = dim // 2
        scale = math.log(10000.0) / (half - 1)
        self.emb = jnp.exp(jnp.arange(half) * -scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        e = x * self.emb
        return jnp.concatenate([jnp.sin(e), jnp.cos(e)], -1)


# ---------- small helper with zero init (DiT trick) ----------
class ZeroLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size: int, out_size: int):
        self.weight = jnp.zeros((out_size, in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weight.T + self.bias


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


# ---------- 1D patch embed/unembed along sequence ----------
class PatchEmbed1D(eqx.Module):
    patch_size: int
    proj: eqx.nn.Linear
    in_channels: int
    seq_len: int
    num_patches: int
    embed_dim: int

    def __init__(
        self, seq_len: int, in_channels: int, patch_size: int, embed_dim: int, *, key
    ):
        assert (seq_len % patch_size) == 0, "seq_len must be divisible by patch_size"
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.proj = eqx.nn.Linear(patch_size * in_channels, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (L, C)
        L, C = x.shape
        ps = self.patch_size
        tokens = rearrange(x, "(n ps) c -> n (ps c)", ps=ps)
        tokens = jax.vmap(self.proj)(tokens)  # (n_patches, embed_dim)
        return tokens


class PatchUnembed1D(eqx.Module):
    patch_size: int
    out_channels: int
    seq_len: int
    num_patches: int
    linear: eqx.nn.Linear

    def __init__(
        self, seq_len: int, out_channels: int, patch_size: int, embed_dim: int, *, key
    ):
        assert (seq_len % patch_size) == 0
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        # map back to (patch_size*out_channels)
        self.linear = eqx.nn.Linear(embed_dim, patch_size * out_channels, key=key)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: (n_patches, embed_dim) -> (L, C_out)
        patches = jax.vmap(self.linear)(tokens)  # (n_patches, ps*C)
        x = rearrange(
            patches, "n (ps c) -> (n ps) c", ps=self.patch_size, c=self.out_channels
        )
        return x


# ---------- DiT block with adaLN ----------
class TS_DiTBlock(eqx.Module):
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
        k1, k2, k3, k4 = jr.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm((embed_dim,))
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout_rate,
            inference=False,
            key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm((embed_dim,))
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = eqx.nn.MLP(embed_dim, embed_dim, hidden, depth=2, key=k2)
        self.ada_mod = eqx.nn.MLP(
            embed_dim, 2 * embed_dim, 2 * embed_dim, depth=1, key=k3
        )
        self.ada_head = ZeroLinear(
            2 * embed_dim, 6 * embed_dim
        )  # (shift/scale/gates) x 2
        self.dropout_rate = dropout_rate

    def __call__(
        self, tokens: jnp.ndarray, cond: jnp.ndarray, *, key=None
    ) -> jnp.ndarray:
        # tokens: (N, E), cond: (E,)
        m = self.ada_head(self.ada_mod(cond))
        sh_a, sc_a, gt_a, sh_m, sc_m, gt_m = jnp.split(m, 6, axis=-1)

        x = jax.vmap(self.norm1)(tokens)
        x = modulate(x, sh_a, sc_a)
        attn = self.attn(x, x, x, key=key)
        tokens = tokens + gt_a * attn

        x = jax.vmap(self.norm2)(tokens)
        x = modulate(x, sh_m, sc_m)
        mlp_out = jax.vmap(self.mlp)(x)
        tokens = tokens + gt_m * mlp_out
        return tokens


# ---------- Final layer ----------
class TS_DiTFinal(eqx.Module):
    norm: eqx.nn.LayerNorm
    ada_mod: eqx.nn.MLP
    ada_head: ZeroLinear
    unembed: PatchUnembed1D

    def __init__(self, embed_dim: int, unembed: PatchUnembed1D, *, key):
        k1, k2 = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm((embed_dim,))
        self.ada_mod = eqx.nn.MLP(embed_dim, embed_dim, embed_dim, depth=1, key=k1)
        self.ada_head = ZeroLinear(embed_dim, 2 * embed_dim)  # shift/scale
        self.unembed = unembed

    def __call__(self, tokens: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        mod = self.ada_head(self.ada_mod(cond))
        sh, sc = jnp.split(mod, 2, axis=-1)
        x = jax.vmap(self.norm)(tokens)
        x = modulate(x, sh, sc)
        return self.unembed(x)


# ---------- Time-series DiT ----------
class TimeDiT1D(eqx.Module):
    # modules
    patch: PatchEmbed1D
    pos_emb: jnp.ndarray
    final: TS_DiTFinal
    blocks: List[TS_DiTBlock]
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP
    # meta
    seq_len: int
    in_channels: int
    out_channels: int
    patch_size: int
    embed_dim: int

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        time_emb_dim: int = 256,
        dropout_rate: float = 0.0,
        learn_sigma: bool = False,
        *,
        key,
    ):
        # keys
        k_patch, k_unpatch, k_pos, k_time, k_final, *k_blocks = jr.split(key, depth + 5)

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch = PatchEmbed1D(
            seq_len, in_channels, patch_size, embed_dim, key=k_patch
        )
        self.pos_emb = jr.normal(k_pos, (self.patch.num_patches, embed_dim)) * 0.02

        unembed = PatchUnembed1D(
            seq_len, self.out_channels, patch_size, embed_dim, key=k_unpatch
        )
        self.final = TS_DiTFinal(embed_dim, unembed, key=k_final)
        self.blocks = [
            TS_DiTBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=kb)
            for kb in k_blocks
        ]

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            time_emb_dim, embed_dim, 2 * embed_dim, depth=2, key=k_time
        )

    # single sample: x (L, C)
    def _forward(
        self, t: float, x: jnp.ndarray, *, key=None, train: bool = False
    ) -> jnp.ndarray:
        tokens = self.patch(x)  # (N_patches, E)
        tokens = tokens + self.pos_emb[: tokens.shape[0]]

        t_emb = self.time_proj(self.time_emb(jnp.array(t)))
        cond = t_emb

        k = key
        for blk in self.blocks:
            if k is not None:
                k, sub = jr.split(k)
            else:
                sub = None
            tokens = blk(tokens, cond, key=sub)

        y = self.final(tokens, cond)  # (L, C_out)
        return y

    def __call__(
        self, t: float, x: jnp.ndarray, *, key=None, train: bool = False
    ) -> jnp.ndarray:
        """
        x: (L, C) or (B, L, C). Returns same shape.
        """
        if x.ndim == 3:
            B = x.shape[0]
            if key is not None:
                keys = jr.split(key, B)
                return jax.vmap(
                    lambda xi, ki: self._forward(t, xi, key=ki, train=train)
                )(x, keys)
            else:
                return jax.vmap(lambda xi: self._forward(t, xi, key=None, train=train))(
                    x
                )
        elif x.ndim == 2:
            return self._forward(t, x, key=key, train=train)
        else:
            raise ValueError(f"Expected (L,C) or (B,L,C), got {x.shape}")
