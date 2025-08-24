# quantbayes/stochax/diffusion/models/tabular_dit.py
from __future__ import annotations
from typing import List, Optional
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


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


# ---------- per-feature embed/unembed (separate params per column) ----------
class FeatureEmbed(eqx.Module):
    weight: jnp.ndarray  # (D, E)
    bias: jnp.ndarray  # (D, E)

    def __init__(self, num_features: int, embed_dim: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.weight = jr.normal(k1, (num_features, embed_dim)) * (
            1.0 / jnp.sqrt(embed_dim)
        )
        self.bias = jr.normal(k2, (num_features, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (D,) -> (D,E)
        return x[:, None] * self.weight + self.bias


class FeatureUnembed(eqx.Module):
    weight: jnp.ndarray  # (D, E)
    bias: jnp.ndarray  # (D,)

    def __init__(self, num_features: int, embed_dim: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.weight = jr.normal(k1, (num_features, embed_dim)) * (
            1.0 / jnp.sqrt(embed_dim)
        )
        self.bias = jr.normal(k2, (num_features,)) * 0.02

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: (D,E) -> (D,)
        return jnp.sum(tokens * self.weight, axis=-1) + self.bias


class LearnableFeatureIDEmb(eqx.Module):
    id_emb: jnp.ndarray  # (D,E)

    def __init__(self, num_features: int, embed_dim: int, *, key):
        self.id_emb = jr.normal(key, (num_features, embed_dim)) * 0.02

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: (D,E)
        D, E = tokens.shape
        return tokens + self.id_emb[:D]


class TabDiTBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    ada_mod: eqx.nn.MLP
    ada_head: ZeroLinear

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
        self.ada_head = ZeroLinear(2 * embed_dim, 6 * embed_dim)

    def __call__(
        self, tokens: jnp.ndarray, cond: jnp.ndarray, *, key=None
    ) -> jnp.ndarray:
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


class TabDiT(eqx.Module):
    feat_embed: FeatureEmbed
    feat_id: LearnableFeatureIDEmb
    blocks: List[TabDiTBlock]
    norm: eqx.nn.LayerNorm
    ada_mod: eqx.nn.MLP
    ada_head: ZeroLinear
    feat_unembed: FeatureUnembed
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP

    num_features: int
    embed_dim: int

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        time_emb_dim: int = 128,
        dropout_rate: float = 0.0,
        *,
        key,
    ):
        k_embed, k_id, k_unembed, k_time, k_final, *k_blocks = jr.split(key, depth + 5)
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.feat_embed = FeatureEmbed(num_features, embed_dim, key=k_embed)
        self.feat_id = LearnableFeatureIDEmb(num_features, embed_dim, key=k_id)
        self.blocks = [
            TabDiTBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=kb)
            for kb in k_blocks
        ]
        self.norm = eqx.nn.LayerNorm((embed_dim,))
        self.ada_mod = eqx.nn.MLP(embed_dim, embed_dim, embed_dim, depth=1, key=k_final)
        self.ada_head = ZeroLinear(embed_dim, 2 * embed_dim)
        self.feat_unembed = FeatureUnembed(num_features, embed_dim, key=k_unembed)

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            time_emb_dim, embed_dim, 2 * embed_dim, depth=2, key=k_time
        )

    def _forward(
        self, t: float, x: jnp.ndarray, *, key=None, train: bool = False
    ) -> jnp.ndarray:
        # x: (D,)
        tokens = self.feat_embed(x)  # (D,E)
        tokens = self.feat_id(tokens)  # add feature-id emb

        cond = self.time_proj(self.time_emb(jnp.array(t)))
        k = key
        for blk in self.blocks:
            if k is not None:
                k, sub = jr.split(k)
            else:
                sub = None
            tokens = blk(tokens, cond, key=sub)

        # final adaLN + norm
        sh, sc = jnp.split(self.ada_head(self.ada_mod(cond)), 2, axis=-1)
        tokens = jax.vmap(self.norm)(tokens)
        tokens = modulate(tokens, sh, sc)

        out = self.feat_unembed(tokens)  # (D,)
        return out

    def __call__(
        self, t: float, x: jnp.ndarray, *, key=None, train: bool = False
    ) -> jnp.ndarray:
        """
        x: (D,) or (B,D). Returns same shape.
        """
        if x.ndim == 2:
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
        elif x.ndim == 1:
            return self._forward(t, x, key=key, train=train)
        else:
            raise ValueError(f"Expected (D,) or (B,D), got {x.shape}")
