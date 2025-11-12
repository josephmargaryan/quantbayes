# quantbayes/stochax/diffusion/models/spectral_unet_2d.py
from __future__ import annotations

import math
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from quantbayes.stochax.layers.spectral_layers import SpectralConv2d


class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jax.Array) -> jax.Array:
        e = x * self.emb
        return jnp.concatenate((jnp.sin(e), jnp.cos(e)), axis=-1)


class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(self, dim: int, key, heads: int = 4, dim_head: int = 32):
        k1, k2 = jr.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden = heads * dim_head
        self.to_qkv = eqx.nn.Conv2d(dim, 3 * hidden, 1, key=k1)
        self.to_out = eqx.nn.Conv2d(hidden, dim, 1, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)
        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


def upsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


def downsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H // factor, factor, W // factor, factor])
    return jnp.mean(y, axis=[2, 4])


def _key_split_maybe(key):
    return (None, None) if key is None else jr.split(key)


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlockSpectral(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int

    # spectral convs (3x3) and 1x1 residual
    conv1: SpectralConv2d
    conv2: SpectralConv2d
    res_conv: SpectralConv2d

    # time mlp
    mlp_layers: list[Union[Callable, eqx.nn.Linear]]

    # optional scaling convs when not biggan-style
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]

    # groupnorm + dropout + (optional) attn
    block1_groupnorm: eqx.nn.GroupNorm
    block2_layers: list[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    attn: Optional[Residual]

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        is_biggan: bool,
        up: bool,
        down: bool,
        time_emb_dim: int,
        dropout_rate: float,
        is_attn: bool,
        heads: int,
        dim_head: int,
        *,
        key,
    ):
        k1, k2, k3, k4, k_attn = jr.split(key, 5)

        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        # time embedding projection
        self.mlp_layers = [jax.nn.silu, eqx.nn.Linear(time_emb_dim, dim_out, key=k1)]

        # spectral 3x3 convs
        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.conv1 = SpectralConv2d(dim_in, dim_out, 3, 3, padding="SAME", key=k2)

        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
        ]
        self.conv2 = SpectralConv2d(dim_out, dim_out, 3, 3, padding="SAME", key=k3)

        # up/down sampling path (same as your UNet)
        assert not self.up or not self.down
        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = eqx.nn.ConvTranspose2d(
                    dim_in, dim_in, kernel_size=4, stride=2, padding=1, key=k4
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in, dim_in, kernel_size=3, stride=2, padding=1, key=k4
                )
            else:
                self.scaling = None

        # 1x1 spectral residual conv for channel match
        self.res_conv = SpectralConv2d(dim_in, dim_out, 1, 1, padding="SAME", key=k4)

        self.attn = (
            Residual(
                LinearTimeSelfAttention(
                    dim_out, heads=heads, dim_head=dim_head, key=k_attn
                )
            )
            if is_attn
            else None
        )

    def __call__(self, x: jax.Array, t: jax.Array, *, key) -> jax.Array:
        C, _, _ = x.shape

        h = jax.nn.silu(self.block1_groupnorm(x))
        if self.up or self.down:
            h = self.scaling(h)  # type: ignore
            x = self.scaling(x)  # type: ignore

        h = self.conv1(h)

        # time FiLM
        for layer in self.mlp_layers:
            t = layer(t)
        h = h + t[..., None, None]

        # second conv + dropout block
        k1, _ = _key_split_maybe(key)
        h = self.block2_layers[0](h)
        h = self.block2_layers[1](h)
        h = self.block2_layers[2](h, key=k1, inference=(k1 is None))
        h = self.conv2(h)

        # residual
        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)
        out = (h + x) / jnp.sqrt(2)

        if self.attn is not None:
            out = self.attn(out)
        return out


class SpectralUNet2d(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP
    first_conv: SpectralConv2d
    down_res_blocks: list[list[ResnetBlockSpectral]]
    mid_block1: ResnetBlockSpectral
    mid_block2: ResnetBlockSpectral
    ups_res_blocks: list[list[ResnetBlockSpectral]]
    final_conv_layers: list[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]

    def __init__(
        self,
        data_shape: tuple[int, int, int],
        is_biggan: bool,
        dim_mults: list[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: list[int],
        *,
        key,
    ):
        k0, k_first, k_mid1, k_mid2, k_final = jr.split(key, 5)
        C, H, W = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.mlp = eqx.nn.MLP(
            hidden_size, hidden_size, 4 * hidden_size, 1, activation=jax.nn.silu, key=k0
        )
        self.first_conv = SpectralConv2d(
            C, hidden_size, 3, 3, padding="SAME", key=k_first
        )

        # Down path
        h, w = H, W
        self.down_res_blocks = []
        num_down_blocks = len(in_out) * num_res_blocks - 1
        keys_down = jr.split(k0, num_down_blocks)
        i = 0
        for idx, (din, dout) in enumerate(in_out):
            is_attn = (h in attn_resolutions) and (w in attn_resolutions)
            blocks = [
                ResnetBlockSpectral(
                    din,
                    dout,
                    is_biggan,
                    False,
                    False,
                    hidden_size,
                    dropout_rate,
                    is_attn,
                    heads,
                    dim_head,
                    key=keys_down[i],
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                blocks.append(
                    ResnetBlockSpectral(
                        dout,
                        dout,
                        is_biggan,
                        False,
                        False,
                        hidden_size,
                        dropout_rate,
                        is_attn,
                        heads,
                        dim_head,
                        key=keys_down[i],
                    )
                )
                i += 1
            if idx < len(in_out) - 1:
                blocks.append(
                    ResnetBlockSpectral(
                        dout,
                        dout,
                        is_biggan,
                        False,
                        True,
                        hidden_size,
                        dropout_rate,
                        is_attn,
                        heads,
                        dim_head,
                        key=keys_down[i],
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(blocks)

        # Middle
        mid = dims[-1]
        self.mid_block1 = ResnetBlockSpectral(
            mid,
            mid,
            is_biggan,
            False,
            False,
            hidden_size,
            dropout_rate,
            True,
            heads,
            dim_head,
            key=k_mid1,
        )
        self.mid_block2 = ResnetBlockSpectral(
            mid,
            mid,
            is_biggan,
            False,
            False,
            hidden_size,
            dropout_rate,
            False,
            heads,
            dim_head,
            key=k_mid2,
        )

        # Up path
        self.ups_res_blocks = []
        num_up_blocks = len(in_out) * (num_res_blocks + 1) - 1
        keys_up = jr.split(k_first, num_up_blocks)
        i = 0
        for idx, (din, dout) in enumerate(reversed(in_out)):
            is_attn = (h in attn_resolutions) and (w in attn_resolutions)
            blocks = []
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ResnetBlockSpectral(
                        dout * 2,
                        dout,
                        is_biggan,
                        False,
                        False,
                        hidden_size,
                        dropout_rate,
                        is_attn,
                        heads,
                        dim_head,
                        key=keys_up[i],
                    )
                )
                i += 1
            blocks.append(
                ResnetBlockSpectral(
                    dout + din,
                    din,
                    is_biggan,
                    False,
                    False,
                    hidden_size,
                    dropout_rate,
                    is_attn,
                    heads,
                    dim_head,
                    key=keys_up[i],
                )
            )
            i += 1
            if idx < len(in_out) - 1:
                blocks.append(
                    ResnetBlockSpectral(
                        din,
                        din,
                        is_biggan,
                        True,
                        False,
                        hidden_size,
                        dropout_rate,
                        is_attn,
                        heads,
                        dim_head,
                        key=keys_up[i],
                    )
                )
                i += 1
                h, w = h * 2, w * 2
            self.ups_res_blocks.append(blocks)

        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, C, 1, key=k_final),
        ]

    def _forward(self, t: jax.Array, y: jax.Array, *, key) -> jax.Array:
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        h = self.first_conv(y)
        hs = [h]

        for blocks in self.down_res_blocks:
            for blk in blocks:
                key, sub = _key_split_maybe(key)
                h = blk(h, t, key=sub)
                hs.append(h)

        key, sub = _key_split_maybe(key)
        h = self.mid_block1(h, t, key=sub)
        key, sub = _key_split_maybe(key)
        h = self.mid_block2(h, t, key=sub)

        for blocks in self.ups_res_blocks:
            for blk in blocks:
                key, sub = _key_split_maybe(key)
                if blk.up:
                    h = blk(h, t, key=sub)
                else:
                    h = blk(jnp.concatenate((h, hs.pop()), axis=0), t, key=sub)

        for layer in self.final_conv_layers:
            h = layer(h)
        return h

    def __call__(
        self, t: jax.Array, y: jax.Array, *, key=None, train: bool | None = None
    ) -> jax.Array:
        if y.ndim == 4:
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda yy, kk: self._forward(t, yy, key=kk))(y, keys)
            else:
                return jax.vmap(lambda yy: self._forward(t, yy, key=None))(y)
        else:
            return self._forward(t, y, key=key)
