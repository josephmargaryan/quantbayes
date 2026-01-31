# quantbayes/stochax/diffusion/models/spectral_mixer_2d.py
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import einops

from quantbayes.stochax.layers.spectral_layers import SpectralConv2d, SpectralTokenMixer


class SinusoidalTimeEmb(eqx.Module):
    emb: jnp.ndarray

    def __init__(self, dim: int):
        half = dim // 2
        f = jnp.exp(jnp.arange(half) * -(jnp.log(10000.0) / (half - 1)))
        self.emb = f

    def __call__(self, x):
        x = jnp.asarray(x)
        e = x * self.emb
        return jnp.concatenate([jnp.sin(e), jnp.cos(e)], -1)


def modulate(y, shift, scale):
    return y * (1.0 + scale) + shift


class SpectralMixer2d(eqx.Module):
    conv_in: SpectralConv2d
    conv_out: eqx.nn.ConvTranspose2d
    token_mixers: list[SpectralTokenMixer]
    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP
    norm_tokens: eqx.nn.LayerNorm

    hidden_size: int
    num_patches: int
    patch_h: int
    patch_w: int

    def __init__(
        self,
        img_size: tuple[int, int, int],  # (C,H,W)
        patch_size: int,
        hidden_size: int,
        num_blocks: int,
        *,
        token_groups: int = 1,  # SpectralTokenMixer groups
        key,
    ):
        C, H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        Ht, Wt = H // patch_size, W // patch_size
        N = Ht * Wt

        k_in, k_out, k_time, *bkeys = jr.split(key, 3 + num_blocks)

        # patchify with a strided spectral conv
        self.conv_in = SpectralConv2d(
            C,
            hidden_size,
            patch_size,
            patch_size,
            strides=(patch_size, patch_size),
            padding="VALID",
            key=k_in,
        )
        # de-patchify with standard transpose conv (no spectral transpose conv available)
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, C, patch_size, stride=patch_size, key=k_out
        )

        self.token_mixers = [
            SpectralTokenMixer(
                n_tokens=N, channels=hidden_size, groups=token_groups, key=bk
            )
            for bk in bkeys
        ]
        self.norm_tokens = eqx.nn.LayerNorm((N, hidden_size))

        self.time_emb = SinusoidalTimeEmb(hidden_size)
        self.time_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=2 * hidden_size,
            width_size=2 * hidden_size,
            depth=1,
            key=k_time,
        )

        self.hidden_size = hidden_size
        self.num_patches = N
        self.patch_h, self.patch_w = Ht, Wt

    def _forward(self, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # y: (C,H,W) -> (hidden, Ht, Wt)
        y = self.conv_in(y)
        c, ht, wt = y.shape
        y = einops.rearrange(y, "c h w -> (h w) c")  # (N, hidden)

        # time FiLM
        shift, scale = jnp.split(
            self.time_proj(self.time_emb(t)), 2, axis=-1
        )  # (hidden,)
        y = modulate(y, shift[None, :], scale[None, :])

        # token mixing (residual inside SpectralTokenMixer by default)
        for mix in self.token_mixers:
            y = self.norm_tokens(y)
            y = mix(y)

        # back to (hidden, Ht, Wt) then upsample to (C,H,W)
        y = einops.rearrange(y, "(h w) c -> c h w", h=ht, w=wt)
        return self.conv_out(y)

    def __call__(
        self, t: jnp.ndarray, y: jnp.ndarray, *, key=None, train: bool | None = None
    ) -> jnp.ndarray:
        if y.ndim == 4:
            return jax.vmap(lambda yy: self._forward(t, yy))(y)
        else:
            return self._forward(t, y)
