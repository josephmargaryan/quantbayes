# diffusion/models/mixer_2d.py

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


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


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
    ):
        tkey, ckey = jr.split(key, 2)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        # y shape is [hidden_size, num_patches]
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm

    time_emb: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,  # kept for API compat; unused here
        *,
        key,
    ):
        in_ch, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)

        k_in, k_out, k_time, *bkeys = jr.split(key, 3 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            in_ch, hidden_size, patch_size, stride=patch_size, key=k_in
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, in_ch, patch_size, stride=patch_size, key=k_out
        )

        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bk
            )
            for bk in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))

        # EDM-style time conditioning (use log σ when training with EDM).
        self.time_emb = SinusoidalTimeEmb(hidden_size)
        self.time_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=2 * hidden_size,  # shift, scale
            width_size=2 * hidden_size,
            depth=1,
            key=k_time,
        )

    def _forward(self, t, y, *, key=None):
        y = self.conv_in(y)  # (hidden, H', W')
        c, h_p, w_p = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")  # (hidden, P)

        # FiLM with time embedding
        t_emb = self.time_emb(t)
        shift, scale = jnp.split(self.time_proj(t_emb), 2, axis=-1)  # (hidden,)
        y = modulate(y, shift[:, None], scale[:, None])

        for block in self.blocks:
            y = block(y)
        y = self.norm(y)

        y = einops.rearrange(y, "c (h w) -> c h w", h=h_p, w=w_p)
        return self.conv_out(y)

    def __call__(self, t, y, *, key=None):
        if y.ndim == 4:
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, k: self._forward(t, sample, key=k))(
                    y, keys
                )
            return jax.vmap(lambda sample: self._forward(t, sample, key=None))(y)
        return self._forward(t, y, key=key)
