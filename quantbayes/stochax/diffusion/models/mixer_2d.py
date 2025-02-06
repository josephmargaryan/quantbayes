# score_diffusion/models/mixer_2d.py

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

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
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        *,
        key,
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            input_size + 1, hidden_size, patch_size, stride=patch_size, key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size, key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bk
            )
            for bk in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def _forward(self, t, y, *, key=None):
        # Now y is assumed to have shape (C, H, W)
        t_scaled = jnp.array(t / self.t1)
        _, height, width = y.shape  # now works because y is (C, H, W)
        t_map = jnp.broadcast_to(t_scaled, (height, width))
        t_map = t_map[None, ...]  # shape (1, H, W)
        # Concatenate time channel
        y = jnp.concatenate([y, t_map], axis=0)
        y = self.conv_in(y)
        c, patch_h, patch_w = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(y, "c (h w) -> c h w", h=patch_h, w=patch_w)
        return self.conv_out(y)

    def __call__(self, t, y, *, key=None):
        # Accept an optional key so that our overall training framework works.
        if y.ndim == 4:  # batched input
            if key is not None:
                # Split key for each sample in the batch.
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, sample_key: self._forward(t, sample, key=sample_key))(
                    y, keys
                )
            else:
                return jax.vmap(lambda sample: self._forward(t, sample, key=None))(y)
        else:
            return self._forward(t, y, key=key)
