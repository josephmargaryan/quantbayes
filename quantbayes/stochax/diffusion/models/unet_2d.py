# score_diffusion/models/unet_2d.py

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

# You may reuse your SinusoidalPosEmb, ResnetBlock, and so on:
class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb

# A single UNet block or others from your code...

class SimpleUNet(eqx.Module):
    """
    A toy 2D UNet for demonstration. 
    In your production code, adapt your advanced version with downsampling, upsampling, etc.
    """
    time_embedding: eqx.nn.MLP
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.Conv2d

    def __init__(self, img_shape, hidden_size, *, key):
        # For simplicity, we'll define a minimal model
        c, h, w = img_shape
        key1, key2, key3 = jr.split(key, 3)
        self.time_embedding = eqx.nn.MLP(
            in_size=1, out_size=hidden_size, width_size=2*hidden_size, depth=2, key=key1
        )
        self.conv_in = eqx.nn.Conv2d(in_channels=c+1, out_channels=hidden_size, kernel_size=3, padding=1, key=key2)
        self.conv_out = eqx.nn.Conv2d(in_channels=hidden_size, out_channels=c, kernel_size=3, padding=1, key=key3)

    def __call__(self, t, x):
        """
        t: scalar time
        x: shape [c, h, w]
        """
        # Embed time
        t_emb = self.time_embedding(t[None])  # shape [hidden_size]
        # We broadcast it spatially:
        _, h, w = x.shape
        t_map = jnp.repeat(t_emb[..., None], h*w).reshape(-1, h, w)
        # Concatenate time map to input
        x_with_t = jnp.concatenate([x, t_map], axis=0)

        h = self.conv_in(x_with_t)
        # Just a single forward pass for demonstration
        out = self.conv_out(h)
        return out
