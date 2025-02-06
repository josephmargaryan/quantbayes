# file: transformer_2d.py

import math
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

############################################################
# SinusoidalTimeEmb + MLP for the diffusion time embedding
############################################################
class SinusoidalTimeEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: scalar time
        emb = x * self.emb
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)

############################################################
# Patch Embedding + Un-Patching
############################################################
class PatchEmbed(eqx.Module):
    """Split image (C,H,W) into non-overlapping patches and project to tokens."""
    patch_size: int
    proj: eqx.nn.Linear
    num_patches: int
    embed_dim: int
    channels: int

    def __init__(self, channels: int, embed_dim: int, patch_size: int, img_size: tuple, *, key):
        """img_size = (C, H, W). We'll embed patches into shape (num_patches, embed_dim)."""
        # We'll flatten each patch into a vector of size patch_size*patch_size*C.
        # Then project into embed_dim with a linear layer.
        self.channels = channels
        self.patch_size = patch_size
        c, h, w = img_size
        assert c == channels
        assert (h % patch_size) == 0 and (w % patch_size) == 0
        num_h = h // patch_size
        num_w = w // patch_size
        self.num_patches = num_h * num_w
        self.embed_dim = embed_dim

        in_features = patch_size * patch_size * channels
        self.proj = eqx.nn.Linear(in_features, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (C,H,W)
        returns tokens: shape (num_patches, embed_dim)
        """
        c, h, w = x.shape
        ph = pw = self.patch_size
        # rearrange into patches
        # shape => (num_patches, patch_size*patch_size*C)
        patches = rearrange(
            x,
            "c (nh ph) (nw pw) -> (nh nw) (c ph pw)",
            ph=ph,
            pw=pw,
        )
        # project
        tokens = jax.vmap(self.proj)(patches)  # shape (num_patches, embed_dim)
        return tokens


class PatchUnembed(eqx.Module):
    """Reverse of PatchEmbed: convert tokens to (C,H,W)."""
    patch_size: int
    proj: eqx.nn.Linear
    num_patches: int
    embed_dim: int
    channels: int
    h: int
    w: int

    def __init__(self, channels: int, embed_dim: int, patch_size: int, img_size: tuple, *, key):
        self.channels = channels
        self.patch_size = patch_size
        c, h, w = img_size
        assert (h % patch_size) == 0 and (w % patch_size) == 0
        num_h = h // patch_size
        num_w = w // patch_size
        self.num_patches = num_h * num_w
        self.embed_dim = embed_dim
        self.h = h
        self.w = w

        out_features = patch_size * patch_size * channels
        self.proj = eqx.nn.Linear(embed_dim, out_features, key=key)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        tokens: shape (num_patches, embed_dim)
        returns x: shape (C,H,W)
        """
        # reverse the linear
        patches = jax.vmap(self.proj)(tokens)  # shape (num_patches, patch_size*patch_size*C)
        # rearrange patches back to (C,H,W)
        ph = pw = self.patch_size
        c = self.channels
        num_h = self.h // ph
        num_w = self.w // pw
        x = rearrange(
            patches,
            "(nh nw) (c ph pw) -> c (nh ph) (nw pw)",
            nh=num_h,
            nw=num_w,
            ph=ph,
            pw=pw,
        )
        return x

############################################################
# Learnable Positional Embedding
############################################################
class LearnablePositionalEmb(eqx.Module):
    pos_emb: jnp.ndarray  # shape (num_patches, embed_dim)

    def __init__(self, num_patches: int, embed_dim: int, *, key):
        self.pos_emb = jr.normal(key, (num_patches, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (num_patches, embed_dim)
        We add pos_emb (broadcast if needed).
        """
        n, d = x.shape
        # if n < self.pos_emb.shape[0], slice
        # or if n > self.pos_emb.shape[0], we might need to tile
        # for simplicity, we assume n == self.pos_emb.shape[0].
        pe = self.pos_emb[:n]
        return x + pe

############################################################
# A single Transformer block
############################################################
class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ff: eqx.nn.MLP
    dropout_rate: float

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float, dropout_rate: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
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
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ff = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=hidden_dim,
            depth=2,
            key=k2,
        )
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        x: shape (num_patches, embed_dim).
        We'll do LN -> self-attn -> residual -> LN -> MLP -> residual.
        """
        # self-attn
        # eqx MultiheadAttention supports shape (seq, dim) or (batch, seq, dim).
        # We'll treat (seq, dim).
        # so pass x as shape (seq, dim)
        seq_len, emb_dim = x.shape

        # part 1
        x_norm = jax.vmap(self.norm1)(x)
        # Because eqx's MultiheadAttention can handle shape (seq, dim):
        attn_out = self.attn(x_norm, x_norm, x_norm, key=key)
        x = x + attn_out

        # part 2
        x_norm = jax.vmap(self.norm2)(x)
        # MLP
        ff_out = jax.vmap(self.ff)(x_norm)  # shape (seq, emb_dim)
        x = x + ff_out

        return x

############################################################
# Diffusion Transformer for Images
############################################################
class DiffusionTransformer2D(eqx.Module):
    # time embedding
    time_emb_fn: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP

    # patch embedding
    patch_embed: PatchEmbed
    pos_embed: LearnablePositionalEmb
    blocks: List[TransformerBlock]
    norm: eqx.nn.LayerNorm
    patch_unembed: PatchUnembed

    embed_dim: int
    num_patches: int

    def __init__(
        self,
        img_size: tuple,
        patch_size: int,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        time_emb_dim: int,
        *,
        key,
    ):
        """
        img_size: (C,H,W)
        patch_size: int
        embed_dim: token embedding dim
        depth: number of transformer blocks
        n_heads: number of attention heads
        mlp_ratio: ratio for feed-forward dimension, e.g. 4.0
        dropout_rate: dropout probability
        time_emb_dim: dimension for time embedding
        """
        c, h, w = img_size
        k1, k2, k3, k4, *bk = jr.split(key, depth + 4)

        # 1) time embedding
        self.time_emb_fn = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=embed_dim,
            width_size=2 * embed_dim,
            depth=2,
            key=k1,
        )

        # 2) patch embedding/unembedding
        self.patch_embed = PatchEmbed(c, embed_dim, patch_size, img_size, key=k2)
        self.patch_unembed = PatchUnembed(c, embed_dim, patch_size, img_size, key=k3)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        # 3) positional embedding
        self.pos_embed = LearnablePositionalEmb(self.num_patches, embed_dim, key=k4)

        # 4) transformer blocks
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=bk[i])
            )
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def _forward(self, t, x, *, key=None):
        # 1) time embedding
        t_emb = self.time_emb_fn(t)        # shape (time_emb_dim,)
        t_emb = self.time_proj(t_emb)      # shape (embed_dim,) not (1, embed_dim)

        # 2) patchify
        tokens = self.patch_embed(x)
        # 3) pos embed
        tokens = self.pos_embed(tokens)
        # 4) broadcast t_emb
        t_map = jnp.broadcast_to(t_emb, tokens.shape)  # shape (num_patches, embed_dim)
        tokens = tokens + t_map

        # 5) blocks
        k = key
        for block in self.blocks:
            k, subk = key_split_allowing_none(k)
            tokens = block(tokens, key=subk)

        tokens = jax.vmap(self.norm)(tokens)
        # 6) unpatchify
        out = self.patch_unembed(tokens)
        return out

    def __call__(self, t, y, *, key=None):
        """
        if y.ndim==4 => batched input: (B,C,H,W)
        else => single sample: (C,H,W)
        returns same shape.
        """
        if y.ndim == 4:
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, kk: self._forward(t, sample, key=kk))(y, keys)
            else:
                return jax.vmap(lambda sample: self._forward(t, sample, key=None))(y)
        else:
            return self._forward(t, y, key=key)
