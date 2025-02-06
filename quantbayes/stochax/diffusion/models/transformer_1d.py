import math
from typing import List
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Helper: key splitting function.
def key_split_allowing_none(key):
    if key is None:
        return key, None
    else:
        return jr.split(key)


########################################
# SinusoidalTimeEmb for diffusion time
########################################
class SinusoidalTimeEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        emb = x * self.emb
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

########################################
# Basic 1D token embedding
########################################
class TokenEmbed(eqx.Module):
    """Embed a 1D sequence (length,) into (length, embed_dim)."""
    proj: eqx.nn.Linear
    seq_length: int
    embed_dim: int

    def __init__(self, seq_length: int, embed_dim: int, *, key):
        # We'll embed each scalar in the sequence into embed_dim, i.e. a linear from R^1 -> R^embed_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.proj = eqx.nn.Linear(1, embed_dim, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (seq_length,) => we embed each time step, returning shape (seq_length, embed_dim).
        """
        if x.shape[0] != self.seq_length:
            # for simplicity assume x is exactly seq_length
            pass
        # expand each scalar to shape (1,) then vmap
        def embed_scalar(scalar):
            return self.proj(scalar[None])  # shape (embed_dim,)
        return jax.vmap(embed_scalar)(x)  # shape (seq_length, embed_dim)

class TokenUnembed(eqx.Module):
    """Reverse of TokenEmbed: convert (seq_length, embed_dim) back to shape (seq_length,)."""
    unproj: eqx.nn.Linear
    seq_length: int
    embed_dim: int

    def __init__(self, seq_length: int, embed_dim: int, *, key):
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.unproj = eqx.nn.Linear(embed_dim, 1, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (seq_length, embed_dim) => returns shape (seq_length,)
        """
        def unproj_token(token):
            scalar = self.unproj(token)
            return scalar[0]  # shape ()
        scalars = jax.vmap(unproj_token)(x)  # shape (seq_length,)
        return scalars

########################################
# Learned positional embedding
########################################
class LearnablePositionalEmb(eqx.Module):
    pos_emb: jnp.ndarray

    def __init__(self, seq_length: int, embed_dim: int, *, key):
        self.pos_emb = jr.normal(key, (seq_length, embed_dim)) * 0.02

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (seq_length, embed_dim).
        We'll just add pos_emb (seq_length, embed_dim).
        """
        seq_len, emb_dim = x.shape
        # if seq_len < self.pos_emb.shape[0], slice
        # or if seq_len > self.pos_emb.shape[0], tile, etc.
        # for simplicity, assume seq_len == self.pos_emb.shape[0].
        pe = self.pos_emb[:seq_len]
        return x + pe

########################################
# A TransformerBlock for 1D tokens
########################################
class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ff: eqx.nn.MLP
    embed_dim: int
    dropout_rate: float

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float, dropout_rate: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

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

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        """
        x: shape (seq_length, embed_dim).
        """
        seq_len, emb_dim = x.shape

        # part 1: LN, attn, residual
        x_norm = jax.vmap(self.norm1)(x)
        # eqx MultiheadAttention can handle (seq, dim)
        attn_out = self.attn(x_norm, x_norm, x_norm, key=key)
        x = x + attn_out

        # part 2: LN, MLP, residual
        x_norm = jax.vmap(self.norm2)(x)
        ff_out = jax.vmap(self.ff)(x_norm)
        x = x + ff_out

        return x

########################################
# Diffusion Transformer for 1D Time-Series
########################################
class DiffusionTransformer1D(eqx.Module):
    # time embedding
    time_emb_fn: SinusoidalTimeEmb
    time_proj: eqx.nn.MLP

    # token embed/unembed
    token_embed: TokenEmbed
    pos_embed: LearnablePositionalEmb
    blocks: List[TransformerBlock]
    norm: eqx.nn.LayerNorm
    token_unembed: TokenUnembed

    seq_length: int
    embed_dim: int

    def __init__(
        self,
        seq_length: int,
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
        seq_length: length of time-series
        embed_dim: dimension of each token
        depth: number of transformer blocks
        n_heads: number of attention heads
        mlp_ratio: ratio for feed-forward dimension
        dropout_rate: dropout prob
        time_emb_dim: dimension for time embedding
        """
        k1, k2, k3, k4, *bk = jr.split(key, depth + 4)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        # 1) time embedding
        self.time_emb_fn = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=embed_dim,
            width_size=2 * embed_dim,
            depth=2,
            key=k1,
        )

        # 2) token embedding + positional embedding
        self.token_embed = TokenEmbed(seq_length, embed_dim, key=k2)
        self.token_unembed = TokenUnembed(seq_length, embed_dim, key=k3)
        self.pos_embed = LearnablePositionalEmb(seq_length, embed_dim, key=k4)

        # 3) transformer blocks
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout_rate, key=bk[i])
            )
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def _forward(self, t, x, *, key=None):
        # 1) time embedding
        t_emb = self.time_emb_fn(t)       # shape (time_emb_dim,)
        t_emb = self.time_proj(t_emb)     # shape (embed_dim,) not (1, embed_dim)

        # 2) token embed
        tokens = self.token_embed(x)
        # 3) pos embed
        tokens = self.pos_embed(tokens)
        # 4) broadcast t_emb
        t_map = jnp.broadcast_to(t_emb, tokens.shape)  # shape => (seq_length, embed_dim)
        tokens = tokens + t_map

        # 5) blocks
        k = key
        for block in self.blocks:
            k, subk = key_split_allowing_none(k)
            tokens = block(tokens, key=subk)
        tokens = jax.vmap(self.norm)(tokens)

        # 6) unembed
        out = self.token_unembed(tokens)
        return out

    def __call__(self, t, y, *, key=None):
        """
        y can be (seq_length,) or (batch, seq_length).
        We output the same shape.
        """
        if y.ndim == 2:
            # batch
            if key is not None:
                keys = jr.split(key, y.shape[0])
                return jax.vmap(lambda sample, kk: self._forward(t, sample, key=kk))(y, keys)
            else:
                return jax.vmap(lambda sample: self._forward(t, sample, key=None))(y)
        else:
            # single sample
            return self._forward(t, y, key=key)
