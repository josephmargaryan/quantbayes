from __future__ import annotations

from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class SinusoidalTimeEmbedding(eqx.Module):
    """Classic sinusoidal time embedding."""

    freqs: jax.Array

    def __init__(self, dim: int):
        dim = int(dim)
        if dim < 2:
            raise ValueError("time embedding dim must be >= 2")
        half = dim // 2
        # log-spaced frequencies
        self.freqs = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(1000.0), half))

    def __call__(self, t: jax.Array) -> jax.Array:
        t = jnp.asarray(t)
        args = t * self.freqs
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        # if dim is odd, pad one zero
        if emb.shape[-1] * 2 != self.freqs.shape[-1] * 2:
            pass
        return emb


@dataclass
class ScoreMLP(eqx.Module):
    """
    Simple score network for low-dimensional diffusion:
      score(t, x) in R^d

    Signature is compatible with your diffusion loss code:
      model(t, y, key=..., train=...)
    """

    dim: int
    time_emb: SinusoidalTimeEmbedding
    mlp: eqx.nn.MLP

    def __init__(
        self,
        dim: int,
        *,
        time_dim: int = 32,
        width_size: int = 128,
        depth: int = 3,
        key: jax.Array,
    ):
        self.dim = int(dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        in_size = self.dim + (time_dim // 2) * 2  # embedding returns 2*half
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=self.dim,
            width_size=int(width_size),
            depth=int(depth),
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(
        self,
        t: jax.Array,
        y: jax.Array,
        *,
        key: jax.Array | None = None,
        train: bool | None = None,
    ) -> jax.Array:
        # key/train are ignored (stateless net), but kept for API compatibility.
        y = jnp.asarray(y)
        t_emb = self.time_emb(t)
        inp = jnp.concatenate([y.reshape(-1), t_emb.reshape(-1)], axis=-1)
        return self.mlp(inp)
