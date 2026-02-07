# quantbayes/stochax/vae/pk/score_model.py
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp


class SinusoidalTimeEmb(eqx.Module):
    freqs: jnp.ndarray

    def __init__(self, dim: int):
        half = dim // 2
        f = jnp.exp(jnp.arange(half) * -(jnp.log(10000.0) / jnp.maximum(half - 1, 1)))
        self.freqs = f

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.asarray(t)
        e = t[..., None] * self.freqs[None, :]
        return jnp.concatenate([jnp.sin(e), jnp.cos(e)], axis=-1)  # (..., dim)


@dataclass(frozen=True)
class ScoreNetConfig:
    dim: int
    hidden: int = 256
    depth: int = 3
    time_emb_dim: int = 64


class LatentScoreNet(eqx.Module):
    """
    Noise-conditional score net:
      s_theta(log_sigma, x) ≈ ∇_x log p_sigma(x)
    """

    cfg: ScoreNetConfig = eqx.field(static=True)
    time_emb: SinusoidalTimeEmb
    mlp: eqx.nn.MLP

    def __init__(self, cfg: ScoreNetConfig, *, key):
        self.cfg = cfg
        k1, k2 = jax.random.split(key, 2)
        self.time_emb = SinusoidalTimeEmb(cfg.time_emb_dim)
        self.mlp = eqx.nn.MLP(
            in_size=cfg.dim + cfg.time_emb_dim,
            out_size=cfg.dim,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=k2,
        )

    def _single(self, log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        te = self.time_emb(log_sigma)  # (time_emb_dim,)
        inp = jnp.concatenate([x, te], axis=-1)
        return self.mlp(inp)

    def __call__(self, log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        log_sigma: scalar or (B,)
        x: (D,) or (B,D)
        returns: same shape as x
        """
        x = jnp.asarray(x)
        if x.ndim == 1:
            # single
            ls = jnp.asarray(log_sigma).reshape(())
            return self._single(ls, x)

        # batch
        b = x.shape[0]
        ls = jnp.asarray(log_sigma)
        if ls.ndim == 0:
            ls = jnp.full((b,), ls)
        elif ls.shape[0] != b:
            ls = jnp.broadcast_to(ls, (b,))
        return jax.vmap(self._single)(ls, x)
