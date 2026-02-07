# quantbayes/stochax/vae/latent_diffusion/model.py
from __future__ import annotations

from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp


class SinusoidalTimeEmb(eqx.Module):
    freqs: jnp.ndarray
    dim: int = eqx.static_field()

    def __init__(self, dim: int):
        self.dim = int(dim)
        half = max((self.dim + 1) // 2, 1)
        self.freqs = jnp.exp(jnp.arange(half) * -(jnp.log(10000.0) / max(half - 1, 1)))

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.asarray(t)
        e = t[..., None] * self.freqs[None, :]
        emb = jnp.concatenate([jnp.sin(e), jnp.cos(e)], axis=-1)
        return emb[..., : self.dim]


@dataclass(frozen=True)
class LatentEDMConfig:
    latent_dim: int = 16
    hidden: int = 256
    depth: int = 3
    time_emb_dim: int = 64


class LatentEDMMLP(eqx.Module):
    cfg: LatentEDMConfig = eqx.static_field()
    time_emb: SinusoidalTimeEmb
    net: eqx.nn.MLP

    def __init__(self, cfg: LatentEDMConfig, *, key):
        self.cfg = cfg
        k1, k2 = jax.random.split(key, 2)
        self.time_emb = SinusoidalTimeEmb(cfg.time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size=cfg.latent_dim + cfg.time_emb_dim,
            out_size=cfg.latent_dim,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=k2,
        )

    def _single(self, log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        te = self.time_emb(log_sigma)
        inp = jnp.concatenate([x, te], axis=-1)
        return self.net(inp)

    def __call__(
        self, log_sigma: jnp.ndarray, x: jnp.ndarray, *, key=None, train: bool = False
    ):
        x = jnp.asarray(x)
        if x.ndim == 1:
            ls = jnp.asarray(log_sigma).reshape(())
            return self._single(ls, x)

        b = x.shape[0]
        ls = jnp.asarray(log_sigma)
        if ls.ndim == 0:
            ls = jnp.full((b,), ls)
        else:
            ls = jnp.broadcast_to(ls, (b,))
        return jax.vmap(self._single)(ls, x)


class EDMNet(eqx.Module):
    """Thin wrapper to match EDM loss signature: model(log_sigma, x, key=..., train=...)."""

    net: eqx.Module

    def __call__(self, log_sigma, x, *, key=None, train: bool = False, **kwargs):
        return self.net(log_sigma, x, key=key, train=train, **kwargs)
