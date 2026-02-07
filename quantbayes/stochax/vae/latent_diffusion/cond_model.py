# quantbayes/stochax/vae/latent_diffusion/cond_model.py
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .model import SinusoidalTimeEmb


@dataclass(frozen=True)
class LatentEDMCondConfig:
    latent_dim: int = 16
    num_classes: int = 10  # MNIST default
    hidden: int = 256
    depth: int = 3
    time_emb_dim: int = 64
    label_emb_dim: int = 64  # separate label emb
    # null label index will be num_classes (embedding table size num_classes+1)


class LatentEDMCondMLP(eqx.Module):
    """
    Class-conditional EDM head for latents.
    Input: concat([z, time_emb(log_sigma), label_emb(label_or_null)]) -> MLP -> D (latent_dim).
    """

    cfg: LatentEDMCondConfig = eqx.static_field()
    time_emb: SinusoidalTimeEmb
    label_emb: eqx.nn.Embedding
    net: eqx.nn.MLP

    def __init__(self, cfg: LatentEDMCondConfig, *, key):
        self.cfg = cfg
        k_time, k_lab, k_mlp = jax.random.split(key, 3)

        self.time_emb = SinusoidalTimeEmb(cfg.time_emb_dim)

        # +1 for null token used by CFG training/inference
        self.label_emb = eqx.nn.Embedding(
            num_embeddings=cfg.num_classes + 1,
            embedding_size=cfg.label_emb_dim,
            key=k_lab,
        )

        in_size = cfg.latent_dim + cfg.time_emb_dim + cfg.label_emb_dim
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=cfg.latent_dim,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=jax.nn.gelu,
            final_activation=lambda x: x,
            key=k_mlp,
        )

    @property
    def null_label(self) -> int:
        return int(self.cfg.num_classes)

    def _single(
        self, log_sigma: jnp.ndarray, z: jnp.ndarray, label: jnp.ndarray
    ) -> jnp.ndarray:
        te = self.time_emb(log_sigma)
        le = self.label_emb(label)
        inp = jnp.concatenate([z, te, le], axis=-1)
        return self.net(inp)

    def __call__(
        self,
        log_sigma: jnp.ndarray,
        z: jnp.ndarray,
        *,
        key=None,
        train: bool = False,
        label=None,
    ) -> jnp.ndarray:
        """
        log_sigma: scalar or (B,)
        z: (D,) or (B,D)
        label:
          - None -> unconditional (null token)
          - int scalar -> replicated across batch
          - (B,) int array -> per-sample labels
        """
        z = jnp.asarray(z)

        # single
        if z.ndim == 1:
            ls = jnp.asarray(log_sigma).reshape(())
            if label is None:
                lab = jnp.asarray(self.null_label, dtype=jnp.int32)
            else:
                lab = jnp.asarray(label, dtype=jnp.int32).reshape(())
            lab = jnp.clip(lab, 0, self.null_label)
            return self._single(ls, z, lab)

        # batch
        b = z.shape[0]
        ls = jnp.asarray(log_sigma)
        if ls.ndim == 0:
            ls = jnp.full((b,), ls)
        else:
            ls = jnp.broadcast_to(ls, (b,))

        if label is None:
            lab = jnp.full((b,), self.null_label, dtype=jnp.int32)
        else:
            lab = jnp.asarray(label, dtype=jnp.int32)
            if lab.ndim == 0:
                lab = jnp.full((b,), int(lab), dtype=jnp.int32)
            else:
                lab = jnp.broadcast_to(lab, (b,))
        lab = jnp.clip(lab, 0, self.null_label)

        return jax.vmap(self._single)(ls, z, lab)
