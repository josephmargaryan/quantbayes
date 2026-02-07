from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

"""
from quantbayes.stochax.diffusion.models.edm_factory import build_edm_denoiser
model = build_edm_denoiser("rfft_dit", img_size=(1,28,28), key=jr.PRNGKey(0))
"""


DenoiserName = Literal[
    "mixer2d",
    "unet2d",
    "adaptive_dit",
    "rfft_dit",
]


def _norm_name(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").strip()


class EDMAdapter(eqx.Module):
    """Wrap a model with signature net(t, x, key=None, train=..., **kw) into EDM signature."""

    net: eqx.Module

    def __call__(self, log_sigma, x, *, key=None, train: bool = False, **kwargs):
        # Many of your nets ignore train; so try train-forward, then fallback.
        try:
            return self.net(log_sigma, x, key=key, train=train, **kwargs)
        except TypeError:
            return self.net(log_sigma, x, key=key, **kwargs)


class AdaptiveDiTEDMAdapter(eqx.Module):
    """Adapter for diffusion/models/adaptive_DiT.py DiT(t,x,label,train,key)."""

    dit: eqx.Module
    default_label: int = eqx.field(static=True)
    learn_sigma: bool = eqx.field(static=True)

    def __call__(
        self, log_sigma, x, *, key=None, train: bool = False, label=None, **kwargs
    ):
        # x may be (B,C,H,W) or (C,H,W)
        x = jnp.asarray(x)

        if self.learn_sigma:
            raise ValueError(
                "adaptive_DiT was constructed with learn_sigma=True (output channels 2C). "
                "Your EDM loss expects output channels == input channels. Set learn_sigma=False."
            )

        # make labels
        if x.ndim == 4:
            b = x.shape[0]
            if label is None:
                lab = jnp.full((b,), self.default_label, dtype=jnp.int32)
            else:
                lab = jnp.asarray(label, dtype=jnp.int32)
                if lab.ndim == 0:
                    lab = jnp.full((b,), int(lab), dtype=jnp.int32)
                else:
                    lab = jnp.broadcast_to(lab, (b,))
            return self.dit(log_sigma, x, lab, train, key=key)

        # single
        if label is None:
            lab = jnp.asarray(self.default_label, dtype=jnp.int32)
        else:
            lab = jnp.asarray(label, dtype=jnp.int32).reshape(())
        return self.dit(log_sigma, x, lab, train, key=key)


@dataclass(frozen=True)
class Mixer2dSpec:
    patch_size: int = 4
    hidden_size: int = 96
    mix_patch_size: int = 512
    mix_hidden_size: int = 512
    num_blocks: int = 4


@dataclass(frozen=True)
class UNet2dSpec:
    # matches your unet_2d.UNet signature
    is_biggan: bool = False
    dim_mults: tuple[int, ...] = (1, 2, 4)
    hidden_size: int = 64
    heads: int = 4
    dim_head: int = 32
    dropout_rate: float = 0.0
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (14, 7)


@dataclass(frozen=True)
class AdaptiveDiTSpec:
    patch_size: int = 4
    embed_dim: int = 384
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    time_emb_dim: int = 128
    num_classes: int = 10
    learn_sigma: bool = False
    default_label: int = 0


def build_edm_denoiser(
    name: str,
    *,
    img_size: tuple[int, int, int],
    key: jr.PRNGKey,
    mixer: Mixer2dSpec = Mixer2dSpec(),
    unet: UNet2dSpec = UNet2dSpec(),
    dit: AdaptiveDiTSpec = AdaptiveDiTSpec(),
) -> eqx.Module:
    """
    Returns a model with EDM signature:
        model(log_sigma, x_in, *, key=None, train=False, **kwargs) -> D
    """
    s = _norm_name(name)

    if s in ("mixer2d", "mixer"):
        from quantbayes.stochax.diffusion.models.mixer_2d import Mixer2d

        net = Mixer2d(
            img_size=img_size,
            patch_size=mixer.patch_size,
            hidden_size=mixer.hidden_size,
            mix_patch_size=mixer.mix_patch_size,
            mix_hidden_size=mixer.mix_hidden_size,
            num_blocks=mixer.num_blocks,
            t1=1.0,
            key=key,
        )
        return EDMAdapter(net)

    if s in ("unet2d", "unet"):
        from quantbayes.stochax.diffusion.models.unet_2d import UNet

        net = UNet(
            data_shape=img_size,
            is_biggan=unet.is_biggan,
            dim_mults=list(unet.dim_mults),
            hidden_size=unet.hidden_size,
            heads=unet.heads,
            dim_head=unet.dim_head,
            dropout_rate=unet.dropout_rate,
            num_res_blocks=unet.num_res_blocks,
            attn_resolutions=list(unet.attn_resolutions),
            key=key,
        )
        return EDMAdapter(net)

    if s in ("adaptive_dit", "adaptivedit", "dit"):
        from quantbayes.stochax.diffusion.models.adaptive_DiT import DiT

        net = DiT(
            img_size=img_size,
            patch_size=dit.patch_size,
            in_channels=img_size[0],
            embed_dim=dit.embed_dim,
            depth=dit.depth,
            n_heads=dit.n_heads,
            mlp_ratio=dit.mlp_ratio,
            dropout_rate=dit.dropout_rate,
            time_emb_dim=dit.time_emb_dim,
            num_classes=dit.num_classes,
            learn_sigma=dit.learn_sigma,
            key=key,
        )
        return AdaptiveDiTEDMAdapter(
            net, default_label=dit.default_label, learn_sigma=dit.learn_sigma
        )

    if s in ("rfftdit", "rfft_dit", "rfftadaptivedit"):
        # IMPORTANT: this assumes your new RFFT DiT already uses EDM signature.
        # If your class name differs, adjust import here.
        from quantbayes.stochax.diffusion.models.rfft_adaptive_dit import (
            RFFTAdaptiveDiT,
        )

        net = RFFTAdaptiveDiT(
            img_size=img_size,
            patch_size=dit.patch_size,
            embed_dim=dit.embed_dim,
            depth=dit.depth,
            n_heads=dit.n_heads,
            mlp_ratio=dit.mlp_ratio,
            dropout_rate=dit.dropout_rate,
            time_emb_dim=dit.time_emb_dim,
            num_classes=dit.num_classes,
            learn_sigma=False,  # keep False for EDM loss compatibility
            use_spectral_proj=True,
            key=key,
        )
        return EDMAdapter(net)

    raise ValueError(
        f"Unknown denoiser name: {name!r}. Valid: {list(DenoiserName.__args__)}"
    )
