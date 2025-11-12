# quantbayes/stochax/diffusion/models/wrappers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.conditioning.cfg import rescale_cfg


@dataclass
class UnconditionalWrapper(eqx.Module):
    model: eqx.Module
    time_mode: str = "vp_t"  # or "log_sigma"

    def __call__(self, t_or_sigma, x, *, key=None, train: bool = False, cond=None):
        t = t_or_sigma
        if self.time_mode == "log_sigma":
            t = jnp.log(jnp.asarray(t_or_sigma, dtype=x.dtype))
        return self.model(t, x, key=key)


@dataclass
class DiTWrapper(eqx.Module):
    model: eqx.Module
    num_classes: int
    time_mode: str = "log_sigma"
    null_label_index: int | None = None
    cfg_rescale: float | None = 0.7  # <-- enable by default (set to None to disable)

    def _time(self, t_or_sigma, x):
        return (
            jnp.log(jnp.asarray(t_or_sigma, dtype=x.dtype))
            if self.time_mode == "log_sigma"
            else t_or_sigma
        )

    def __call__(
        self,
        t_or_sigma,
        x,
        *,
        label: Optional[jnp.ndarray] = None,
        cfg_scale: Optional[float] = None,
        train: bool = False,
        key=None,
    ):
        t = self._time(t_or_sigma, x)

        # If no label provided and no CFG requested, use unconditional label.
        if (label is None) and (cfg_scale is None):
            nl = (
                self.num_classes
                if self.null_label_index is None
                else self.null_label_index
            )
            label = (
                jnp.full((x.shape[0],), nl, dtype=jnp.int32)
                if x.ndim == 4
                else jnp.array(nl, jnp.int32)
            )

        # No CFG mixing requested -> single forward.
        if (cfg_scale is None) or (label is None):
            return self.model(t, x, label, train, key=key)

        # CFG branch
        key_u, key_c = jax.random.split(key) if key is not None else (None, None)
        nl = (
            self.num_classes if self.null_label_index is None else self.null_label_index
        )
        uncond_label = jnp.full_like(label, nl)

        y_u = self.model(t, x, uncond_label, train, key=key_u)
        y_c = self.model(t, x, label, train, key=key_c)
        guided = y_u + cfg_scale * (y_c - y_u)

        if self.cfg_rescale is not None:
            guided = rescale_cfg(y_u, guided, rescale=self.cfg_rescale)

        return guided
