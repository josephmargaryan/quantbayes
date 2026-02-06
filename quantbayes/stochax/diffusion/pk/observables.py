from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import jax
import jax.numpy as jnp


class CoarseObservable(Protocol):
    """Coarse map d = T(x) with analytic gradient wrt x."""

    def value(self, x: jnp.ndarray) -> jnp.ndarray: ...
    def value_and_grad(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]: ...


@dataclass(frozen=True)
class InkFractionObservable:
    """
    Ink fraction observable for MNIST-like images in [-1, 1].

    Define "ink" as a soft threshold on x01 = (x+1)/2 in [0,1]:
        s = sigmoid((x01 - thr)/temp)
        d = mean(s)

    Returns:
      d:    (B,)
      grad: (B, ...) same shape as x
    """

    thr: float = 0.35
    temp: float = 0.08
    clip_min: float = -1.0
    clip_max: float = 1.0

    def _ensure_batched(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        x = jnp.asarray(x)
        if x.ndim == 0:
            raise ValueError("InkFractionObservable expects at least 1D input.")
        if x.ndim in (3, 2, 1):
            # allow (C,H,W) or (H,W) or (P,) but treat as single sample
            return x[None, ...], True
        return x, False

    def value_and_grad(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xb, squeeze = self._ensure_batched(x)

        temp = jnp.maximum(jnp.asarray(self.temp, dtype=xb.dtype), 1e-12)

        # clip
        x_clipped = jnp.clip(xb, self.clip_min, self.clip_max)

        # map [-1,1] -> [0,1] (assumes clip_min=-1, clip_max=1)
        x01 = (x_clipped - self.clip_min) / (self.clip_max - self.clip_min)

        u = (x01 - self.thr) / temp
        s = jax.nn.sigmoid(u)

        # mean over all non-batch axes
        axes = tuple(range(1, s.ndim))
        d = jnp.mean(s, axis=axes)  # (B,)

        # analytic gradient:
        # ds/dx = s(1-s) * d(x01)/dx * 1/temp
        # x01 = (x - clip_min)/(clip_max-clip_min) => d(x01)/dx = 1/(clip_max-clip_min)
        scale = 1.0 / (self.clip_max - self.clip_min)
        ds_dx = s * (1.0 - s) * (scale / temp)

        # mean => divide by number of elements per sample
        n_pix = 1
        for dim in xb.shape[1:]:
            n_pix *= int(dim)
        ds_dx = ds_dx / float(n_pix)

        # “respect clipping”: gradient is ~0 when clipped flat
        inside = (xb > self.clip_min) & (xb < self.clip_max)
        grad = ds_dx * inside.astype(xb.dtype)

        if squeeze:
            return d[0], grad[0]
        return d, grad

    def value(self, x: jnp.ndarray) -> jnp.ndarray:
        d, _ = self.value_and_grad(x)
        return d
