# quantbayes/stochax/diffusion/schedules/vp.py
from __future__ import annotations
import jax.numpy as jnp


def make_vp_int_beta(schedule="linear", *, beta_min=0.1, beta_max=20.0, t1=1.0):
    s = (schedule or "linear").lower()
    t1 = jnp.asarray(t1)
    if s == "linear":
        b0, bdiff = jnp.asarray(beta_min), jnp.asarray(beta_max - beta_min)
        return lambda t: b0 * t + 0.5 * bdiff * (t * t) / jnp.maximum(t1, 1e-8)
    if s == "cosine":
        k = 0.008
        c = jnp.pi / 2.0

        def alpha_bar(t):
            u = jnp.clip(t / jnp.maximum(t1, 1e-8), 0.0, 1.0)
            num = jnp.cos((u + k) / (1.0 + k) * c) ** 2
            den = jnp.cos(k / (1.0 + k) * c) ** 2
            return jnp.clip(num / den, 1e-12, 1.0)

        return lambda t: -jnp.log(alpha_bar(t))
    raise ValueError(f"unknown schedule {schedule!r}")
