# quantbayes/ball_dp/privacy/ball_dpsgd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .rdp_wor_gaussian import RDPAccountantWOR

Array = jnp.ndarray
Pytree = Any


def _per_example_l2_norms(grad_batched: Pytree) -> Array:
    leaves = jax.tree_util.tree_leaves(grad_batched)
    parts = [
        jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in leaves
    ]  # (B,)
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return jnp.sqrt(total + 1e-12)


def _scale_like_batch(g: Array, scale: Array) -> Array:
    return g * scale.reshape((scale.shape[0],) + (1,) * (g.ndim - 1))


def _per_example_clip(grad_batched: Pytree, C: float) -> Pytree:
    norms = _per_example_l2_norms(grad_batched)
    scale = jnp.minimum(1.0, float(C) / (norms + 1e-12))
    return jax.tree_util.tree_map(lambda g: _scale_like_batch(g, scale), grad_batched)


def _sum_over_batch(grad_batched: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), grad_batched)


def _tree_add(a: Pytree, b: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a: Pytree, s: float) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x * s, a)


@dataclass
class BallDPSGDConfig:
    """
    Ball-DP-SGD config.

    Key parameters:
      - radius r and certified/assumed Lz (record Lipschitz of per-example gradients)
      - noise_multiplier nm: defines Gaussian noise scale relative to sensitivity Δ:
            noise_std_on_sum = nm * (Lz * r)
        (then dividing by batch size yields noise on the mean)

    Accountant:
      - fixed-size subsampling without replacement RDP bound (Wang-style)
      - delta used only for reporting epsilon; you can set any delta you want
    """

    radius: float
    lz: float
    noise_multiplier: float
    delta: float = 1e-5

    # Optional optimization clipping (NOT needed for privacy if Lz certificate is valid)
    clipping_norm: Optional[float] = None

    orders: Tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 256])


class BallDPPrivacyEngine:
    """
    Computes noisy gradients for Ball-DP-SGD and tracks privacy with WOR-RDP accountant.
    """

    def __init__(self, cfg: BallDPSGDConfig):
        self.cfg = cfg
        self.accountant = RDPAccountantWOR(orders=cfg.orders)

    def _per_example_grads(
        self,
        loss_fn,
        model: eqx.Module,
        state: Any,
        xb: Array,
        yb: Array,
        key: jr.PRNGKey,
    ) -> Pytree:
        infer_model = eqx.nn.inference_mode(model, value=True)

        def single(x1, y1, k1):
            (val, _aux), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                infer_model, state, x1[None, ...], y1[None, ...], k1
            )
            return g

        keys = jr.split(key, xb.shape[0])
        return jax.vmap(single)(xb, yb, keys)

    def noisy_grad(
        self,
        loss_fn,
        model: eqx.Module,
        state: Any,
        xb: Array,
        yb: Array,
        *,
        key: jr.PRNGKey,
        sample_rate: float,
    ) -> Tuple[Pytree, Any]:
        """
        Returns (noisy_mean_grad, new_state).
        Noise is added to the SUM of (optionally clipped) per-example grads:

          sensitivity (Ball, replacement within radius r):
            || sum_g(D) - sum_g(D') || <= Lz * r
          so noise std on SUM:
            std = nm * (Lz * r)

        Then we divide by batch size to get mean gradient.
        """
        B = int(xb.shape[0])
        if B <= 0:
            raise ValueError("Empty batch.")

        rng = key
        rng, sub = jr.split(rng)

        g_batched = self._per_example_grads(loss_fn, model, state, xb, yb, sub)
        if self.cfg.clipping_norm is not None and self.cfg.clipping_norm > 0:
            g_batched = _per_example_clip(g_batched, float(self.cfg.clipping_norm))

        g_sum = _sum_over_batch(g_batched)

        # noise on SUM calibrated to Ball sensitivity
        std_sum = (
            float(self.cfg.noise_multiplier)
            * float(self.cfg.lz)
            * float(self.cfg.radius)
        )

        leaves = jax.tree_util.tree_leaves(g_sum)
        keys = jr.split(rng, len(leaves))
        it = iter(keys)

        def add_noise(leaf):
            return leaf + std_sum * jr.normal(
                next(it), shape=leaf.shape, dtype=leaf.dtype
            )

        noisy_sum = jax.tree_util.tree_map(add_noise, g_sum)
        noisy_mean = _tree_scale(noisy_sum, 1.0 / float(B))

        # update accountant for this step
        # base u = (Δ/σ)^2 = (Lz*r / (nm*Lz*r))^2 = 1/nm^2
        nm = float(self.cfg.noise_multiplier)
        u = 1.0 / (nm * nm)
        self.accountant.accumulate(steps=1, q=float(sample_rate), u=float(u))

        # update state once
        _, new_state = loss_fn(model, state, xb, yb, jr.fold_in(key, 12345))
        return noisy_mean, new_state

    def epsilon(self, *, delta: Optional[float] = None) -> float:
        delt = float(self.cfg.delta if delta is None else delta)
        return float(self.accountant.epsilon(delta=delt))
