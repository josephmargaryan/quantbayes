# quantbayes/ball_dp/privacy/dpsgd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

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


def _tree_scale(a: Pytree, s: float) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x * s, a)


@dataclass
class DPSGDConfig:
    """
    Standard (ε,δ)-DP-SGD config.

    - clipping_norm C: L2 clip per-example gradients.
    - noise_multiplier nm: Gaussian noise std on SUM is σ_sum = nm * C.
    - Accountant: WOR-RDP bound in rdp_wor_gaussian.py.
    """

    clipping_norm: float
    noise_multiplier: float
    delta: float = 1e-5
    orders: Tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 256])


class DPPrivacyEngine:
    """
    Standard DP-SGD engine for Eqx models, using per-example clipping + Gaussian noise
    and WOR-RDP accountant.

    This is intentionally parallel to BallDPPrivacyEngine so you can compare DP vs Ball-DP.
    """

    def __init__(self, cfg: DPSGDConfig):
        if cfg.clipping_norm is None or float(cfg.clipping_norm) <= 0:
            raise ValueError("DPSGDConfig.clipping_norm must be > 0.")
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

        Standard DP-SGD:
          - Clip per-example grads to C
          - Noise on SUM: std_sum = nm * C
          - Mean grad = (sum + noise)/B
        """
        B = int(xb.shape[0])
        if B <= 0:
            raise ValueError("Empty batch.")

        rng = key
        rng, sub = jr.split(rng)

        g_batched = self._per_example_grads(loss_fn, model, state, xb, yb, sub)
        g_batched = _per_example_clip(g_batched, float(self.cfg.clipping_norm))
        g_sum = _sum_over_batch(g_batched)

        std_sum = float(self.cfg.noise_multiplier) * float(self.cfg.clipping_norm)

        leaves = jax.tree_util.tree_leaves(g_sum)
        keys = jr.split(rng, len(leaves))
        it = iter(keys)

        def add_noise(leaf):
            return leaf + std_sum * jr.normal(
                next(it), shape=leaf.shape, dtype=leaf.dtype
            )

        noisy_sum = jax.tree_util.tree_map(add_noise, g_sum)
        noisy_mean = _tree_scale(noisy_sum, 1.0 / float(B))

        # WOR-RDP update: u=(Δ/σ)^2 = (C/(nm*C))^2 = 1/nm^2
        nm = float(self.cfg.noise_multiplier)
        u = 1.0 / (nm * nm)
        self.accountant.accumulate(steps=1, q=float(sample_rate), u=float(u))

        # update state once (if your loss_fn carries state)
        _, new_state = loss_fn(model, state, xb, yb, jr.fold_in(key, 12345))
        return noisy_mean, new_state

    def epsilon(self, *, delta: Optional[float] = None) -> float:
        delt = float(self.cfg.delta if delta is None else delta)
        return float(self.accountant.epsilon(delta=delt))
