# quantbayes/stochax/privacy/dp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

Array = jnp.ndarray
Pytree = Any


def _per_example_l2_norms(grad_batched: Pytree) -> Array:
    leaves = jax.tree_util.tree_leaves(grad_batched)
    sq_parts = [
        jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in leaves
    ]  # (B,) each
    total_sq = sq_parts[0]
    for s in sq_parts[1:]:
        total_sq = total_sq + s
    return jnp.sqrt(total_sq + 1e-12)


def _scale_like_batch(g: Array, scale: Array) -> Array:
    return g * scale.reshape((scale.shape[0],) + (1,) * (g.ndim - 1))


def _per_example_clip(grad_batched: Pytree, C: float) -> Pytree:
    norms = _per_example_l2_norms(grad_batched)  # (B,)
    scale = jnp.minimum(1.0, C / (norms + 1e-12))  # (B,)
    return jax.tree_util.tree_map(lambda g: _scale_like_batch(g, scale), grad_batched)


def _mean_over_batch(grad_batched: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grad_batched)


def _add_gaussian_noise(p: Pytree, key: Array, std: float) -> Pytree:
    keys = jr.split(key, len(jax.tree_util.tree_leaves(p)))
    it = iter(keys)

    def add(leaf):
        return leaf + std * jr.normal(next(it), shape=leaf.shape, dtype=leaf.dtype)

    return jax.tree_util.tree_map(add, p)


@dataclass
class RDPAccountant:
    orders: tuple = (1.25, 1.5, 2, 3, 4, 8, 16, 32)
    alphas_eps: Optional[dict] = None
    steps: int = 0
    sigma: float = 1.0
    delta: float = 1e-5

    def reset(self, *, sigma: float, delta: float):
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.steps = 0
        self.alphas_eps = None

    def accumulate(self, num_steps: int = 1):
        self.steps += int(num_steps)
        self.alphas_eps = {
            a: self.steps * (a / (2.0 * self.sigma**2)) for a in self.orders
        }

    def get_epsilon(self) -> float:
        assert self.alphas_eps is not None
        candidates = []
        for a, eps_a in self.alphas_eps.items():
            if a <= 1.0:
                continue
            candidates.append(eps_a + jnp.log(1.0 / self.delta) / (a - 1.0))
        return float(jnp.min(jnp.array(candidates)))


@dataclass
class DPSGDConfig:
    clipping_norm: float
    noise_multiplier: float
    delta: float = 1e-5


class DPPrivacyEngine:
    """
    Per-example clipping + Gaussian noise; plug into local steps or inner loops.
    """

    def __init__(self, cfg: DPSGDConfig):
        self.cfg = cfg
        self.accountant = RDPAccountant()
        self.accountant.reset(sigma=cfg.noise_multiplier, delta=cfg.delta)

    def noisy_grad(
        self,
        loss_fn: Callable[
            [eqx.Module, Any, Array, Array, jax.Array], tuple[Array, Any]
        ],
        model: eqx.Module,
        state: Any,
        xb: Array,
        yb: Array,
        *,
        key: jax.Array,
    ):
        B = xb.shape[0]

        def loss_single(m, s, x1, y1, k1):
            (val, new_s), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                m, s, x1[None, ...], y1[None, ...], k1
            )
            return g, new_s

        keys = jr.split(key, B)
        grad_batched, state_batched = jax.vmap(
            loss_single, in_axes=(None, None, 0, 0, 0)
        )(model, state, xb, yb, keys)
        clipped = _per_example_clip(grad_batched, self.cfg.clipping_norm)
        mean_grad = _mean_over_batch(clipped)
        std = (self.cfg.noise_multiplier * self.cfg.clipping_norm) / max(1, B)
        noisy = _add_gaussian_noise(mean_grad, key, std)
        self.accountant.accumulate(1)
        new_state = jax.tree_util.tree_map(lambda *s: s[-1], state_batched)
        return noisy, new_state

    def epsilon(self) -> float:
        return self.accountant.get_epsilon()
