# quantbayes/stochax/federated/optimizers.py
from __future__ import annotations
from typing import List, Tuple, Any
import jax, jax.numpy as jnp, equinox as eqx

Pytree = Any


def _tree_add(a: Pytree, b: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a: Pytree, s: float) -> Pytree:
    return jax.tree_util.tree_map(lambda x: s * x, a)


def _tree_zeros_like(a: Pytree) -> Pytree:
    return jax.tree_util.tree_map(jnp.zeros_like, a)


class FedAdamServer:
    """
    Reddi et al. (2021): adaptive federated optimization.
    Maintain m,v on the server over model deltas.
    """

    def __init__(self, lr_server=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr_server
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def apply(
        self, theta_g: eqx.Module, local_models: List[eqx.Module], weights: List[float]
    ) -> eqx.Module:
        p_g = eqx.filter(theta_g, eqx.is_inexact_array)
        # weighted delta
        p_loc = [eqx.filter(m, eqx.is_inexact_array) for m in local_models]

        def wsum(*leaves):
            return sum(w * leaf for w, leaf in zip(weights, leaves))

        avg_params = jax.tree_util.tree_map(wsum, *p_loc)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, avg_params, p_g)

        if self.m is None:
            self.m = _tree_zeros_like(delta)
            self.v = _tree_zeros_like(delta)
        self.t += 1
        self.m = jax.tree_util.tree_map(
            lambda m, d: self.b1 * m + (1 - self.b1) * d, self.m, delta
        )
        self.v = jax.tree_util.tree_map(
            lambda v, d: self.b2 * v + (1 - self.b2) * (d * d), self.v, delta
        )

        mhat = _tree_scale(self.m, 1.0 / (1 - self.b1**self.t))
        vhat = _tree_scale(self.v, 1.0 / (1 - self.b2**self.t))
        update = jax.tree_util.tree_map(
            lambda m, v: self.lr * m / (jnp.sqrt(v) + self.eps), mhat, vhat
        )
        new_params = jax.tree_util.tree_map(lambda a, u: a + u, p_g, update)
        _, static = eqx.partition(theta_g, eqx.is_inexact_array)
        return eqx.combine(new_params, static)
