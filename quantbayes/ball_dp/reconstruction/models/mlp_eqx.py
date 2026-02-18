# quantbayes/ball_dp/reconstruction/models/mlp_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class MLPClassifierEqx(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 10,
        *,
        width: int = 256,
        depth: int = 2,
        key: jr.PRNGKey,
        activation: str = "gelu",
    ):
        act = jax.nn.gelu if activation == "gelu" else jax.nn.relu
        self.mlp = eqx.nn.MLP(
            in_size=int(in_dim),
            out_size=int(n_classes),
            width_size=int(width),
            depth=int(depth),
            activation=act,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)


def multiclass_loss(
    model: eqx.Module,
    state: Any,
    xb: jnp.ndarray,  # (B,d)
    yb: jnp.ndarray,  # (B,)
    key: jr.PRNGKey,
    *,
    weight_decay: float = 0.0,
) -> Tuple[jnp.ndarray, Any]:
    logits = eqx.filter_vmap(model)(xb)  # (B,K)
    logZ = jax.nn.logsumexp(logits, axis=1)
    logp_y = logits[jnp.arange(logits.shape[0]), yb] - logZ
    loss = -jnp.mean(logp_y)

    if weight_decay > 0:
        params = eqx.filter(model, eqx.is_inexact_array)
        leaves = jax.tree_util.tree_leaves(params)
        l2 = sum([jnp.sum(p * p) for p in leaves])
        loss = loss + 0.5 * weight_decay * l2

    return loss, state
