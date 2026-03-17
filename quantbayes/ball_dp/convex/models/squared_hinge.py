# quantbayes/ball_dp/convex/models/squared_hinge.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .binary_logistic import encode_binary_pm1


class SquaredHingeModel(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, d_in: int, *, key: jr.PRNGKey, init_scale: float = 1e-2):
        k1, _ = jr.split(key)
        self.w = init_scale * jr.normal(k1, (d_in,))
        self.b = jnp.array(0.0, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.w) + self.b


def squared_hinge_loss(
    model: SquaredHingeModel,
    state: Any,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    key: jr.PRNGKey,
    *,
    lam: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    scores = jax.vmap(model)(xb)
    margins = 1.0 - yb * scores
    loss = jnp.mean(jnp.maximum(0.0, margins) ** 2)
    reg = 0.5 * lam * (jnp.sum(model.w**2) + model.b**2)
    return loss + reg, {"loss": loss, "reg": reg}


def squared_hinge_scores(model: SquaredHingeModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.vmap(model)(jnp.asarray(x, dtype=jnp.float32)))


def squared_hinge_predictions(model: SquaredHingeModel, x: np.ndarray) -> np.ndarray:
    return np.where(squared_hinge_scores(model, x) >= 0.0, 1, -1).astype(np.int64)


def squared_hinge_grad_vector(
    model: SquaredHingeModel, x: np.ndarray, y_pm1: int
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y_pm1 = int(y_pm1)
    score = float(np.dot(x, np.asarray(model.w)) + float(model.b))
    margin = float(y_pm1) * score
    if margin >= 1.0:
        return np.zeros(x.shape[0] + 1, dtype=np.float64)
    coeff = -2.0 * (1.0 - margin) * float(y_pm1)
    return coeff * np.concatenate([x, np.array([1.0], dtype=np.float32)], axis=0)


def squared_hinge_missing_gradient(
    model: SquaredHingeModel,
    x_minus: np.ndarray,
    y_minus_pm1: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    grad_sum = np.zeros(x_minus.shape[1] + 1, dtype=np.float64)
    for x, y in zip(x_minus, y_minus_pm1):
        grad_sum += squared_hinge_grad_vector(model, x, int(y))
    theta = np.concatenate(
        [
            np.asarray(model.w, dtype=np.float64),
            np.array([float(model.b)], dtype=np.float64),
        ]
    )
    return -float(lam) * float(n_total) * theta - grad_sum
