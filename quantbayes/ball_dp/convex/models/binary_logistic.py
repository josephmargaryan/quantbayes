# quantbayes/ball_dp/convex/models/binary_logistic.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class BinaryLogisticModel(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, d_in: int, *, key: jr.PRNGKey, init_scale: float = 1e-2):
        k1, _ = jr.split(key)
        self.w = init_scale * jr.normal(k1, (d_in,))
        self.b = jnp.array(0.0, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.w) + self.b


def encode_binary_pm1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    uniq = sorted(np.unique(y).tolist())
    if uniq == [-1, 1]:
        return y.astype(np.int64)
    if uniq == [0, 1]:
        return (2 * y.astype(np.int64) - 1).astype(np.int64)
    raise ValueError(
        f"Binary logistic requires labels in {{0,1}} or {{-1,+1}}; got {uniq}"
    )


def binary_logistic_loss(
    model: BinaryLogisticModel,
    state: Any,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    key: jr.PRNGKey,
    *,
    lam: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    scores = jax.vmap(model)(xb)
    loss = jnp.mean(jnp.logaddexp(0.0, -yb * scores))
    reg = 0.5 * lam * (jnp.sum(model.w**2) + model.b**2)
    return loss + reg, {"loss": loss, "reg": reg}


def binary_logits(model: BinaryLogisticModel, x: np.ndarray) -> np.ndarray:
    xj = jnp.asarray(x)
    return np.asarray(jax.vmap(model)(xj))


def binary_predictions(model: BinaryLogisticModel, x: np.ndarray) -> np.ndarray:
    logits = binary_logits(model, x)
    return (logits >= 0.0).astype(np.int64)


def binary_logistic_grad_vector(
    model: BinaryLogisticModel, x: np.ndarray, y_pm1: int
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y_pm1 = int(y_pm1)
    score = float(np.dot(x, np.asarray(model.w)) + float(model.b))
    coeff = -float(y_pm1) / (1.0 + np.exp(float(y_pm1) * score))
    return coeff * np.concatenate([x, np.array([1.0], dtype=np.float32)], axis=0)


def binary_logistic_missing_gradient(
    model: BinaryLogisticModel,
    x_minus: np.ndarray,
    y_minus_pm1: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    grad_sum = np.zeros(x_minus.shape[1] + 1, dtype=np.float64)
    for x, y in zip(x_minus, y_minus_pm1):
        grad_sum += binary_logistic_grad_vector(model, x, int(y))
    theta = np.concatenate(
        [
            np.asarray(model.w, dtype=np.float64),
            np.array([float(model.b)], dtype=np.float64),
        ]
    )
    return -float(lam) * float(n_total) * theta - grad_sum
