# quantbayes/ball_dp/convex/models/softmax_logistic.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class SoftmaxLinearModel(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(
        self, d_in: int, n_classes: int, *, key: jr.PRNGKey, init_scale: float = 1e-2
    ):
        k1, k2 = jr.split(key)
        self.W = init_scale * jr.normal(k1, (n_classes, d_in))
        self.b = init_scale * jr.normal(k2, (n_classes,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.W @ x + self.b


def softmax_loss(
    model: SoftmaxLinearModel,
    state: Any,
    xb: jnp.ndarray,
    yb: jnp.ndarray,
    key: jr.PRNGKey,
    *,
    lam: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    logits = jax.vmap(model)(xb)
    log_z = jax.nn.logsumexp(logits, axis=1)
    logp = logits[jnp.arange(logits.shape[0]), yb.astype(jnp.int32)] - log_z
    loss = -jnp.mean(logp)
    reg = 0.5 * lam * (jnp.sum(model.W**2) + jnp.sum(model.b**2))
    return loss + reg, {"loss": loss, "reg": reg}


def softmax_logits(model: SoftmaxLinearModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.vmap(model)(jnp.asarray(x, dtype=jnp.float32)))


def softmax_predictions(model: SoftmaxLinearModel, x: np.ndarray) -> np.ndarray:
    return np.argmax(softmax_logits(model, x), axis=-1).astype(np.int64)


def softmax_grad_matrix(model: SoftmaxLinearModel, x: np.ndarray, y: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    logits = np.asarray(model.W) @ x + np.asarray(model.b)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    a = probs.copy()
    a[int(y)] -= 1.0
    x_aug = np.concatenate([x, np.array([1.0], dtype=np.float32)], axis=0)
    return a[:, None] * x_aug[None, :]


def softmax_missing_gradient(
    model: SoftmaxLinearModel,
    x_minus: np.ndarray,
    y_minus: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    k = int(model.W.shape[0])
    d_aug = int(model.W.shape[1]) + 1
    grad_sum = np.zeros((k, d_aug), dtype=np.float64)
    for x, y in zip(x_minus, y_minus):
        grad_sum += softmax_grad_matrix(model, x, int(y))
    W_aug = np.concatenate(
        [
            np.asarray(model.W, dtype=np.float64),
            np.asarray(model.b, dtype=np.float64)[:, None],
        ],
        axis=1,
    )
    return -float(lam) * float(n_total) * W_aug - grad_sum
