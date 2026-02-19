# quantbayes/ball_dp/reconstruction/convex/eqx_trainers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

Array = jnp.ndarray

# Loss signature used by your convex heads:
#   loss_fn(model, state, xb, yb, key) -> (loss_scalar, aux_dict)
LossFn = Callable[
    [eqx.Module, Any, Array, Array, jr.PRNGKey], Tuple[Array, Dict[str, Array]]
]
MakeModelFn = Callable[[jr.PRNGKey], eqx.Module]


@dataclass(frozen=True)
class FullBatchGDConfig:
    steps: int = 2000
    learning_rate: float = 0.1
    grad_clip_norm: Optional[float] = None
    seed: int = 0
    jit: bool = True


class EqxFullBatchGDTrainer:
    """
    Deterministic full-batch optimizer for convex Eqx heads.

    - Trains on the full dataset (no minibatching)
    - Deterministic given cfg.seed and a deterministic loss_fn
    - Returns the trained Eqx model
    """

    def __init__(
        self, *, make_model: MakeModelFn, loss_fn: LossFn, cfg: FullBatchGDConfig
    ):
        self.make_model = make_model
        self.loss_fn = loss_fn
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> eqx.Module:
        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y))

        key = jr.PRNGKey(int(self.cfg.seed))
        model = self.make_model(key)
        state = None

        if self.cfg.grad_clip_norm is not None and float(self.cfg.grad_clip_norm) > 0:
            tx = optax.chain(
                optax.clip_by_global_norm(float(self.cfg.grad_clip_norm)),
                optax.sgd(float(self.cfg.learning_rate)),
            )
        else:
            tx = optax.sgd(float(self.cfg.learning_rate))

        params0 = eqx.filter(model, eqx.is_inexact_array)
        opt_state = tx.init(params0)

        def step_fn(m, opt_state, k):
            (loss_val, _aux), grads = eqx.filter_value_and_grad(
                self.loss_fn, has_aux=True
            )(m, state, Xj, yj, k)
            params = eqx.filter(m, eqx.is_inexact_array)
            updates, opt_state2 = tx.update(grads, opt_state, params=params)
            m2 = eqx.apply_updates(m, updates)
            return m2, opt_state2, loss_val

        if self.cfg.jit:
            step_fn = eqx.filter_jit(step_fn)

        for t in range(int(self.cfg.steps)):
            key, sub = jr.split(key)
            model, opt_state, _ = step_fn(model, opt_state, sub)

        return model


def softmax_params_numpy(model) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (W,b) as numpy arrays from SoftmaxLinearEqx."""
    W = np.asarray(jax.device_get(model.W), dtype=np.float64)
    b = np.asarray(jax.device_get(model.b), dtype=np.float64)
    return W, b


def binary_linear_params_numpy(model) -> Tuple[np.ndarray, float]:
    """Extract (w,b) as numpy arrays from LogisticRegressorEqx / SquaredHingeSVMEqx."""
    w = np.asarray(jax.device_get(model.w), dtype=np.float64)
    b = float(np.asarray(jax.device_get(model.b), dtype=np.float64))
    return w, b
