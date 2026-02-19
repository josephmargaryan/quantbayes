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

# Your convex head objective style:
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
    Deterministic full-batch optimizer for convex Eqx heads (simple baseline).
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

        for _ in range(int(self.cfg.steps)):
            key, sub = jr.split(key)
            model, opt_state, _ = step_fn(model, opt_state, sub)

        return model


@dataclass(frozen=True)
class FullBatchLBFGSConfig:
    """
    Uses your tested quantbayes.stochax.train_lbfgs.
    Set batch_size_full=True to run true full-batch LBFGS (recommended for convex ERM).
    """

    num_epochs: int = 80
    patience: int = 20
    batch_size_full: bool = True
    seed: int = 0


class EqxFullBatchLBFGSTrainer:
    """
    Deterministic full-batch L-BFGS trainer for convex heads.

    Why this exists:
      The softmax equation-solver reconstruction assumes the released model is at the true ERM optimum.
      Simple GD often isn’t accurate enough; L-BFGS gets you close enough that the missing-gradient
      becomes (approximately) rank-1 again.
    """

    def __init__(
        self, *, make_model: MakeModelFn, loss_fn: LossFn, cfg: FullBatchLBFGSConfig
    ):
        self.make_model = make_model
        self.loss_fn = loss_fn
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> eqx.Module:
        import jax.numpy as jnp
        from quantbayes.stochax import train_lbfgs  # your well-tested trainer

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        key = jr.PRNGKey(int(self.cfg.seed))
        model = self.make_model(key)
        state = None

        bs = (
            int(Xj.shape[0])
            if bool(self.cfg.batch_size_full)
            else min(1024, int(Xj.shape[0]))
        )

        # stochax.train_lbfgs expects loss_fn -> (loss, new_state)
        def loss_only(m, st, xb, yb, k):
            loss_val, _aux = self.loss_fn(m, st, xb, yb, k)
            return loss_val, st

        # For convex ERM we can set val=train (deterministic objective)
        best_model, best_state, *_ = train_lbfgs(
            model=model,
            state=state,
            X_train=Xj,
            y_train=yj,
            X_val=Xj,
            y_val=yj,
            batch_size=bs,
            num_epochs=int(self.cfg.num_epochs),
            patience=int(self.cfg.patience),
            key=key,
            loss_fn=loss_only,
            deterministic_objective=True,
        )
        return best_model


def softmax_params_numpy(model) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (W,b) from SoftmaxLinearEqx."""
    W = np.asarray(jax.device_get(model.W), dtype=np.float64)
    b = np.asarray(jax.device_get(model.b), dtype=np.float64)
    return W, b


def binary_linear_params_numpy(model) -> Tuple[np.ndarray, float]:
    """Extract (w,b) from LogisticRegressorEqx / SquaredHingeSVMEqx."""
    w = np.asarray(jax.device_get(model.w), dtype=np.float64)
    b = float(np.asarray(jax.device_get(model.b), dtype=np.float64))
    return w, b
