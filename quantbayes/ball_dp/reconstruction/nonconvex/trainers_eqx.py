# quantbayes/ball_dp/reconstruction/nonconvex/trainers_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ..types import Array, StochasticTrainer

# JAX/Eqx typing is intentionally loose here to keep public API simple.


@dataclass
class EqxTrainerConfig:
    batch_size: int = 256
    num_epochs: int = 10
    patience: int = 3
    shuffle: bool = True


@dataclass
class EqxNonPrivateTrainer(StochasticTrainer):
    """
    Wraps your existing non-private stochax trainer:
        quantbayes.stochax.train.train(...)
    """

    make_model: Callable[[int], Tuple[Any, Any]]  # seed -> (model, state)
    make_optimizer: Callable[[Any], Tuple[Any, Any]]  # model -> (optimizer, opt_state)
    loss_fn: Callable  # (model, state, xb, yb, key) -> (loss, new_state)
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        import jax.random as jr
        import jax.numpy as jnp
        from quantbayes.stochax.train import train  # your existing function

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        # simple split: last 10% as val (informed adversary knows D^-; this is for training stability)
        n = int(Xj.shape[0])
        n_val = max(1, int(0.1 * n))
        X_train, y_train = Xj[:-n_val], yj[:-n_val]
        X_val, y_val = Xj[-n_val:], yj[-n_val:]

        out = train(
            model=model,
            state=state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=int(self.cfg.batch_size),
            num_epochs=int(self.cfg.num_epochs),
            patience=int(self.cfg.patience),
            key=key,
            shuffle=bool(self.cfg.shuffle),
        )
        # stochax.train returns (best_model, best_state, ...)
        best_model, best_state = out[0], out[1]
        return best_model, best_state


@dataclass
class EqxDPSGDTrainer(StochasticTrainer):
    """
    Wraps your existing DP-SGD trainer:
        quantbayes.stochax.privacy.dp_train.dp_eqx_train(...)
    """

    make_model: Callable[[int], Tuple[Any, Any]]  # seed -> (model, state)
    make_optimizer: Callable[[Any], Tuple[Any, Any]]  # model -> (optimizer, opt_state)
    loss_fn: Callable
    dp_config: Any  # quantbayes.stochax.privacy.dp.DPSGDConfig
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        import jax.random as jr
        import jax.numpy as jnp
        from quantbayes.stochax.privacy.dp_train import dp_eqx_train

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        n = int(Xj.shape[0])
        n_val = max(1, int(0.1 * n))
        X_train, y_train = Xj[:-n_val], yj[:-n_val]
        X_val, y_val = Xj[-n_val:], yj[-n_val:]

        best_model, best_state, *_ = dp_eqx_train(
            model=model,
            state=state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dp_config=self.dp_config,
            batch_size=int(self.cfg.batch_size),
            num_epochs=int(self.cfg.num_epochs),
            patience=int(self.cfg.patience),
            key=key,
            shuffle=bool(self.cfg.shuffle),
        )
        return best_model, best_state


@dataclass
class EqxBallDPSGDTrainer(StochasticTrainer):
    """
    Ball-DP-SGD trainer (your thesis Ball adjacency) using the new Ball-DP engine below.
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    ball_dp_config: Any  # quantbayes.ball_dp.privacy.ball_dpsgd.BallDPSGDConfig
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        import jax.random as jr
        import jax.numpy as jnp
        from quantbayes.ball_dp.privacy.ball_dpsgd_train import ball_dp_eqx_train

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        n = int(Xj.shape[0])
        n_val = max(1, int(0.1 * n))
        X_train, y_train = Xj[:-n_val], yj[:-n_val]
        X_val, y_val = Xj[-n_val:], yj[-n_val:]

        best_model, best_state, *_ = ball_dp_eqx_train(
            model=model,
            state=state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ball_dp_config=self.ball_dp_config,
            batch_size=int(self.cfg.batch_size),
            num_epochs=int(self.cfg.num_epochs),
            patience=int(self.cfg.patience),
            key=key,
            shuffle=bool(self.cfg.shuffle),
        )
        return best_model, best_state
