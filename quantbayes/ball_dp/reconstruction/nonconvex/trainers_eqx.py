# quantbayes/ball_dp/reconstruction/nonconvex/trainers_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ..types import Array, StochasticTrainer


@dataclass
class EqxTrainerConfig:
    """
    Common training config for reconstruction experiments.

    IMPORTANT:
      In informed-adversary reconstruction, the target/candidate record must be TRAINED ON.
      Therefore we DO NOT do a disjoint train/val split that can drop the last appended record.
      We train on ALL points, and (optionally) evaluate on:
        - the full training set, or
        - a subset (still included in training).
    """

    batch_size: int = 256
    num_epochs: int = 10
    patience: int = 3
    shuffle: bool = True

    # If >0, we evaluate on a subset of the training data for early stopping,
    # but we still train on ALL points.
    val_fraction: float = 0.0
    val_seed: int = 0


def _make_val_subset(
    X: Any,
    y: Any,
    *,
    val_fraction: float,
    seed: int,
):
    import jax.random as jr
    import jax.numpy as jnp

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = int(X.shape[0])

    if val_fraction <= 0.0 or n <= 1:
        return X, y

    m = max(1, int(val_fraction * n))
    key = jr.PRNGKey(int(seed))
    idx = jr.permutation(key, n)[:m]
    return X[idx], y[idx]


@dataclass
class EqxNonPrivateTrainer(StochasticTrainer):
    """
    Wraps your existing non-private trainer:
        quantbayes.stochax.train.train(...)

    Trains on ALL points to respect the informed-adversary threat model.
    """

    make_model: Callable[[int], Tuple[Any, Any]]  # seed -> (model, state)
    make_optimizer: Callable[[Any], Tuple[Any, Any]]  # model -> (optimizer, opt_state)
    loss_fn: Callable  # (model, state, xb, yb, key) -> (loss, new_state)
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        try:
            import jax.random as jr
            import jax.numpy as jnp
            from quantbayes.stochax import train  # your existing function
        except Exception as e:
            raise ImportError(
                "EqxNonPrivateTrainer requires quantbayes.stochax.train."
            ) from e

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        X_train, y_train = Xj, yj
        X_val, y_val = _make_val_subset(
            Xj,
            yj,
            val_fraction=float(self.cfg.val_fraction),
            seed=int(seed + self.cfg.val_seed),
        )

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
        best_model, best_state = out[0], out[1]
        return best_model, best_state


@dataclass
class EqxDPSGDTrainer(StochasticTrainer):
    """
    Backwards-compatible wrapper around your existing stochax DP-SGD trainer:
        quantbayes.stochax.privacy.dp_train.dp_eqx_train(...)

    Trains on ALL points; val is a subset (still included in training).
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    dp_config: Any  # quantbayes.stochax.privacy.dp.DPSGDConfig
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        try:
            import jax.random as jr
            import jax.numpy as jnp
            from quantbayes.stochax.privacy.dp_train import dp_eqx_train
        except Exception as e:
            raise ImportError(
                "EqxDPSGDTrainer requires quantbayes.stochax.privacy.dp_train."
            ) from e

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        X_train, y_train = Xj, yj
        X_val, y_val = _make_val_subset(
            Xj,
            yj,
            val_fraction=float(self.cfg.val_fraction),
            seed=int(seed + self.cfg.val_seed),
        )

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
class EqxStandardDPSGDTrainer(StochasticTrainer):
    """
    Standard DP-SGD trainer implemented inside quantbayes.ball_dp (no stochax dependency).

    Uses:
      quantbayes.ball_dp.privacy.dpsgd_train.dp_eqx_train
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    dp_config: Any  # quantbayes.ball_dp.privacy.dpsgd.DPSGDConfig
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        import jax.random as jr
        import jax.numpy as jnp
        from quantbayes.ball_dp.privacy.dpsgd_train import dp_eqx_train

        key = jr.PRNGKey(int(seed))
        model, state = self.make_model(int(seed))
        optimizer, opt_state = self.make_optimizer(model)

        Xj = jnp.asarray(np.asarray(X, dtype=np.float32))
        yj = jnp.asarray(np.asarray(y, dtype=np.int64))

        X_train, y_train = Xj, yj
        X_val, y_val = _make_val_subset(
            Xj,
            yj,
            val_fraction=float(self.cfg.val_fraction),
            seed=int(seed + self.cfg.val_seed),
        )

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
    Ball-DP-SGD trainer (your thesis Ball adjacency).

    Uses:
      quantbayes.ball_dp.privacy.ball_dpsgd_train.ball_dp_eqx_train

    Trains on ALL points; val is a subset (still included in training).
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

        X_train, y_train = Xj, yj
        X_val, y_val = _make_val_subset(
            Xj,
            yj,
            val_fraction=float(self.cfg.val_fraction),
            seed=int(seed + self.cfg.val_seed),
        )

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
