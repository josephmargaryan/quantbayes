# quantbayes/ball_dp/reconstruction/nonconvex/trainers_eqx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple
import inspect
import numpy as np

from ..types import Array, StochasticTrainer


def _call_with_supported_kwargs(fn, **kwargs):
    """
    Robust adapter: only pass kwargs that exist in fn's signature.
    Prevents breakage when upstream trainer signatures differ.
    """
    sig = inspect.signature(fn)
    filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filt)


@dataclass
class EqxTrainerConfig:
    """
    IMPORTANT:
      In informed-adversary reconstruction, the target/candidate record MUST be trained on.
      So we train on ALL points. If you want "validation", we evaluate on a subset but
      we DO NOT remove any points from training.
    """

    batch_size: int = 256
    num_epochs: int = 10
    patience: int = 3
    # stochax.train internally shuffles; kept for interface parity (not forwarded)
    shuffle: bool = True

    # Evaluate on subset of training points (still included in training).
    val_fraction: float = 0.0
    val_seed: int = 0


def _make_val_subset(X, y, *, val_fraction: float, seed: int):
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
        from quantbayes.stochax import train

    NOTE: your train() does not accept shuffle=..., so we do NOT pass it.
    """

    make_model: Callable[[int], Tuple[Any, Any]]  # seed -> (model, state)
    make_optimizer: Callable[[Any], Tuple[Any, Any]]  # model -> (optimizer, opt_state)
    loss_fn: Callable  # (model, state, xb, yb, key) -> (loss, new_state)
    cfg: EqxTrainerConfig

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        import jax.random as jr
        import jax.numpy as jnp
        from quantbayes.stochax import train  # your function

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

        out = _call_with_supported_kwargs(
            train,
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
            # DO NOT pass shuffle: your train() doesn’t accept it
        )
        best_model, best_state = out[0], out[1]
        return best_model, best_state


@dataclass
class EqxDPSGDTrainer(StochasticTrainer):
    """
    Wraps your existing DP-SGD trainer (stochax):
        from quantbayes.stochax.privacy.dp_train import dp_eqx_train

    We filter kwargs by signature to avoid mismatches.
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    dp_config: Any
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

        X_train, y_train = Xj, yj
        X_val, y_val = _make_val_subset(
            Xj,
            yj,
            val_fraction=float(self.cfg.val_fraction),
            seed=int(seed + self.cfg.val_seed),
        )

        out = _call_with_supported_kwargs(
            dp_eqx_train,
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
            # shuffle filtered if not supported
            shuffle=bool(self.cfg.shuffle),
        )
        best_model, best_state = out[0], out[1]
        return best_model, best_state


@dataclass
class EqxStandardDPSGDTrainer(StochasticTrainer):
    """
    Standard DP-SGD inside quantbayes.ball_dp:
        from quantbayes.ball_dp.privacy.dpsgd_train import dp_eqx_train
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    dp_config: Any
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

        n = int(Xj.shape[0])
        n_val = max(1, int(0.1 * n))
        # keep candidate IN training set: do not drop last point; use subset for val
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
    Ball-DP-SGD inside quantbayes.ball_dp:
        from quantbayes.ball_dp.privacy.ball_dpsgd_train import ball_dp_eqx_train
    """

    make_model: Callable[[int], Tuple[Any, Any]]
    make_optimizer: Callable[[Any], Tuple[Any, Any]]
    loss_fn: Callable
    ball_dp_config: Any
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
