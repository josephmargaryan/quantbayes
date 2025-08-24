# quantbayes/stochax/wrapper/wrapper_zoom.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from quantbayes.stochax.trainer.zoom import (
    train_zoom as _train_zoom,
    train_zoom_full_data as _train_zoom_full,
)
from quantbayes.stochax.trainer.train import (
    predict_batched_efficient as _predict_eff,
    regression_loss as _regression_loss,
    binary_loss as _binary_loss,
    multiclass_loss as _multiclass_loss,
    AugmentFn,
)

Array = jnp.ndarray

__all__ = [
    "EQXRegressorZoom",
    "EQXBinaryClassifierZoom",
    "EQXMulticlassClassifierZoom",
]


# —— utils ——
def _as_jnp(x, dtype=jnp.float32) -> Array:
    if isinstance(x, jnp.ndarray):
        return x.astype(dtype)
    return jnp.array(np.asarray(x), dtype=dtype)


def _rng(seed: Optional[int]) -> jr.key:
    return jr.PRNGKey(0 if seed is None else int(seed))


def _holdout_split(X: Array, y: Array, val_frac: float, seed: int):
    n = X.shape[0]
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    rs.shuffle(idx)
    cut = int(max(1, round((1.0 - val_frac) * n)))
    tr = idx[:cut]
    va = idx[cut:] if cut < n else idx[:1]
    return X[tr], X[va], y[tr], y[va]


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=axis, keepdims=True)


class _BaseZoom(BaseEstimator):
    """
    Sklearn-compatible wrapper for Zoom (Strong-Wolfe) line-search
    with SGD direction. Mirrors the LBFGS/backtracking wrappers:
    deterministic objective, penalties, checkpoints, optional refit
    on full data, and stable proba utilities.
    """

    def __init__(
        self,
        *,
        model_cls: Callable[..., eqx.Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        batch_size: int = 512,
        num_epochs: int = 20,
        patience: int = 5,
        val_frac: float = 0.1,
        # regularisation
        lambda_spec: float = 0.0,
        lambda_frob: float = 0.0,
        # data aug
        augment_fn: Optional[AugmentFn] = None,
        # inference
        predict_batch_size: int = 1024,
        # Zoom knobs
        max_linesearch_steps: int = 10,
        zoom_c1: float = 1e-4,
        zoom_c2: float = 0.9,
        deterministic_objective: bool = True,
        # checkpoints
        ckpt_path: Optional[str] = None,  # e.g. "ckpts/run-epoch-{epoch:03d}.ckpt"
        checkpoint_interval: Optional[int] = None,
        # optional refit full data for best_epoch length
        refit_on_full_data: bool = False,
        # loss (subclass default if None)
        loss_fn: Optional[Callable] = None,
    ):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.val_frac = val_frac

        self.lambda_spec = lambda_spec
        self.lambda_frob = lambda_frob
        self.augment_fn = augment_fn

        self.predict_batch_size = predict_batch_size

        self.max_linesearch_steps = max_linesearch_steps
        self.zoom_c1 = zoom_c1
        self.zoom_c2 = zoom_c2
        self.deterministic_objective = deterministic_objective

        self.ckpt_path = ckpt_path
        self.checkpoint_interval = checkpoint_interval
        self.refit_on_full_data = refit_on_full_data

        self.loss_fn = loss_fn

    def _more_tags(self):
        return {"requires_y": True, "non_deterministic": self.augment_fn is not None}

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_") and bool(self.is_fitted_)

    # —— shared internals ——
    def _prepare_xy(self, X, y):
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_jax = _as_jnp(y, dtype=jnp.float32)
        return X_jax, y_jax

    def _set_n_features(self, X: Array):
        if X.ndim != 2:
            raise ValueError(f"Expected X as (N, D); got {tuple(X.shape)}")
        self.n_features_in_ = X.shape[1]

    def _init_model(self, init_key: jr.key):
        kwargs = self.model_kwargs or {}
        return eqx.nn.make_with_state(self.model_cls)(init_key, **kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        import inspect

        # Pick the base class whose __init__ lists the real params
        sig = inspect.signature(
            self.__class__.__mro__[1].__init__
        )  # _BaseBacktrack.__init__ or _BaseLBFGS.__init__
        names = [n for n in sig.parameters.keys() if n != "self"]

        params: Dict[str, Any] = {}
        for name in names:
            if hasattr(self, name):
                val = getattr(self, name)
                params[name] = val
                if deep and hasattr(val, "get_params"):
                    sub = val.get_params(deep=True)
                    for k, v in sub.items():
                        params[f"{name}__{k}"] = v
        return params

    def fit(self, X, y):
        # to JAX
        X_jax, y_jax = self._prepare_xy(X, y)
        self._set_n_features(X_jax)

        # split
        seed = 0 if self.random_state is None else int(self.random_state)
        X_tr, X_val, y_tr, y_val = _holdout_split(X_jax, y_jax, self.val_frac, seed)

        # PRNG + init
        master = _rng(self.random_state)
        master, mkey, tkey = jr.split(master, 3)
        model, state = self._init_model(mkey)

        # train
        want_pen = True
        out = _train_zoom(
            model=model,
            state=state,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            lambda_spec=self.lambda_spec,
            lambda_frob=self.lambda_frob,
            key=tkey,
            augment_fn=self.augment_fn,
            loss_fn=self.loss_fn,
            max_linesearch_steps=self.max_linesearch_steps,
            zoom_c1=self.zoom_c1,
            zoom_c2=self.zoom_c2,
            deterministic_objective=self.deterministic_objective,
            ckpt_path=self.ckpt_path,
            checkpoint_interval=self.checkpoint_interval,
            return_penalty_history=want_pen,
        )

        best_model, best_state, tr_hist, va_hist, pen_hist = out
        self.train_losses_ = list(map(float, tr_hist))
        self.val_losses_ = list(map(float, va_hist))
        self.penalty_history_ = list(map(float, pen_hist))
        self.best_epoch_ = (
            int(np.argmin(self.val_losses_) + 1) if len(self.val_losses_) else 1
        )

        # optional refit on full data
        if self.refit_on_full_data:
            master, rf_key = jr.split(master)
            out_full = _train_zoom_full(
                model=best_model,
                state=best_state,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_val,
                y_val=y_val,
                batch_size=self.batch_size,
                num_epochs=self.best_epoch_,
                key=rf_key,
                augment_fn=self.augment_fn,
                lambda_spec=self.lambda_spec,
                lambda_frob=self.lambda_frob,
                loss_fn=self.loss_fn,
                max_linesearch_steps=self.max_linesearch_steps,
                zoom_c1=self.zoom_c1,
                zoom_c2=self.zoom_c2,
                deterministic_objective=self.deterministic_objective,
                ckpt_path=self.ckpt_path,
                checkpoint_interval=self.checkpoint_interval,
                return_penalty_history=False,
            )
            best_model, best_state, _, _ = out_full

        # freeze for inference
        self.model_ = best_model
        self.state_ = best_state
        self.predict_model_ = eqx.nn.inference_mode(best_model)
        self.is_fitted_ = True
        return self

    def _predict_logits(self, X) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("Estimator is not fitted.")
        X_jax = _as_jnp(X, dtype=jnp.float32)
        key = _rng(self.random_state)
        logits = _predict_eff(
            self.predict_model_,
            self.state_,
            X_jax,
            key,
            batch_size=int(self.predict_batch_size),
        )
        return np.array(jax.device_get(logits))

    def save_ckpt(self, prefix: str) -> None:
        import os, json, equinox as eqx

        if not self.__sklearn_is_fitted__():
            raise RuntimeError("Call fit() first.")
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

        eqx.tree_serialise_leaves(prefix + ".model.eqx", self.model_)
        eqx.tree_serialise_leaves(prefix + ".state.eqx", self.state_)
        # save the exact inference module used
        eqx.tree_serialise_leaves(prefix + ".predict.eqx", self.predict_model_)

        meta = {
            "n_features_in_": getattr(self, "n_features_in_", None),
            "classes_": (self.classes_.tolist() if hasattr(self, "classes_") else None),
        }
        with open(prefix + ".meta.json", "w") as f:
            json.dump(meta, f)

    def load_ckpt(self, prefix: str):
        import json, equinox as eqx, numpy as np

        key = _rng(self.random_state)
        model_t, state_t = self._init_model(key)

        self.model_ = eqx.tree_deserialise_leaves(prefix + ".model.eqx", model_t)
        self.state_ = eqx.tree_deserialise_leaves(prefix + ".state.eqx", state_t)
        try:
            self.predict_model_ = eqx.tree_deserialise_leaves(
                prefix + ".predict.eqx", model_t
            )
        except FileNotFoundError:
            self.predict_model_ = eqx.nn.inference_mode(self.model_)

        try:
            with open(prefix + ".meta.json", "r") as f:
                meta = json.load(f)
            nfi = meta.get("n_features_in_")
            if nfi is not None:
                self.n_features_in_ = int(nfi)
            cls = meta.get("classes_")
            if cls is not None:
                self.classes_ = np.array(cls)
        except FileNotFoundError:
            pass

        self.is_fitted_ = True
        return self


# —— Regressor ——
class EQXRegressorZoom(_BaseZoom, RegressorMixin):
    def __init__(self, **kwargs):
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _regression_loss
        super().__init__(**kwargs)

    def predict(self, X) -> np.ndarray:
        z = self._predict_logits(X)
        if z.ndim >= 2 and z.shape[-1] == 1:
            z = z.reshape(z.shape[0])
        return z


# —— Binary classifier ——
class EQXBinaryClassifierZoom(_BaseZoom, ClassifierMixin):
    def __init__(self, **kwargs):
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _binary_loss
        super().__init__(**kwargs)

    def _prepare_xy(self, X, y):
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_np = np.asarray(y).reshape(-1)

        if y_np.dtype == bool:
            uniques = np.array([False, True])
        else:
            uniques = np.unique(y_np)
            if len(uniques) != 2:
                raise ValueError(f"Expected 2 classes, got {uniques}.")
        self.classes_ = uniques

        mapping = {uniques[0]: 0, uniques[1]: 1}
        y01 = np.vectorize(mapping.get, otypes=[np.int32])(y_np)
        y_jax = _as_jnp(y01, dtype=jnp.float32)
        return X_jax, y_jax

    def decision_function(self, X) -> np.ndarray:
        z = self._predict_logits(X)
        if z.ndim >= 2 and z.shape[-1] == 1:
            z = z.reshape(z.shape[0])
        return z

    def predict_proba(self, X) -> np.ndarray:
        z = self.decision_function(X)
        p1 = _sigmoid_np(z)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


# —— Multiclass classifier ——
class EQXMulticlassClassifierZoom(_BaseZoom, ClassifierMixin):
    def __init__(self, **kwargs):
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _multiclass_loss
        super().__init__(**kwargs)

    def _prepare_xy(self, X, y):
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_np = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y_np)
        mapping = {lab: i for i, lab in enumerate(self.classes_)}
        y_idx = np.vectorize(mapping.get, otypes=[np.int32])(y_np)
        y_jax = _as_jnp(y_idx, dtype=jnp.int32)
        return X_jax, y_jax

    def decision_function(self, X) -> np.ndarray:
        z = self._predict_logits(X)
        if z.ndim == 1:
            z = z[:, None]
        return z

    def predict_proba(self, X) -> np.ndarray:
        return _softmax_np(self.decision_function(X), axis=1)

    def predict(self, X) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]
