# quantbayes/stochax/wrapper/wrapper_backtrack.py

import numpy as np
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split

from quantbayes.stochax import train_backtrack
from quantbayes.stochax import predict as nn_predict
from quantbayes.stochax import regression_loss, binary_loss, multiclass_loss
from quantbayes.stochax.trainer.train import AugmentFn

__all__ = [
    "EQXRegressorBacktrack",
    "EQXBinaryClassifierBacktrack",
    "EQXMulticlassClassifierBacktrack",
]


class EQXBaseBacktrack(BaseEstimator):
    def __init__(
        self,
        model_cls: Callable[..., eqx.Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        key_seed: int = 0,
        batch_size: int = 512,
        num_epochs: int = 20,
        patience: int = 5,
        val_frac: float = 0.1,
        lambda_spec: float = 0.0,
        augment_fn: Optional[AugmentFn] = None,
        return_penalty_history: bool = False,
    ):
        # store exactly what the user passed
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.key_seed = key_seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.lambda_spec = lambda_spec
        self.augment_fn = augment_fn
        self.return_penalty_history = return_penalty_history

    def get_params(self, deep=True):
        # mirror the __init__ signature exactly
        return {
            "model_cls": self.model_cls,
            "model_kwargs": self.model_kwargs,
            "key_seed": self.key_seed,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "val_frac": self.val_frac,
            "lambda_spec": self.lambda_spec,
            "augment_fn": self.augment_fn,
            "return_penalty_history": self.return_penalty_history,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_fn: Callable[..., Tuple[Any, Any, List[float], List[float], List[float]]],
        loss_fn: Callable,
    ):
        # 1) numpy → JAX
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # record feature count for sklearn compatibility
        self.n_features_in_ = X.shape[1]

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)

        # 2) train/val split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_jax,
            y_jax,
            test_size=self.val_frac,
            random_state=self.key_seed,
            shuffle=True,
        )

        # 3) PRNG setup
        master_key = jr.PRNGKey(self.key_seed)
        master_key, init_key, train_key = jr.split(master_key, 3)

        # 4) init model + state (treat None → {} here)
        mkw = self.model_kwargs or {}
        self.model, self.state = eqx.nn.make_with_state(self.model_cls)(init_key, **mkw)

        # 5) train loop
        results = train_fn(
            self.model,
            self.state,
            X_tr,
            y_tr,
            X_val,
            y_val,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            lambda_spec=self.lambda_spec,
            key=train_key,
            augment_fn=self.augment_fn,
            loss_fn=loss_fn,
            return_penalty_history=self.return_penalty_history,
        )

        # 6) unpack results
        if self.return_penalty_history:
            final_model, final_state, tr, va, pen = results  # type: ignore
            self.penalty_history_ = pen
        else:
            final_model, final_state, tr, va = results  # type: ignore

        # 7) stash final model & losses
        self.model, self.state = final_model, final_state
        self.train_loss_ = tr
        self.val_loss_ = va

        return self


class EQXRegressorBacktrack(EQXBaseBacktrack, RegressorMixin):
    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._fit(X, y_arr, train_fn=train_backtrack, loss_fn=regression_loss)

    def predict(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        preds = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        return np.array(preds).reshape(-1)


class EQXBinaryClassifierBacktrack(EQXBaseBacktrack, ClassifierMixin):
    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._fit(X, y_arr, train_fn=train_backtrack, loss_fn=binary_loss)

    def predict_proba(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        logits = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        logits = np.array(logits).reshape(-1)
        pos = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - pos, pos]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EQXMulticlassClassifierBacktrack(EQXBaseBacktrack, ClassifierMixin):
    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
        self.classes_ = np.unique(y_arr)
        return self._fit(X, y_arr, train_fn=train_backtrack, loss_fn=multiclass_loss)

    def predict_proba(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        logits = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        logits = np.array(logits)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
