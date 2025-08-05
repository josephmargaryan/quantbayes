import jax
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional, Callable, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from quantbayes.stochax import train as nn_train, predict as nn_predict
from quantbayes.stochax import regression_loss, binary_loss, multiclass_loss

# Type for an on‑device augmentation fn
AugmentFn = Callable[[jr.key, jnp.ndarray], jnp.ndarray]

__all__ = ["EQXRegressor", "EQXMulticlassClassifier", "EQXBinaryClassifier"]


class EQXBase(BaseEstimator):
    """
    Shared base for Equinox wrappers (regression, binary, multiclass),
    with proper hold-out validation for genuine early stopping.
    """

    def __init__(
        self,
        model_cls: Callable[..., eqx.Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        key_seed: int = 0,
        batch_size: int = 128,
        num_epochs: int = 200,
        patience: int = 20,
        val_frac: float = 0.1,
        init_lr: float = 1e-3,
        end_lr: float = 1e-4,
        lr_decay_steps: int = 500,
        weight_decay: float = 1e-4,
        loss_fn: Callable = None,
        lambda_spec: float = 0.0,
        optimizer: Optional[optax.GradientTransformation] = None,
        augment_fn: Optional[AugmentFn] = None,
    ):
        # store exactly what the user passed
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.key_seed = key_seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.lr_decay_steps = lr_decay_steps
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.lambda_spec = lambda_spec
        self.optimizer = optimizer
        self.augment_fn = augment_fn

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
            "init_lr": self.init_lr,
            "end_lr": self.end_lr,
            "lr_decay_steps": self.lr_decay_steps,
            "weight_decay": self.weight_decay,
            "loss_fn": self.loss_fn,
            "lambda_spec": self.lambda_spec,
            "optimizer": self.optimizer,
            "augment_fn": self.augment_fn,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        # 1) Numpy → JAX arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

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

        # 3) PRNG
        master_key = jr.PRNGKey(self.key_seed)
        master_key, init_key, train_key = jr.split(master_key, 3)

        # 4) init model + state (treat None→{} here)
        mkw = self.model_kwargs or {}
        self.model, self.state = eqx.nn.make_with_state(self.model_cls)(init_key, **mkw)

        # 5) choose optimizer
        if self.optimizer is None:
            lr_schedule = optax.linear_schedule(
                init_value=self.init_lr,
                end_value=self.end_lr,
                transition_steps=self.lr_decay_steps,
            )
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay),
            )
        else:
            optimizer = self.optimizer

        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        # 6) train
        best_model, best_state, _, _ = nn_train(
            model=self.model,
            state=self.state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            key=train_key,
            lambda_spec=self.lambda_spec,
            augment_fn=self.augment_fn,
        )

        # 7) switch to inference mode
        self.model, self.state = (
            eqx.nn.inference_mode(best_model),
            best_state,
        )
        return self


class EQXRegressor(EQXBase, RegressorMixin):
    """Wrapper for Equinox regression nets."""

    def predict(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        preds = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        return np.array(preds).reshape(-1)


class EQXBinaryClassifier(EQXBase, ClassifierMixin):
    """Wrapper for Equinox binary classifiers (single‑logit)."""

    def predict_proba(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        logits = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        logits = np.array(logits).reshape(-1)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - probs_pos, probs_pos]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EQXMulticlassClassifier(EQXBase, ClassifierMixin):
    """Wrapper for Equinox multiclass classifiers."""

    def fit(self, X, y):
        # integer labels & stash classes_
        X_jax = jnp.array(X, dtype=jnp.float32)
        y_arr = np.array(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y_arr)
        y_jax = jnp.array(y_arr, dtype=jnp.int32)

        # PRNG + init
        mk = jr.PRNGKey(self.key_seed)
        mk, mkey, train_key = jr.split(mk, 3)
        self.model, self.state = eqx.nn.make_with_state(self.model_cls)(
            mkey, **self.model_kwargs
        )

        # choose optimizer (reuse same logic)
        if self.optimizer is None:
            lr_schedule = optax.linear_schedule(
                init_value=self.init_lr,
                end_value=self.end_lr,
                transition_steps=self.lr_decay_steps,
            )
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay),
            )
        else:
            optimizer = self.optimizer

        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        # train
        best_model, best_state, _, _ = nn_train(
            model=self.model,
            state=self.state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            X_train=X_jax,
            y_train=y_jax,
            X_val=X_jax,
            y_val=y_jax,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            key=train_key,
            lambda_spec=self.lambda_spec,
            augment_fn=self.augment_fn,
        )

        self.model, self.state = eqx.nn.inference_mode(best_model), best_state
        return self

    def predict_proba(self, X):
        X_jax = jnp.array(X, dtype=jnp.float32)
        logits = nn_predict(self.model, self.state, X_jax, jr.PRNGKey(self.key_seed))
        logits = np.array(logits)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class EQXImageClassifier(EQXBase, ClassifierMixin):
    """
    EQX wrapper for image‐classification nets.
    Expects:
      - X: np.ndarray, shape (N, C, H, W), dtype float32
      - y: np.ndarray, shape (N,), integer labels
    """

    def fit(self, X, y):
        # 1) to JAX arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).ravel()
        self.n_features_in_ = X.shape[1:]  # (C,H,W)

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)

        # 2) train/val split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_jax,
            y_jax,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.key_seed,
        )

        # 3) PRNG
        key = jr.PRNGKey(self.key_seed)

        # 4) init model+state
        init_key, train_key = jr.split(key)
        self.model, self.state = eqx.nn.make_with_state(self.model_cls)(
            init_key, **(self.model_kwargs or {})
        )

        # 5) optimizer
        if self.optimizer is None:
            optimizer = optax.adamw(self.init_lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer
        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        # 6) train
        best_model, best_state, _, _ = nn_train(
            model=self.model,
            state=self.state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=multiclass_loss,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            key=train_key,
            augment_fn=self.augment_fn,
        )

        # 7) finalize
        self.model, self.state = eqx.nn.inference_mode(best_model), best_state
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_jax = jnp.array(X)
        # batched predict over (N,C,H,W)
        logits = nn_predict(
            self.model,
            self.state,
            X_jax,
            jr.PRNGKey(self.key_seed),
            batch_size=self.batch_size,
        )
        probs = jax.nn.softmax(logits, axis=-1)
        return np.array(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=-1)]


class EQXImageSegmenter(EQXBase):
    """
    EQX wrapper for semantic‐segmentation nets.
    Expects:
      - X: np.ndarray, shape (N, C, H, W), dtype float32
      - y: np.ndarray, shape (N, H, W) or (N,1,H,W), integer masks
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        # flatten singleton channel
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        self.n_features_in_ = X.shape[1:]  # (C,H,W)

        X_jax = jnp.array(X)
        y_jax = jnp.array(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_jax,
            y_jax,
            test_size=self.val_frac,
            shuffle=True,
            random_state=self.key_seed,
        )

        key = jr.PRNGKey(self.key_seed)
        init_key, train_key = jr.split(key)
        self.model, self.state = eqx.nn.make_with_state(self.model_cls)(
            init_key, **(self.model_kwargs or {})
        )

        if self.optimizer is None:
            optimizer = optax.adamw(self.init_lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer
        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        best_model, best_state, _, _ = nn_train(
            model=self.model,
            state=self.state,
            opt_state=opt_state,
            optimizer=optimizer,
            loss_fn=multiclass_loss,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            patience=self.patience,
            key=train_key,
            augment_fn=self.augment_fn,
        )

        self.model, self.state = eqx.nn.inference_mode(best_model), best_state
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_jax = jnp.array(X)
        logits = nn_predict(
            self.model,
            self.state,
            X_jax,
            jr.PRNGKey(self.key_seed),
            batch_size=self.batch_size,
        )
        # logits shape (N, H, W, n_classes) or (N, n_classes, H, W) depending on your model
        # assume model outputs (N, H, W, C): transpose to (N, C, H, W)
        if logits.ndim == 4 and logits.shape[-1] != self.model_kwargs.get("classes", 1):
            # assume channel-last
            logits = logits.transpose(0, 3, 1, 2)
        probs = jax.nn.softmax(logits, axis=1)
        return np.array(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)  # (N, H, W)
