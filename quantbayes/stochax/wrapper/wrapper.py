# quantbayes/stochax/wrapper/wrapper.py

from __future__ import annotations

import math
import pathlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Literal

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# —— Delegate everything heavy to your SOTA trainer/inference layer ———————
from quantbayes.stochax.trainer.train import (
    train as _train,
    train_sam as _train_sam,
    train_on_full_data as _train_full,
    train_on_full_data_sam as _train_full_sam,
    predict as _predict_single,  # non-batched
    predict_batched as _predict_batched,  # jitted batched
    predict_batched_efficient as _predict_eff,  # streaming/memory-safe
    regression_loss as _regression_loss,
    binary_loss as _binary_loss,
    multiclass_loss as _multiclass_loss,
)
from quantbayes.stochax.utils import EMA as _EMA, swap_ema_params as _swap_ema

# Types
Array = jnp.ndarray
Key = jr.PRNGKey
AugmentFn = Callable[[Key, Array], Tuple[Array, Array]] | Callable[[Key, Array], Array]

__all__ = [
    "EQXRegressor",
    "EQXBinaryClassifier",
    "EQXMulticlassClassifier",
]


# ===============================================================
# Internal helpers
# ===============================================================


def _np_sigmoid_stable(z: np.ndarray) -> np.ndarray:
    # numerically-stable sigmoid
    # clip to avoid overflow in exp for very large |z|
    zc = np.clip(z, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _np_softmax_stable(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _as_jnp(x: np.ndarray | Array, dtype=jnp.float32) -> Array:
    # Convert to JAX array with a standard dtype
    if isinstance(x, jnp.ndarray):
        return x.astype(dtype)
    return jnp.array(np.asarray(x), dtype=dtype)


def _rng_from_seed(seed: Optional[int]) -> jr.key:
    if seed is None:
        seed = 0
    return jr.PRNGKey(int(seed))


def _holdout_split(
    X: Array, y: Array, val_frac: float, seed: int
) -> Tuple[Array, Array, Array, Array]:
    n = X.shape[0]
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    rs.shuffle(idx)
    cut = int(max(1, round((1.0 - val_frac) * n)))
    tr_idx = idx[:cut]
    va_idx = idx[cut:] if cut < n else idx[:1]  # ensure non-empty
    return X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]


# ===============================================================
# Base wrapper
# ===============================================================


class EQXBaseEstimator(BaseEstimator):
    """
    Base sklearn-compatible wrapper for Equinox models.

    Assumptions / conventions
    -------------------------
    - `model_cls` must have signature `(key, **model_kwargs) -> eqx.Module`
      and its `__call__(x, key, state) -> (out, new_state)` returns either:
         * regression: out with shape (B, ...) (usually (B, 1) or (B,))
         * binary: out with shape (B, 1) or (B,)
         * multiclass: out with shape (B, C)
    - Training & inference are delegated to quantbayes.stochax.trainer.train.
    - Designed for 2D X: shape (N, D). Keep vision wrappers separate.
    """

    # NOTE: Do NOT do any work in __init__. Keep arguments as bare attributes
    # so sklearn.clone works without surprises.
    def __init__(
        self,
        *,
        model_cls: Callable[..., eqx.Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,  # sklearn-style alias for PRNG seed
        batch_size: int = 128,
        num_epochs: int = 200,
        patience: int = 20,
        val_frac: float = 0.1,
        # regularization
        lambda_spec: float = 0.0,
        lambda_frob: float = 0.0,
        # optimizer defaults (used if `optimizer` is None)
        peak_lr: float = 1e-3,
        end_lr: float = 1e-5,
        warmup_steps: Optional[int] = None,  # if None: ≈ 1 epoch or 100 steps
        weight_decay: float = 1e-4,
        grad_clip_norm: Optional[float] = 1.0,
        optimizer: Optional[optax.GradientTransformation] = None,
        # data augmentation (on-device Augmax etc.)
        augment_fn: Optional[AugmentFn] = None,
        # EMA knobs
        use_ema: bool = False,
        ema_decay: float = 0.999,
        eval_with_ema: bool = True,
        return_ema: bool = False,
        # SAM / ASAM knobs
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_mode: Literal["sam", "asam"] = "sam",
        asam_eps: float = 1e-12,
        require_no_batchnorm: bool = True,
        freeze_norm_on_perturbed: bool = True,
        # checkpoints
        ckpt_path: Optional[str] = None,  # e.g. "ckpts/run-epoch-{epoch:03d}.ckpt"
        checkpoint_interval: Optional[int] = None,  # int (epochs)
        # refit after early stop on full data for the best epoch length
        refit_on_full_data: bool = False,
        # inference
        predict_batch_size: int = 1024,
        # loss (must be set in subclass default or explicitly by user)
        loss_fn: Optional[Callable] = None,
        specpen_recorder: Optional[Callable[[Dict[str, Any]], None]] = None,
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

        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.optimizer = optimizer

        self.augment_fn = augment_fn

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.eval_with_ema = eval_with_ema
        self.return_ema = return_ema

        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.sam_mode = sam_mode
        self.asam_eps = asam_eps
        self.require_no_batchnorm = require_no_batchnorm
        self.freeze_norm_on_perturbed = freeze_norm_on_perturbed

        self.ckpt_path = ckpt_path
        self.checkpoint_interval = checkpoint_interval

        self.refit_on_full_data = refit_on_full_data
        self.predict_batch_size = predict_batch_size

        self.loss_fn = loss_fn
        self.specpen_recorder = specpen_recorder

        # fitted attrs (set in fit)
        # self.model_
        # self.state_
        # self.predict_model_
        # self.n_features_in_
        # self.train_losses_
        # self.val_losses_
        # self.penalty_history_
        # self.best_epoch_
        # self.ema_

    # ---------------------------
    # sklearn compatibility tags
    # ---------------------------
    def _more_tags(self) -> Dict[str, Any]:
        # avoid scikit-learn trying to validate shapes we don't promise beyond 2D arrays
        # (we still expect 2D X here, but make randomness explicit)
        return {
            "non_deterministic": self.augment_fn is not None,
            "requires_y": True,
            # Keep default validation (2darray) for these wrappers.
        }

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "is_fitted_") and bool(self.is_fitted_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        import inspect

        # Use the base class __init__ signature as the source of truth
        sig = inspect.signature(EQXBaseEstimator.__init__)
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

    # ---------------------------
    # internal utils
    # ---------------------------
    def _prepare_xy(
        self, X: np.ndarray | Array, y: np.ndarray | Array
    ) -> Tuple[Array, Array]:
        # Subclasses override y dtype handling when needed
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_jax = _as_jnp(y, dtype=jnp.float32)
        return X_jax, y_jax

    def _set_n_features(self, X: Array) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected X as (N, D). Got shape {tuple(X.shape)}")
        self.n_features_in_ = X.shape[1]

    def _init_model(self, init_key: jr.key) -> Tuple[eqx.Module, Any]:
        kwargs = self.model_kwargs or {}
        model, state = eqx.nn.make_with_state(self.model_cls)(init_key, **kwargs)
        return model, state

    def _build_default_optimizer(
        self, n_train: int, batch_size: int
    ) -> optax.GradientTransformation:
        steps_per_epoch = max(1, math.ceil(n_train / batch_size))
        total_steps = max(1, self.num_epochs * steps_per_epoch)
        warmup = self.warmup_steps
        if warmup is None:
            warmup = max(100, steps_per_epoch)  # ≈ one epoch or 100 steps

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(self.peak_lr),
            warmup_steps=int(warmup),
            decay_steps=max(1, total_steps - int(warmup)),
            end_value=float(self.end_lr),
        )

        pieces = []
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            pieces.append(optax.clip_by_global_norm(self.grad_clip_norm))
        # AdamW is a robust default; swap to Adan if you prefer
        pieces.append(
            optax.adamw(learning_rate=lr_schedule, weight_decay=self.weight_decay)
        )
        return optax.chain(*pieces)

    def _choose_train_fn(self):
        if self.use_sam:
            return _train_sam
        return _train

    def _choose_train_full_fn(self):
        if self.use_sam:
            return _train_full_sam
        return _train_full

    def _finalize_predict_model(self, model: eqx.Module) -> eqx.Module:
        # Use EMA weights for inference if requested and available
        m = model
        if getattr(self, "ema_", None) is not None and self.eval_with_ema:
            m = _swap_ema(m, self.ema_)
        return eqx.nn.inference_mode(m)

    # ---------------------------
    # core API
    # ---------------------------
    def fit(self, X: np.ndarray | Array, y: np.ndarray | Array):
        # 1) prepare data
        X_jax, y_jax = self._prepare_xy(X, y)
        self._set_n_features(X_jax)

        # 2) split
        seed = 0 if self.random_state is None else int(self.random_state)
        X_tr, X_val, y_tr, y_val = _holdout_split(X_jax, y_jax, self.val_frac, seed)

        # 3) PRNG and init
        master = _rng_from_seed(self.random_state)
        master, init_key, train_key = jr.split(master, 3)
        model, state = self._init_model(init_key)

        # 4) optimizer
        if self.optimizer is None:
            optimizer = self._build_default_optimizer(
                int(X_tr.shape[0]), self.batch_size
            )
        else:
            optimizer = self.optimizer
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # 5) train (early stopping on the true hold-out)
        train_fn = self._choose_train_fn()

        # SAM-only kwargs (avoid passing them to the plain trainer)
        sam_kwargs = {}
        if self.use_sam:
            sam_kwargs = {
                "require_no_batchnorm": self.require_no_batchnorm,
                "freeze_norm_on_perturbed": self.freeze_norm_on_perturbed,
                "sam_rho": self.sam_rho,
                "sam_mode": self.sam_mode,
                "asam_eps": self.asam_eps,
            }

        # We want histories + optionally EMA object
        want_pen = True
        want_ema = bool(self.return_ema and self.use_ema)

        out = train_fn(
            model=model,
            state=state,
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
            augment_fn=self.augment_fn,
            lambda_spec=self.lambda_spec,
            lambda_frob=self.lambda_frob,
            ckpt_path=self.ckpt_path,
            checkpoint_interval=self.checkpoint_interval,
            return_penalty_history=want_pen,
            # EMA
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            eval_with_ema=self.eval_with_ema,
            return_ema=want_ema,
            specpen_recorder=self.specpen_recorder,
            # SAM-only knobs
            **sam_kwargs,
        )

        # Unpack variant returns
        # Base: (best_model, best_state, train_losses, val_losses)
        # +pen  : (..., penalty_history)
        # +ema  : (..., EMA)
        # +both : (..., penalty_history, EMA)
        if want_pen and want_ema:
            best_model, best_state, tr_hist, va_hist, pen_hist, ema_obj = out
            self.ema_ = ema_obj
            self.penalty_history_ = pen_hist
        elif want_pen and not want_ema:
            best_model, best_state, tr_hist, va_hist, pen_hist = out
            self.ema_ = None
            self.penalty_history_ = pen_hist
        elif (not want_pen) and want_ema:
            best_model, best_state, tr_hist, va_hist, ema_obj = out
            self.ema_ = ema_obj
            self.penalty_history_ = None
        else:
            best_model, best_state, tr_hist, va_hist = out
            self.ema_ = None
            self.penalty_history_ = None

        self.train_losses_ = list(map(float, tr_hist))
        self.val_losses_ = list(map(float, va_hist))
        self.best_epoch_ = (
            int(np.argmin(self.val_losses_) + 1) if len(self.val_losses_) else 1
        )

        # 6) optionally refit on full data for best epoch length (fresh run)
        if self.refit_on_full_data:
            full_fn = self._choose_train_full_fn()
            # Recompute optimizer for full dataset (same schedule length as requested best epochs)
            if self.optimizer is None:
                optimizer = self._build_default_optimizer(
                    int(X_jax.shape[0]), self.batch_size
                )
            else:
                optimizer = self.optimizer
            opt_state = optimizer.init(eqx.filter(best_model, eqx.is_inexact_array))

            master, rf_key = jr.split(master)
            # Train for `best_epoch_` epochs, no early stopping
            sam_kwargs = {}
            if self.use_sam:
                sam_kwargs = {
                    "require_no_batchnorm": self.require_no_batchnorm,
                    "freeze_norm_on_perturbed": self.freeze_norm_on_perturbed,
                    "sam_rho": self.sam_rho,
                    "sam_mode": self.sam_mode,
                    "asam_eps": self.asam_eps,
                }

            out_full = full_fn(
                model=best_model,
                state=best_state,
                opt_state=opt_state,
                optimizer=optimizer,
                loss_fn=self.loss_fn,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_val,  # ignored internally
                y_val=y_val,  # ignored internally
                batch_size=self.batch_size,
                num_epochs=self.best_epoch_,
                key=rf_key,
                augment_fn=self.augment_fn,
                lambda_spec=self.lambda_spec,
                lambda_frob=self.lambda_frob,
                ckpt_path=self.ckpt_path,
                checkpoint_interval=self.checkpoint_interval,
                return_penalty_history=False,
                # EMA
                use_ema=self.use_ema,
                ema_decay=self.ema_decay,
                eval_with_ema=self.eval_with_ema,
                return_ema=False,
                specpen_recorder=self.specpen_recorder,
                # SAM-only knobs
                **sam_kwargs,
            )
            best_model, best_state, _, _ = out_full

        # 7) freeze for inference and stash
        self.model_ = best_model
        self.state_ = best_state
        self.predict_model_ = self._finalize_predict_model(best_model)

        self.is_fitted_ = True
        return self

    # Common low-level inference returning raw model outputs/logits
    def _predict_logits(self, X: np.ndarray | Array) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("Estimator is not fitted yet.")
        X_jax = _as_jnp(X, dtype=jnp.float32)
        # memory-safe batched prediction
        key = _rng_from_seed(self.random_state)
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

        # training weights + state
        eqx.tree_serialise_leaves(prefix + ".model.eqx", self.model_)
        eqx.tree_serialise_leaves(prefix + ".state.eqx", self.state_)

        # if you validated with EMA, also store the exact inference weights you used
        if getattr(self, "ema_", None) is not None and self.eval_with_ema:
            inf_model = self._finalize_predict_model(self.model_)
            eqx.tree_serialise_leaves(prefix + ".predict.eqx", inf_model)

        # tiny metadata
        meta = {
            "n_features_in_": getattr(self, "n_features_in_", None),
            "classes_": (self.classes_.tolist() if hasattr(self, "classes_") else None),
            "eval_with_ema": bool(self.eval_with_ema),
        }
        with open(prefix + ".meta.json", "w") as f:
            json.dump(meta, f)

    def load_ckpt(self, prefix: str):
        import json, equinox as eqx, jax.random as jr, numpy as np

        key = _rng_from_seed(self.random_state)

        # templates must match the trained structure
        model_t, state_t = self._init_model(key)
        self.model_ = eqx.tree_deserialise_leaves(prefix + ".model.eqx", model_t)
        self.state_ = eqx.tree_deserialise_leaves(prefix + ".state.eqx", state_t)

        # prefer exact inference weights if present; otherwise finalize on the fly
        try:
            self.predict_model_ = eqx.tree_deserialise_leaves(
                prefix + ".predict.eqx", model_t
            )
        except FileNotFoundError:
            self.predict_model_ = self._finalize_predict_model(self.model_)

        # restore metadata (classes_ is crucial for multiclass)
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


# ===============================================================
# Regressor
# ===============================================================


class EQXRegressor(EQXBaseEstimator, RegressorMixin):
    """
    Equinox -> sklearn wrapper for regression nets.
    """

    def __init__(self, **kwargs):
        # default loss if user didn't pass one
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _regression_loss
        super().__init__(**kwargs)

    def predict(self, X: np.ndarray | Array) -> np.ndarray:
        z = self._predict_logits(X)
        # Flatten to (N,) if (N,1)
        if z.ndim >= 2 and z.shape[-1] == 1:
            z = z.reshape(z.shape[0])
        return z


# ===============================================================
# Binary classifier
# ===============================================================


class EQXBinaryClassifier(EQXBaseEstimator, ClassifierMixin):
    """
    Equinox -> sklearn wrapper for binary classifiers (single-logit).
    Produces `classes_` which preserves the original class labels.
    """

    def __init__(self, **kwargs):
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _binary_loss
        super().__init__(**kwargs)

    def _prepare_xy(
        self, X: np.ndarray | Array, y: np.ndarray | Array
    ) -> Tuple[Array, Array]:
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_np = np.asarray(y).reshape(-1)

        if y_np.dtype == bool:
            uniques = np.array([False, True])
        else:
            uniques = np.unique(y_np)
            if len(uniques) != 2:
                raise ValueError(
                    f"BinaryClassifier expects exactly 2 classes; got {uniques}."
                )
        self.classes_ = uniques

        mapping = {uniques[0]: 0, uniques[1]: 1}
        y01 = np.vectorize(mapping.get, otypes=[np.int32])(y_np)
        y_jax = _as_jnp(y01, dtype=jnp.float32)  # BCE expects float
        return X_jax, y_jax

    def decision_function(self, X: np.ndarray | Array) -> np.ndarray:
        z = self._predict_logits(X)
        # ensure (N,)
        if z.ndim >= 2 and z.shape[-1] == 1:
            z = z.reshape(z.shape[0])
        return z

    def predict_proba(self, X: np.ndarray | Array) -> np.ndarray:
        z = self.decision_function(X)  # (N,)
        p1 = _np_sigmoid_stable(z)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray | Array) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


# ===============================================================
# Multiclass classifier
# ===============================================================


class EQXMulticlassClassifier(EQXBaseEstimator, ClassifierMixin):
    """
    Equinox -> sklearn wrapper for multiclass classifiers (C >= 2).
    """

    def __init__(self, **kwargs):
        if "loss_fn" not in kwargs or kwargs["loss_fn"] is None:
            kwargs["loss_fn"] = _multiclass_loss
        super().__init__(**kwargs)

    def _prepare_xy(
        self, X: np.ndarray | Array, y: np.ndarray | Array
    ) -> Tuple[Array, Array]:
        X_jax = _as_jnp(X, dtype=jnp.float32)
        y_np = np.asarray(y, dtype=int).reshape(-1)
        self.classes_ = np.unique(y_np)
        # Ensure labels are 0..(C-1) for the integer-label cross-entropy
        # Build mapping from actual labels -> 0..C-1
        mapping = {lab: i for i, lab in enumerate(self.classes_)}
        y_idx = np.vectorize(mapping.get, otypes=[np.int32])(y_np)
        y_jax = _as_jnp(y_idx, dtype=jnp.int32)
        return X_jax, y_jax

    def decision_function(self, X: np.ndarray | Array) -> np.ndarray:
        # Return raw logits (N, C)
        z = self._predict_logits(X)
        # make sure 2D shape
        if z.ndim == 1:
            z = z[:, None]
        return z

    def predict_proba(self, X: np.ndarray | Array) -> np.ndarray:
        logits = self.decision_function(X)
        return _np_softmax_stable(logits, axis=1)

    def predict(self, X: np.ndarray | Array) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
