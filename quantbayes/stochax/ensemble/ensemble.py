# quantbayes/stochax/ensemble/ensemble.py

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pathlib

# Optional: weight optimization for weighted-averaging (falls back to uniform if unavailable)
try:
    from scipy.optimize import minimize as _scipy_minimize

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# Plug into your training utilities
from quantbayes.stochax.trainer.train import (
    train,  # expected to return (best_model, best_state, train_hist, val_hist)
    predict,  # batches internally, returns raw outputs/logits
    binary_loss,
    multiclass_loss,
    regression_loss,
)

__all__ = [
    "EquinoxEnsembleBase",
    "EquinoxEnsembleBinary",
    "EquinoxEnsembleMulticlass",
    "EquinoxEnsembleRegression",
]

EPS = 1e-12


# ------------------------------ helpers ---------------------------------


def _init_member(
    constructor: Callable[[jr.PRNGKey], Union[eqx.Module, Tuple[eqx.Module, Any]]],
    key: jr.PRNGKey,
) -> Tuple[eqx.Module, Any]:
    """Allow constructors that return `model` or `(model, state)`."""
    out = constructor(key)
    if isinstance(out, tuple) and len(out) == 2:
        model, state = out
    else:
        model, state = out, None
    return model, state


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log(1 - p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _geom_mean_binary(pos_cols: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Proper normalized geometric mean for binary:
      p = geom_mean(p_k), q = geom_mean(1 - p_k), return p/(p+q).
    pos_cols: (N, M), weights: (M,)
    """
    P = np.clip(pos_cols, EPS, 1 - EPS)
    w = _normalize_weights(weights)
    log_p = (np.log(P) * w).sum(axis=1)
    log_q = (np.log(1.0 - P) * w).sum(axis=1)
    g_p = np.exp(log_p)
    g_q = np.exp(log_q)
    return g_p / (g_p + g_q)


def _geom_mean_multiclass(stack_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Normalized geometric mean across models for multiclass.
    stack_probs: (M, N, C)  weights: (M,)
    returns: (N, C)
    """
    A = np.clip(stack_probs, EPS, 1.0)
    w = _normalize_weights(weights)
    # tensordot over models: sum_k w_k * log p_k
    logp = np.tensordot(w, np.log(A), axes=1)  # (N, C)
    G = np.exp(logp)
    G /= G.sum(axis=1, keepdims=True)
    return G


def _maybe_minimize_weights_classification(
    list_of_probs: List[np.ndarray],
    y_val: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """
    Optimize nonnegative, sum-to-one weights to minimize validation log loss.
    list_of_probs: [ (N,C), ... ]
    """
    if not _HAS_SCIPY:
        return np.ones(len(list_of_probs)) / len(list_of_probs)

    P = np.stack(list_of_probs, axis=-1)  # (N, C, M)
    N, C, M = P.shape
    # encode y as indices 0..C-1 in `classes` order
    y_idx = np.searchsorted(classes, y_val)

    def negloglik(w):
        w = np.asarray(w)
        if np.any(w < -1e-12):
            return 1e9
        w = _normalize_weights(w)
        Pw = np.tensordot(P, w, axes=([2], [0]))  # (N, C)
        Pw = np.clip(Pw, EPS, 1.0)
        return -np.mean(np.log(Pw[np.arange(N), y_idx]))

    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w},  # w >= 0
    )
    w0 = np.ones(M) / M
    res = _scipy_minimize(negloglik, w0, method="SLSQP", constraints=cons)
    if not (res and res.success):
        return w0
    return _normalize_weights(res.x)


def _maybe_minimize_weights_regression(
    list_of_preds: List[np.ndarray],
    y_val: np.ndarray,
) -> np.ndarray:
    """Optimize nonnegative, sum-to-one weights to minimize MSE on validation."""
    if not _HAS_SCIPY:
        return np.ones(len(list_of_preds)) / len(list_of_preds)

    P = np.stack([p.reshape(-1) for p in list_of_preds], axis=1)  # (N, M)
    N, M = P.shape
    y = y_val.reshape(-1)

    def mse(w):
        w = np.asarray(w)
        if np.any(w < -1e-12):
            return 1e9
        w = _normalize_weights(w)
        pred = P @ w
        return np.mean((pred - y) ** 2)

    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w},
    )
    w0 = np.ones(M) / M
    res = _scipy_minimize(mse, w0, method="SLSQP", constraints=cons)
    if not (res and res.success):
        return w0
    return _normalize_weights(res.x)


# ------------------------------ base class ---------------------------------


class EquinoxEnsembleBase:
    """
    JAX/Equinox-native ensemble trainer.

    Works with Equinox models and Optax optimizers directly. Supports:
      * weighted-averaging and stacking
      * constructors returning model or (model, state)
      * per-member optimizer or one shared optimizer
      * JIT-cached inference
      * TTA at inference (mean aggregation)
    """

    def __init__(
        self,
        model_constructors: Sequence[
            Callable[[jr.PRNGKey], Union[eqx.Module, Tuple[eqx.Module, Any]]]
        ],
        *,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[
            optax.GradientTransformation, Sequence[optax.GradientTransformation]
        ],
        ensemble_method: str = "weighted_average",  # "weighted_average" | "stacking"
        weights: Optional[Sequence[float]] = None,  # for weighted_average
        meta_learner: Optional[Any] = None,  # sklearn-like estimator (for stacking)
        # TTA
        tta_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
        tta_samples: int = 0,
        # Inference
        jit_infer: bool = True,
        # Weight optimization on the validation set (only weighted_average)
        optimize_weights: bool = False,
        random_state: Optional[int] = None,
    ):
        if ensemble_method not in {"weighted_average", "stacking"}:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")

        self.model_constructors = list(model_constructors)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        self.tta_fn = tta_fn
        self.tta_samples = int(tta_samples) if tta_samples else 0
        self.jit_infer = bool(jit_infer)
        self.optimize_weights = bool(optimize_weights)
        self.random_state = random_state

        # training artifacts
        self.members_: List[Tuple[eqx.Module, Any]] = []  # (model, state)
        self.opt_states_: List[Any] = []
        self._member_predict_fns_: List[Callable[[jnp.ndarray, jr.PRNGKey], Any]] = []

        # weights (for weighted_average)
        if weights is not None and ensemble_method == "weighted_average":
            if len(weights) != len(self.model_constructors):
                raise ValueError(
                    "weights length must match number of model constructors"
                )
            self.weights_ = _normalize_weights(np.asarray(weights, dtype=float))
        else:
            self.weights_ = None  # learned later (uniform or optimized)

        # rng
        base = int(jax.random.key_data(jr.PRNGKey(0))[0])
        seed = self.random_state if self.random_state is not None else base
        self._master_key = jr.PRNGKey(seed)

    # ------------------------- training & setup -------------------------

    def _broadcast_optimizers(self) -> List[optax.GradientTransformation]:
        if isinstance(self.optimizer, (list, tuple)):
            if len(self.optimizer) != len(self.model_constructors):
                raise ValueError(
                    "optimizer list length must match number of constructors"
                )
            return list(self.optimizer)
        return [self.optimizer] * len(self.model_constructors)

    def _cache_predict_fn(self, model: eqx.Module, state: Any):
        # Bind model/state; expose signature (X, key) -> outputs
        fn = lambda X, key, m=model, s=state: predict(m, s, X, key)
        if self.jit_infer:
            fn = eqx.filter_jit(fn)
        return fn

    def fit(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        y_val: jnp.ndarray,
        *,
        batch_size: int = 512,
        num_epochs: int = 50,
        patience: int = 5,
        key: Optional[jr.PRNGKey] = None,
        # passthrough to train() if your trainer supports them
        ckpt_path: Optional[Union[str, "pathlib.Path"]] = None,
        checkpoint_interval: Optional[int] = None,
        # penalties if your trainer supports them (lbfgs/zoom/backtrack variants ignore)
        lambda_spec: float = 0.0,
        lambda_frob: float = 0.0,
        loss_fn: Optional[Callable] = None,  # override per-call if desired
    ):
        """
        Train each member on (X_train, y_train) with early-stopping on (X_val, y_val).
        For stacking, train the meta-learner on validation meta-features.
        """
        self.task_ = self._infer_task(y_train)
        self.classes_ = (
            np.unique(np.asarray(y_train)) if self.task_ != "regression" else None
        )
        self.n_members_ = len(self.model_constructors)

        # choose loss
        self.loss_fn = (
            loss_fn
            or self.loss_fn
            or {
                "binary": binary_loss,
                "multiclass": multiclass_loss,
                "regression": regression_loss,
            }[self.task_]
        )

        # member training
        self.members_.clear()
        self._member_predict_fns_.clear()
        opt_list = self._broadcast_optimizers()

        # deterministic split of master key
        master = key if key is not None else self._master_key
        member_keys = jr.split(master, self.n_members_ + 1)
        self._master_key = member_keys[0]  # keep a fresh one
        member_keys = member_keys[1:]

        for i, (constructor, opt_tx, mkey) in enumerate(
            zip(self.model_constructors, opt_list, member_keys)
        ):
            model, state = _init_member(constructor, mkey)
            opt_state = opt_tx.init(eqx.filter(model, eqx.is_inexact_array))

            best_model, best_state, _, _ = train(
                model=model,
                state=state,
                opt_state=opt_state,
                optimizer=opt_tx,
                loss_fn=self.loss_fn,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=batch_size,
                num_epochs=num_epochs,
                patience=patience,
                key=mkey,
                # passthroughs if your train() supports them
                ckpt_path=(
                    None if ckpt_path is None else str(ckpt_path).format(model=i)
                ),
                checkpoint_interval=checkpoint_interval,
                lambda_spec=lambda_spec,
                lambda_frob=lambda_frob,
            )
            self.members_.append((best_model, best_state))
            self._member_predict_fns_.append(
                self._cache_predict_fn(best_model, best_state)
            )

        # stacking path
        if self.ensemble_method == "stacking":
            self._fit_meta(X_val, y_val)
        else:
            # weighted-average path: if no weights provided and optimize requested, learn on validation preds
            if self.weights_ is None:
                self.weights_ = self._learn_weights_on_val(X_val, y_val)
            else:
                self.weights_ = _normalize_weights(self.weights_)

        return self

    # ------------------------- abstract-ish API -------------------------

    def _infer_task(self, y: jnp.ndarray) -> str:
        y_np = np.asarray(y)
        return (
            "regression"
            if np.issubdtype(y_np.dtype, np.floating) and len(np.unique(y_np)) > 20
            else ("binary" if len(np.unique(y_np)) == 2 else "multiclass")
        )

    # These must be implemented by subclasses:
    def _fit_meta(
        self, X_val: jnp.ndarray, y_val: jnp.ndarray
    ):  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None):
        raise NotImplementedError

    def predict_proba(self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None):
        raise NotImplementedError

    # ------------------------- inference utilities -------------------------

    def _run_member(self, idx: int, X: jnp.ndarray, key: jr.PRNGKey):
        """Forward one member with optional TTA; returns raw model outputs."""
        fn = self._member_predict_fns_[idx]
        if self.tta_fn is None or self.tta_samples <= 0:
            return fn(X, key)

        # mean aggregation over TTA samples in the *probability* domain for classification,
        # and raw outputs for regression (subclasses post-process appropriately).
        outs = []
        subkeys = jr.split(key, self.tta_samples)
        for sk in subkeys:
            X_aug = self.tta_fn(sk, X)
            outs.append(fn(X_aug, sk))
        # shape handling done in subclasses
        return outs

    def _per_member_probs(
        self,
        X: jnp.ndarray,
        key: Optional[jr.PRNGKey],
        postproc: Callable[[np.ndarray], np.ndarray],
    ) -> List[np.ndarray]:
        """
        Run all members and return a list of probability arrays per member.
        `postproc` converts raw outputs/logits → probabilities for each member.
        Handles TTA by averaging in probability space.
        """
        master = key if key is not None else self._master_key
        keys = jr.split(master, self.n_members_ + 1)
        self._master_key = keys[0]
        keys = keys[1:]

        out_probs: List[np.ndarray] = []
        for i, k in enumerate(keys):
            out = self._run_member(i, X, k)
            if isinstance(out, list):  # TTA: list of raw outputs
                probs = [postproc(np.array(o)) for o in out]
                out_probs.append(np.stack(probs, axis=0).mean(axis=0))
            else:
                out_probs.append(postproc(np.array(out)))
        return out_probs

    # for weighted-averaging
    def _learn_weights_on_val(
        self, X_val: jnp.ndarray, y_val: jnp.ndarray
    ) -> np.ndarray:
        if self.task_ == "regression":
            preds = self._per_member_regression_preds(X_val, key=None)
            return _maybe_minimize_weights_regression(preds, np.asarray(y_val))
        else:
            probs = self._per_member_class_probs(X_val, key=None)
            return _maybe_minimize_weights_classification(
                probs, np.asarray(y_val), self.classes_
            )

    # stubs the subclasses can use
    def _per_member_regression_preds(
        self, X: jnp.ndarray, key: Optional[jr.PRNGKey]
    ) -> List[np.ndarray]:
        raise NotImplementedError

    def _per_member_class_probs(
        self, X: jnp.ndarray, key: Optional[jr.PRNGKey]
    ) -> List[np.ndarray]:
        raise NotImplementedError


# ------------------------------ Binary ---------------------------------


class EquinoxEnsembleBinary(EquinoxEnsembleBase):
    """
    Binary classification ensemble.
    Weighted-average combiners: "proba" | "geom" | "logit"
    Stacking uses a scikit-learn classifier (default LogisticRegression).
    """

    def __init__(
        self,
        model_constructors: Sequence[
            Callable[[jr.PRNGKey], Union[eqx.Module, Tuple[eqx.Module, Any]]]
        ],
        *,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[
            optax.GradientTransformation, Sequence[optax.GradientTransformation]
        ],
        ensemble_method: str = "weighted_average",
        weights: Optional[Sequence[float]] = None,
        meta_learner: Optional[Any] = None,
        combine: str = "logit",  # "proba" | "geom" | "logit"
        pos_label: Optional[Union[int, float]] = None,
        tta_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
        tta_samples: int = 0,
        jit_infer: bool = True,
        optimize_weights: bool = False,
        random_state: Optional[int] = None,
        calibrate_meta: Optional[str] = None,  # None | "sigmoid" | "isotonic"
    ):
        super().__init__(
            model_constructors,
            loss_fn=loss_fn,
            optimizer=optimizer,
            ensemble_method=ensemble_method,
            weights=weights,
            meta_learner=meta_learner,
            tta_fn=tta_fn,
            tta_samples=tta_samples,
            jit_infer=jit_infer,
            optimize_weights=optimize_weights,
            random_state=random_state,
        )
        if combine not in {"proba", "geom", "logit"}:
            raise ValueError("combine must be one of {'proba','geom','logit'}")
        self.combine = combine
        self.pos_label = pos_label
        self.calibrate_meta = calibrate_meta

    # ---- meta-learner training ----

    def _fit_meta(self, X_val: jnp.ndarray, y_val: jnp.ndarray):
        # features = per-member positive-class probabilities on validation set
        features = np.column_stack(
            self._per_member_class_probs(X_val, key=None)
        )  # (N, M)
        y_val_np = np.asarray(y_val)

        meta = self.meta_learner
        if meta is None:
            from sklearn.linear_model import LogisticRegression

            meta = LogisticRegression(solver="lbfgs", max_iter=1000)

        if self.calibrate_meta:
            from sklearn.calibration import CalibratedClassifierCV

            meta = CalibratedClassifierCV(meta, method=self.calibrate_meta, cv=3)

        self.meta_ = meta.fit(features, y_val_np)

        # For completeness, learn uniform weights for fallback use (not used in stacking path)
        if self.weights_ is None:
            self.weights_ = np.ones(features.shape[1]) / features.shape[1]

        # establish class order
        self.classes_ = np.unique(y_val_np)
        self.pos_label_ = (
            self.pos_label if self.pos_label is not None else np.max(self.classes_)
        )
        self.neg_label_ = (
            self.classes_[0]
            if self.classes_[1] == self.pos_label_
            else self.classes_[1]
        )

    # ---- per-member outputs ----

    def _per_member_class_probs(
        self, X: jnp.ndarray, key: Optional[jr.PRNGKey]
    ) -> List[np.ndarray]:
        # Convert raw outputs to pos-class probabilities
        def to_pos_prob(raw: np.ndarray) -> np.ndarray:
            # raw expected shape (N, 1) logits or (N,) logits
            z = raw.reshape(-1)
            return _sigmoid(z)

        probs = self._per_member_probs(X, key, postproc=lambda r: to_pos_prob(r))
        # return list of (N,) arrays
        return probs

    def _per_member_regression_preds(self, X, key):  # not used here
        raise RuntimeError("Internal misuse: regression path in binary ensemble")

    # ---- predict / predict_proba ----

    def predict_proba(
        self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None
    ) -> np.ndarray:
        X = jnp.asarray(X)

        if self.ensemble_method == "stacking":
            feats = np.column_stack(self._per_member_class_probs(X, key))
            if hasattr(self.meta_, "predict_proba"):
                pos = self.meta_.predict_proba(feats)
                # meta classes can be arbitrary order – map to (neg, pos) using self.classes_
                pos_idx = int(
                    np.where(
                        self.meta_.classes_
                        == (self.pos_label or np.max(self.meta_.classes_))
                    )[0][0]
                )
                pos = pos[:, pos_idx]
            else:
                # fallback: use decision_function or predict and squash
                if hasattr(self.meta_, "decision_function"):
                    df = self.meta_.decision_function(feats)
                    pos = _sigmoid(np.asarray(df).reshape(-1))
                else:
                    pos = np.asarray(self.meta_.predict(feats)).astype(float)
            p_pos = np.clip(pos, 0.0, 1.0)

        else:
            cols = self._per_member_class_probs(X, key)
            P = np.column_stack(cols)  # (N, M)
            w = (
                self.weights_
                if self.weights_ is not None
                else np.ones(P.shape[1]) / P.shape[1]
            )
            w = _normalize_weights(np.asarray(w, dtype=float))

            if self.combine == "proba":
                p_pos = np.clip(P @ w, 0.0, 1.0)
            elif self.combine == "geom":
                p_pos = _geom_mean_binary(P, w)
            else:  # "logit"
                z = (_logit(P) * w).sum(axis=1)
                p_pos = _sigmoid(z)

        # assemble (N, 2) ordered by self.classes_
        classes = (
            self.classes_
            if hasattr(self, "classes_") and self.classes_ is not None
            else np.array([0, 1])
        )
        pos_label = self.pos_label if self.pos_label is not None else np.max(classes)
        neg_label = classes[0] if classes[1] == pos_label else classes[1]

        proba = np.zeros((len(p_pos), 2), dtype=float)
        # find column indices per class
        neg_idx = int(np.where(classes == neg_label)[0][0])
        pos_idx = int(np.where(classes == pos_label)[0][0])
        proba[:, pos_idx] = p_pos
        proba[:, neg_idx] = 1.0 - p_pos
        return proba

    def predict(self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None):
        proba = self.predict_proba(X, key=key)
        # pos column index according to self.classes_
        pos_label = (
            self.pos_label if self.pos_label is not None else np.max(self.classes_)
        )
        pos_idx = int(np.where(self.classes_ == pos_label)[0][0])
        return (proba[:, pos_idx] >= 0.5).astype(self.classes_.dtype)


# ------------------------------ Multiclass ---------------------------------


class EquinoxEnsembleMulticlass(EquinoxEnsembleBase):
    """
    Multiclass classification ensemble.
    Weighted-average combiners: "proba" | "geom"
    Stacking uses concatenated per-class probabilities from each member.
    """

    def __init__(
        self,
        model_constructors: Sequence[
            Callable[[jr.PRNGKey], Union[eqx.Module, Tuple[eqx.Module, Any]]]
        ],
        *,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[
            optax.GradientTransformation, Sequence[optax.GradientTransformation]
        ],
        ensemble_method: str = "weighted_average",
        weights: Optional[Sequence[float]] = None,
        meta_learner: Optional[Any] = None,
        combine: str = "proba",  # "proba" | "geom"
        tta_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
        tta_samples: int = 0,
        jit_infer: bool = True,
        optimize_weights: bool = False,
        random_state: Optional[int] = None,
        calibrate_meta: Optional[str] = None,  # None | "sigmoid" | "isotonic"
    ):
        super().__init__(
            model_constructors,
            loss_fn=loss_fn,
            optimizer=optimizer,
            ensemble_method=ensemble_method,
            weights=weights,
            meta_learner=meta_learner,
            tta_fn=tta_fn,
            tta_samples=tta_samples,
            jit_infer=jit_infer,
            optimize_weights=optimize_weights,
            random_state=random_state,
        )
        if combine not in {"proba", "geom"}:
            raise ValueError("combine must be 'proba' or 'geom'")
        self.combine = combine
        self.calibrate_meta = calibrate_meta

    # ---- meta-learner training ----

    def _fit_meta(self, X_val: jnp.ndarray, y_val: jnp.ndarray):
        probs = self._per_member_class_probs(X_val, key=None)  # list of (N,C)
        # Concatenate per-member class probs → (N, M*C)
        features = np.hstack(probs)
        y_val_np = np.asarray(y_val)

        meta = self.meta_learner
        if meta is None:
            from sklearn.linear_model import LogisticRegression

            meta = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto")

        if self.calibrate_meta:
            from sklearn.calibration import CalibratedClassifierCV

            meta = CalibratedClassifierCV(meta, method=self.calibrate_meta, cv=3)

        self.meta_ = meta.fit(features, y_val_np)
        self.classes_ = np.unique(y_val_np)

        # fallback weights (uniform) for completeness
        if self.weights_ is None:
            self.weights_ = np.ones(len(probs)) / len(probs)

    # ---- per-member outputs ----

    def _per_member_class_probs(
        self, X: jnp.ndarray, key: Optional[jr.PRNGKey]
    ) -> List[np.ndarray]:
        # Convert raw outputs/logits to softmax probs
        def to_softmax(raw: np.ndarray) -> np.ndarray:
            # raw expected shape (N, C) logits
            z = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(z)
            p = exp / exp.sum(axis=1, keepdims=True)
            return p.astype(np.float64)

        probs = self._per_member_probs(X, key, postproc=lambda r: to_softmax(r))
        return probs  # list of (N,C)

    def _per_member_regression_preds(self, X, key):  # not used here
        raise RuntimeError("Internal misuse: regression path in multiclass ensemble")

    # ---- predict / predict_proba ----

    def predict_proba(
        self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None
    ) -> np.ndarray:
        X = jnp.asarray(X)

        if self.ensemble_method == "stacking":
            blocks = self._per_member_class_probs(X, key)  # list of (N,C)
            feats = np.hstack(blocks)  # (N, M*C)
            if hasattr(self.meta_, "predict_proba"):
                proba = self.meta_.predict_proba(feats)
            else:
                # fallback: predict classes then one-hot
                preds = self.meta_.predict(feats)
                proba = np.zeros((len(preds), len(self.classes_)), dtype=float)
                for i, cls in enumerate(self.meta_.classes_):
                    proba[preds == cls, i] = 1.0
            # align columns to self.classes_
            if hasattr(self.meta_, "classes_") and not np.array_equal(
                self.meta_.classes_, self.classes_
            ):
                aligned = np.zeros_like(proba)
                for i, cls in enumerate(self.meta_.classes_):
                    j = int(np.where(self.classes_ == cls)[0][0])
                    aligned[:, j] = proba[:, i]
                return aligned
            return proba

        # weighted-average
        probs = self._per_member_class_probs(X, key)  # list of (N,C)
        A = np.stack(probs, axis=0)  # (M, N, C)
        w = (
            self.weights_
            if self.weights_ is not None
            else np.ones(A.shape[0]) / A.shape[0]
        )
        w = _normalize_weights(np.asarray(w, dtype=float))

        if self.combine == "proba":
            out = np.tensordot(w, A, axes=1)  # (N, C)
            out = np.clip(out, 0.0, 1.0)
            out /= out.sum(axis=1, keepdims=True)
            return out
        else:  # "geom"
            return _geom_mean_multiclass(A, w)

    def predict(self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None):
        proba = self.predict_proba(X, key=key)
        return self.classes_[np.argmax(proba, axis=1)]


# ------------------------------ Regression ---------------------------------


class EquinoxEnsembleRegression(EquinoxEnsembleBase):
    """
    Regression ensemble.
    Weighted-average (optionally weight-optimized on validation) or stacking with a meta-regressor.
    """

    def __init__(
        self,
        model_constructors: Sequence[
            Callable[[jr.PRNGKey], Union[eqx.Module, Tuple[eqx.Module, Any]]]
        ],
        *,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[
            optax.GradientTransformation, Sequence[optax.GradientTransformation]
        ],
        ensemble_method: str = "weighted_average",
        weights: Optional[Sequence[float]] = None,
        meta_learner: Optional[
            Any
        ] = None,  # e.g., sklearn.linear_model.LinearRegression()
        tta_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
        tta_samples: int = 0,
        jit_infer: bool = True,
        optimize_weights: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            model_constructors,
            loss_fn=loss_fn,
            optimizer=optimizer,
            ensemble_method=ensemble_method,
            weights=weights,
            meta_learner=meta_learner,
            tta_fn=tta_fn,
            tta_samples=tta_samples,
            jit_infer=jit_infer,
            optimize_weights=optimize_weights,
            random_state=random_state,
        )

    # ---- meta-learner training ----

    def _fit_meta(self, X_val: jnp.ndarray, y_val: jnp.ndarray):
        cols = self._per_member_regression_preds(X_val, key=None)  # list of (N,)
        feats = np.column_stack(cols)
        y_val_np = np.asarray(y_val).reshape(-1)

        meta = self.meta_learner
        if meta is None:
            from sklearn.linear_model import LinearRegression

            meta = LinearRegression()

        self.meta_ = meta.fit(feats, y_val_np)
        if self.weights_ is None:
            self.weights_ = np.ones(feats.shape[1]) / feats.shape[1]

    # ---- per-member outputs ----

    def _per_member_regression_preds(
        self, X: jnp.ndarray, key: Optional[jr.PRNGKey]
    ) -> List[np.ndarray]:
        def identity(raw: np.ndarray) -> np.ndarray:
            return raw.reshape(-1)

        outs = self._per_member_probs(X, key, postproc=identity)
        return outs  # list of (N,)

    def _per_member_class_probs(self, X, key):  # not used here
        raise RuntimeError(
            "Internal misuse: classification path in regression ensemble"
        )

    # ---- predict ----

    def predict(
        self, X: jnp.ndarray, *, key: Optional[jr.PRNGKey] = None
    ) -> np.ndarray:
        X = jnp.asarray(X)

        if self.ensemble_method == "stacking":
            cols = self._per_member_regression_preds(X, key)
            feats = np.column_stack(cols)
            return np.asarray(self.meta_.predict(feats)).reshape(-1)

        cols = self._per_member_regression_preds(X, key)  # list of (N,)
        P = np.column_stack(cols)  # (N, M)
        w = (
            self.weights_
            if self.weights_ is not None
            else np.ones(P.shape[1]) / P.shape[1]
        )
        w = _normalize_weights(np.asarray(w, dtype=float))
        return (P @ w).reshape(-1)
