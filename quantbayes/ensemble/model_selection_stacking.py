#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_weighted_ensemble.py

Greedy forward selection for weighted-average ensembles.
Now with:
- selection_mode: {'weight_opt', 'equal_weight'} (Kaggle-style hill-climb)
- with_replacement for equal_weight mode
- bagged ensemble selection (BES): n_bags, bag_frac
- optional dev split for stopping (dev_size)
- optional simple weight regularization (l2, entropic)
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------- utilities -------------------------- #


def _align_proba_to_global(
    P: np.ndarray, local_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    """
    Align predicted probabilities P (n, C_local) to global class order (n, C_global).
    Missing classes are filled with zeros.
    """
    Cg = len(global_classes)
    out = np.zeros((P.shape[0], Cg), dtype=float)
    for i, c in enumerate(local_classes):
        j = int(np.where(global_classes == c)[0][0])
        out[:, j] = P[:, i]
    # numerical guard
    s = out.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-15, None)
    out = out / s
    return out


def _eval_loss(
    task: str, y_true: np.ndarray, pred: np.ndarray, classes: Optional[Sequence] = None
) -> float:
    """Compute loss given task, with safe clipping for classification."""
    if task == "regression":
        return mean_squared_error(y_true, pred)
    # classification
    P = np.clip(pred, 1e-15, 1 - 1e-15)
    P /= np.clip(P.sum(axis=1, keepdims=True), 1e-15, None)
    return log_loss(y_true, P, labels=classes)


# ----------------------- weight optimizer ---------------------- #


class WeightOptimizer(BaseEstimator):
    """
    Solve for non-negative, sum-to-one weights on a fixed set of prediction arrays.

    Parameters
    ----------
    task : {'binary','multiclass','regression'}
    tol  : float
    l2   : float, optional  (ridge on weights)
    entropic : float, optional ( -sum w log w regularizer )
    classes : array-like, optional (for classification log_loss labels)
    clip_eps : float, small epsilon for prob safety
    """

    def __init__(
        self,
        task: str = "binary",
        tol: float = 1e-9,
        l2: float = 0.0,
        entropic: float = 0.0,
        classes: Optional[Sequence] = None,
        clip_eps: float = 1e-15,
    ):
        self.task = task
        self.tol = tol
        self.l2 = l2
        self.entropic = entropic
        self.classes = None if classes is None else np.asarray(classes)
        self.clip_eps = clip_eps

    def fit(self, preds: List[np.ndarray], y: np.ndarray) -> "WeightOptimizer":
        y = np.asarray(y)
        K = len(preds)
        if K == 0:
            raise ValueError("No predictions provided to optimize over.")
        if K == 1:
            self.coef_ = np.array([1.0], dtype=float)
            self.best_loss_ = _eval_loss(self.task, y, preds[0], self.classes)
            return self

        if self.task == "regression":
            # P: (n, K)
            P = np.stack(preds, axis=1)

            def obj(w):
                base = mean_squared_error(y, P.dot(w))
                reg = self.l2 * np.sum(w**2) - self.entropic * np.sum(
                    np.where(w > 0, w * np.log(np.clip(w, self.clip_eps, 1.0)), 0.0)
                )
                return base + reg

        else:
            # P: (n, C, K)
            P = np.stack(preds, axis=-1)

            def obj(w):
                Pw = np.tensordot(P, w, axes=([2], [0]))  # (n, C)
                Pw = np.clip(Pw, self.clip_eps, 1 - self.clip_eps)
                Pw /= np.clip(Pw.sum(axis=1, keepdims=True), self.clip_eps, None)
                base = log_loss(y, Pw, labels=self.classes)
                reg = self.l2 * np.sum(w**2) - self.entropic * np.sum(
                    np.where(w > 0, w * np.log(np.clip(w, self.clip_eps, 1.0)), 0.0)
                )
                return base + reg

        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: w},  # w >= 0
        ]
        w0 = np.ones(K, dtype=float) / K
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            constraints=cons,
            options={"ftol": self.tol, "maxiter": 500},
        )
        if not res.success:
            _logger.error("Weight optimization failed: %s", res.message)
            raise RuntimeError(f"WeightOptimizer failed: {res.message}")

        self.coef_ = res.x
        self.best_loss_ = res.fun
        return self

    def predict(self, preds: List[np.ndarray]) -> np.ndarray:
        check_is_fitted(self, "coef_")
        if self.task == "regression":
            return np.stack(preds, axis=1).dot(self.coef_)
        P = np.stack(preds, axis=-1)
        Pw = np.tensordot(P, self.coef_, axes=([2], [0]))
        Pw = np.clip(Pw, self.clip_eps, 1 - self.clip_eps)
        s = np.clip(Pw.sum(axis=1, keepdims=True), self.clip_eps, None)
        return Pw / s


# -------------------- greedy weighted selector -------------------- #


class GreedyWeightedEnsembleSelector(BaseEstimator):
    """
    Greedy forward selection for weighted-average ensembles.

    Parameters
    ----------
    base_estimators : Dict[str, BaseEstimator]
    task            : {'binary','multiclass','regression'}
    metric          : {'log_loss','mse'}, default='log_loss'
    cv              : int or BaseCrossValidator
    n_jobs          : int
    max_models      : int, optional
    random_state    : int, optional
    tol             : float, default=1e-6

    # New:
    selection_mode  : {'weight_opt','equal_weight'}
        'weight_opt'  → SLSQP re-fit of simplex weights at each add (convex step)
        'equal_weight'→ Kaggle hill-climb (equal-weight add); can use with_replacement
    with_replacement: bool (only used in equal_weight)
    n_bags          : int = 0 (bagged ensemble selection)
    bag_frac        : float in (0,1], default 0.7
    dev_size        : Optional[float]  (0<dev_size<1). If given, use this subset of OOF
                      indices to drive selection/early stopping.
    l2              : float (weight regularization) for 'weight_opt'
    entropic        : float (entropic regularization) for 'weight_opt'
    patience        : int, optional; if >0, allow up to `patience` non-improving
                      accepted steps (only relevant for equal_weight + with_replacement)
    """

    SUPPORTED_TASKS = {"binary", "multiclass", "regression"}
    SUPPORTED_METRICS = {"log_loss", "mse"}

    def __init__(
        self,
        base_estimators: Dict[str, BaseEstimator],
        task: str = "binary",
        metric: str = "log_loss",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: int = 1,
        max_models: Optional[int] = None,
        random_state: Optional[int] = None,
        tol: float = 1e-6,
        selection_mode: str = "weight_opt",
        with_replacement: bool = False,
        n_bags: int = 0,
        bag_frac: float = 0.7,
        dev_size: Optional[float] = None,
        l2: float = 0.0,
        entropic: float = 0.0,
        patience: int = 0,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={task!r}")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric={metric!r}")
        if not base_estimators:
            raise ValueError("base_estimators must not be empty.")
        if selection_mode not in {"weight_opt", "equal_weight"}:
            raise ValueError("selection_mode must be 'weight_opt' or 'equal_weight'")
        self.base_estimators = base_estimators
        self.task = task
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_models = max_models
        self.random_state = random_state
        self.tol = tol

        self.selection_mode = selection_mode
        self.with_replacement = with_replacement
        self.n_bags = int(n_bags)
        self.bag_frac = float(bag_frac)
        self.dev_size = dev_size
        self.l2 = l2
        self.entropic = entropic
        self.patience = int(patience)

    # --------------- CV splitter --------------- #
    def _get_cv(self) -> BaseCrossValidator:
        if isinstance(self.cv, BaseCrossValidator):
            return self.cv
        if self.task == "regression":
            return KFold(
                n_splits=int(self.cv), shuffle=True, random_state=self.random_state
            )
        return StratifiedKFold(
            n_splits=int(self.cv), shuffle=True, random_state=self.random_state
        )

    # --------------- OOF builder --------------- #
    def _get_oof(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray],
    ) -> np.ndarray:
        splitter = self._get_cv()
        n = X.shape[0]

        if self.task == "regression":
            oof = np.zeros(n, dtype=float)
        else:
            assert classes is not None
            oof = np.zeros((n, len(classes)), dtype=float)

        for tr, va in splitter.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            if self.task == "regression":
                oof[va] = np.asarray(m.predict(X[va])).ravel()
            else:
                P = m.predict_proba(X[va])
                P = _align_proba_to_global(P, m.classes_, classes)
                oof[va] = P
        return oof

    # --------------- selection passes --------------- #
    def _single_pass(
        self,
        names: List[str],
        y: np.ndarray,
        oof_dict: Dict[str, np.ndarray],
        eval_idx: np.ndarray,
        classes: Optional[np.ndarray],
    ) -> Tuple[List[str], np.ndarray, float]:
        """
        Run one greedy selection pass on eval_idx.
        Returns (selected_unique_names, weights, best_loss_on_eval_idx)
        """
        rng = np.random.RandomState(self.random_state)
        selected_seq: List[str] = []
        best_loss = np.inf
        best_weights = None

        def current_equal_weight_blend(seq: List[str]) -> np.ndarray:
            if len(seq) == 0:
                return None  # type: ignore
            mats = [oof_dict[s][eval_idx] for s in seq]
            S = np.sum(mats, axis=0)
            return S / len(seq)

        no_improve_steps = 0
        while True:
            improved = False
            cand_name = None
            cand_loss = best_loss
            cand_weights = None

            for n in names:
                if self.selection_mode == "equal_weight":
                    # with_replacement? if False: skip already used
                    if (not self.with_replacement) and (n in selected_seq):
                        continue
                    cur = current_equal_weight_blend(selected_seq)
                    Pn = oof_dict[n][eval_idx]
                    if cur is None:
                        blended = Pn
                    else:
                        k = len(selected_seq)
                        blended = (cur * k + Pn) / (k + 1)
                    loss = _eval_loss(self.task, y[eval_idx], blended, classes)
                    if loss < cand_loss * (1 - self.tol):
                        improved, cand_name, cand_loss = True, n, loss
                        cand_weights = None
                else:
                    # weight_opt mode: re-opt simplex weights each try
                    # if without replacement, skip already included
                    if (not self.with_replacement) and (n in selected_seq):
                        continue
                    trial_seq = selected_seq + [n]
                    trial_preds = [oof_dict[m][eval_idx] for m in trial_seq]
                    optimizer = WeightOptimizer(
                        task=self.task,
                        tol=max(self.tol * 1e-2, 1e-10),
                        l2=self.l2,
                        entropic=self.entropic,
                        classes=classes,
                    )
                    optimizer.fit(trial_preds, y[eval_idx])
                    loss = optimizer.best_loss_
                    if loss < cand_loss * (1 - self.tol):
                        improved, cand_name, cand_loss = True, n, loss
                        cand_weights = optimizer.coef_.copy()

            if not improved or cand_name is None:
                break

            selected_seq.append(cand_name)
            best_loss = cand_loss
            best_weights = cand_weights
            if self.max_models and len(selected_seq) >= self.max_models:
                break

            # patience only meaningful if we allow adding even with tiny/no gains
            if improved:
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if self.patience and no_improve_steps > self.patience:
                    break

        # finalize: convert to unique/weights
        if self.selection_mode == "equal_weight":
            # counts → weights
            counts = {}
            for s in selected_seq:
                counts[s] = counts.get(s, 0) + 1
            if not counts:
                return [], np.array([]), np.inf
            sel_unique = list(counts.keys())
            w = np.array([counts[s] for s in sel_unique], dtype=float)
            w = w / w.sum()
            return sel_unique, w, best_loss

        # weight_opt: re-opt weights on full eval_idx using unique selected set
        if len(selected_seq) == 0:
            return [], np.array([]), np.inf
        sel_unique = []
        for s in selected_seq:
            if s not in sel_unique:
                sel_unique.append(s)
        trial_preds = [oof_dict[m][eval_idx] for m in sel_unique]
        optimizer = WeightOptimizer(
            task=self.task,
            tol=max(self.tol * 1e-2, 1e-10),
            l2=self.l2,
            entropic=self.entropic,
            classes=classes,
        )
        optimizer.fit(trial_preds, y[eval_idx])
        return sel_unique, optimizer.coef_.copy(), optimizer.best_loss_

    # ------------------------------- fit ------------------------------- #
    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]):
        X_arr, y_arr = check_X_y(X, y, multi_output=False)
        check_array(X_arr)
        y_arr = np.ravel(y_arr)
        self.n_features_in_ = X_arr.shape[1]

        # record global classes for classification
        if self.task in ("binary", "multiclass"):
            self.classes_ = np.unique(y_arr)
        else:
            self.classes_ = None

        # build OOF predictions for all candidates
        names = list(self.base_estimators.keys())
        _logger.info("Computing OOF predictions for %d candidates", len(names))
        oof_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_oof)(self.base_estimators[n], X_arr, y_arr, self.classes_)
            for n in names
        )
        self.oof_preds_ = dict(zip(names, oof_list))

        n = len(y_arr)
        rng = np.random.RandomState(self.random_state)

        # choose eval indices for selection (dev split) if requested
        if self.dev_size is not None:
            if not (0.0 < float(self.dev_size) < 1.0):
                raise ValueError("dev_size must be in (0,1) if provided.")
            m = int(np.ceil(n * float(self.dev_size)))
            eval_idx = rng.choice(n, size=m, replace=False)
        else:
            eval_idx = np.arange(n)

        # ------------- selection (bagged or single pass) ------------- #
        if self.n_bags > 0:
            counts = np.zeros(len(names), dtype=float)
            for b in range(self.n_bags):
                sz = max(2, int(len(eval_idx) * float(self.bag_frac)))
                bag_idx = rng.choice(eval_idx, size=sz, replace=True)
                sel_b, w_b, _ = self._single_pass(
                    names, y_arr, self.oof_preds_, bag_idx, self.classes_
                )
                for s in sel_b:
                    counts[names.index(s)] += 1.0

            mask = counts > 0
            if not np.any(mask):
                _logger.warning(
                    "Bagging produced no selections; falling back to single pass."
                )
                sel, w, best_loss = self._single_pass(
                    names, y_arr, self.oof_preds_, eval_idx, self.classes_
                )
            else:
                sel = [names[i] for i in np.where(mask)[0]]
                # if too many and max_models set: keep top-k by counts
                if self.max_models and len(sel) > self.max_models:
                    order = np.argsort(counts[np.where(mask)[0]])[::-1]
                    sel = [sel[i] for i in order[: self.max_models]]
                # final weights:
                if self.selection_mode == "equal_weight":
                    w = np.ones(len(sel), dtype=float) / len(sel)
                else:
                    preds = [self.oof_preds_[m][eval_idx] for m in sel]
                    optimizer = WeightOptimizer(
                        task=self.task,
                        tol=max(self.tol * 1e-2, 1e-10),
                        l2=self.l2,
                        entropic=self.entropic,
                        classes=self.classes_,
                    )
                    optimizer.fit(preds, y_arr[eval_idx])
                    w = optimizer.coef_.copy()
                # compute final eval loss with chosen weights
                if self.task == "regression":
                    blended = np.stack(
                        [self.oof_preds_[m][eval_idx] for m in sel], axis=1
                    ).dot(w)
                else:
                    mats = [self.oof_preds_[m][eval_idx] for m in sel]
                    blended = np.tensordot(np.stack(mats, axis=-1), w, axes=([2], [0]))
                best_loss = _eval_loss(
                    self.task, y_arr[eval_idx], blended, self.classes_
                )
        else:
            sel, w, best_loss = self._single_pass(
                names, y_arr, self.oof_preds_, eval_idx, self.classes_
            )

        # fallback: best single by CV if nothing selected
        if not sel:
            _logger.warning("No ensemble gain; selecting best single by CV.")
            outer_cv = self._get_cv()
            scoring = (
                "neg_log_loss"
                if self.metric == "log_loss"
                else "neg_mean_squared_error"
            )
            raw = {}
            for n, est in self.base_estimators.items():
                try:
                    sc = cross_val_score(
                        clone(est),
                        X_arr,
                        y_arr,
                        scoring=scoring,
                        cv=outer_cv,
                        n_jobs=self.n_jobs,
                    )
                    raw[n] = -np.mean(sc)
                except Exception as e:
                    _logger.warning("CV failed for %s: %s", n, e)
            if not raw:
                raise RuntimeError("All base estimators failed CV.")
            sel, w, best_loss = (
                [min(raw, key=raw.get)],
                np.array([1.0], dtype=float),
                raw[min(raw, key=raw.get)],
            )

        # ------------- refit selected on full data ------------- #
        self.selected_names_ = sel
        self.weights_ = np.asarray(w, dtype=float)
        self.fitted_estimators_ = {
            n: clone(self.base_estimators[n]).fit(X_arr, y_arr) for n in sel
        }
        self.best_score_ = best_loss
        return self

    # ------------------------------- predict ------------------------------- #
    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        check_is_fitted(self, ["selected_names_", "weights_", "fitted_estimators_"])
        X_arr = check_array(X)

        if self.task == "regression":
            preds = [m.predict(X_arr).ravel() for m in self.fitted_estimators_.values()]
            yhat = np.stack(preds, axis=1).dot(self.weights_)
            return yhat

        mats = []
        for m in self.fitted_estimators_.values():
            P = m.predict_proba(X_arr)
            mats.append(_align_proba_to_global(P, m.classes_, self.classes_))
        combo = np.tensordot(np.stack(mats, axis=-1), self.weights_, axes=([2], [0]))
        return self.classes_[combo.argmax(axis=1)]

    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if self.task == "regression":
            raise AttributeError("predict_proba not available for regression")
        check_is_fitted(self, ["selected_names_", "weights_", "fitted_estimators_"])
        X_arr = check_array(X)
        mats = []
        for m in self.fitted_estimators_.values():
            P = m.predict_proba(X_arr)
            mats.append(_align_proba_to_global(P, m.classes_, self.classes_))
        return np.tensordot(np.stack(mats, axis=-1), self.weights_, axes=([2], [0]))


# ------------------------------- demo ------------------------------- #
if __name__ == "__main__":
    """
    Example: synthetic regression & classification.
    Demonstrates both selection modes and bagged selection.
    """
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (
        RandomForestRegressor,
        RandomForestClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

    rng = 0

    # ——— Regression demo ———
    Xr, yr = make_regression(
        n_samples=800, n_features=30, n_informative=20, noise=0.4, random_state=rng
    )
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.25, random_state=rng
    )

    regressors = {
        "ridge": Ridge(alpha=1.0, random_state=rng),
        "rf": RandomForestRegressor(n_estimators=200, random_state=rng),
        "gbr": GradientBoostingRegressor(random_state=rng),
    }

    print("\n[Weighted] Regression – weight_opt + BES")
    reg_selector = GreedyWeightedEnsembleSelector(
        base_estimators=regressors,
        task="regression",
        metric="mse",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        selection_mode="weight_opt",
        n_bags=15,
        bag_frac=0.7,
        dev_size=0.2,
        l2=0.0,
        entropic=0.0,
        tol=1e-6,
    ).fit(Xr_tr, yr_tr)
    yr_pred = reg_selector.predict(Xr_te)
    print("Selected models:   ", reg_selector.selected_names_)
    print("Learned weights:   ", np.round(reg_selector.weights_, 4))
    print("OOF (eval) MSE:    ", reg_selector.best_score_)
    print("Test    MSE:       ", mean_squared_error(yr_te, yr_pred))

    print("\n[Weighted] Regression – equal_weight (Kaggle mode) + with_replacement")
    reg_eq = GreedyWeightedEnsembleSelector(
        base_estimators=regressors,
        task="regression",
        metric="mse",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        selection_mode="equal_weight",
        with_replacement=True,
        n_bags=15,
        bag_frac=0.7,
        dev_size=0.2,
        tol=1e-6,
    ).fit(Xr_tr, yr_tr)
    yr_pred_eq = reg_eq.predict(Xr_te)
    print("Selected models:   ", reg_eq.selected_names_)
    print("Equal weights:     ", np.round(reg_eq.weights_, 4))
    print("OOF (eval) MSE:    ", reg_eq.best_score_)
    print("Test    MSE:       ", mean_squared_error(yr_te, yr_pred_eq))

    # ——— Classification demo ———
    Xc, yc = make_classification(
        n_samples=1200,
        n_features=40,
        n_informative=18,
        n_redundant=6,
        n_classes=3,
        weights=[0.5, 0.3, 0.2],
        random_state=rng,
    )
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.25, stratify=yc, random_state=rng
    )

    classifiers = {
        "lr": LogisticRegression(max_iter=2000, random_state=rng),
        "rf": RandomForestClassifier(n_estimators=300, random_state=rng),
    }

    print("\n[Weighted] Multiclass – weight_opt + BES")
    clf_selector = GreedyWeightedEnsembleSelector(
        base_estimators=classifiers,
        task="multiclass",
        metric="log_loss",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        selection_mode="weight_opt",
        n_bags=20,
        bag_frac=0.7,
        dev_size=0.2,
        l2=0.0,
        entropic=0.0,
        tol=1e-6,
    ).fit(Xc_tr, yc_tr)
    Pc = clf_selector.predict_proba(Xc_te)
    yc_pred = clf_selector.predict(Xc_te)
    print("Selected models:      ", clf_selector.selected_names_)
    print("Learned weights:      ", np.round(clf_selector.weights_, 4))
    print("OOF (eval) log_loss:  ", clf_selector.best_score_)
    print("Test    log_loss:     ", log_loss(yc_te, Pc, labels=clf_selector.classes_))
    print("Test    accuracy:     ", accuracy_score(yc_te, yc_pred))

    print("\n[Weighted] Multiclass – equal_weight (Kaggle mode) + with_replacement")
    clf_eq = GreedyWeightedEnsembleSelector(
        base_estimators=classifiers,
        task="multiclass",
        metric="log_loss",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        selection_mode="equal_weight",
        with_replacement=True,
        n_bags=20,
        bag_frac=0.7,
        dev_size=0.2,
        tol=1e-6,
    ).fit(Xc_tr, yc_tr)
    Pc_eq = clf_eq.predict_proba(Xc_te)
    yc_pred_eq = clf_eq.predict(Xc_te)
    print("Selected models:      ", clf_eq.selected_names_)
    print("Equal weights:        ", np.round(clf_eq.weights_, 4))
    print("OOF (eval) log_loss:  ", clf_eq.best_score_)
    print("Test    log_loss:     ", log_loss(yc_te, Pc_eq, labels=clf_eq.classes_))
    print("Test    accuracy:     ", accuracy_score(yc_te, yc_pred_eq))
