#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_stacking_ensemble_v2.py

Greedy selection of base learners for stacking with a meta-learner.
Now with:
- bagged selection (n_bags, bag_frac)
- optional dev split for selection (dev_size)
- class-probability alignment
- small API robustness tweaks
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _align_proba_to_global(
    P: np.ndarray, local_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    Cg = len(global_classes)
    out = np.zeros((P.shape[0], Cg), dtype=float)
    for i, c in enumerate(local_classes):
        j = int(np.where(global_classes == c)[0][0])
        out[:, j] = P[:, i]
    s = out.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-15, None)
    return out / s


class GreedyStackingEnsembleSelector(BaseEstimator):
    """
    Greedy forward selection of base models for stacking.

    Parameters
    ----------
    base_estimators : Dict[str, BaseEstimator]
    meta_learner    : BaseEstimator (if None → LinearRegression or LogisticRegression)
    task            : {'binary','multiclass','regression'}
    cv              : int or BaseCrossValidator
    n_jobs          : int
    max_models      : int, optional
    random_state    : int, optional
    scoring         : Optional[str]; default neg_log_loss (cls) / neg_mse (reg)
    tol             : float, default=1e-6

    # New:
    n_bags          : int = 0 (bagged selection)
    bag_frac        : float in (0,1], default 0.7
    dev_size        : Optional[float] in (0,1) for selection/early stopping
    """

    SUPPORTED_TASKS = {"binary", "multiclass", "regression"}

    def __init__(
        self,
        base_estimators: Dict[str, BaseEstimator],
        meta_learner: Optional[BaseEstimator] = None,
        task: str = "binary",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: int = 1,
        max_models: Optional[int] = None,
        random_state: Optional[int] = None,
        scoring: Optional[str] = None,
        tol: float = 1e-6,
        n_bags: int = 0,
        bag_frac: float = 0.7,
        dev_size: Optional[float] = None,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={task!r}")
        if not base_estimators:
            raise ValueError("base_estimators must not be empty.")
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.task = task
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_models = max_models
        self.random_state = random_state
        self.scoring = scoring
        self.tol = tol
        self.n_bags = int(n_bags)
        self.bag_frac = float(bag_frac)
        self.dev_size = dev_size

    def _default_meta(self):
        return (
            LinearRegression()
            if self.task == "regression"
            else LogisticRegression(solver="lbfgs", max_iter=2000)
        )

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

    def _aligned_oof(self, model, X, y, classes, splitter) -> np.ndarray:
        n = X.shape[0]
        if self.task == "regression":
            oof = np.zeros((n, 1), dtype=float)
            for tr, va in splitter.split(X, y):
                m = clone(model).fit(X[tr], y[tr])
                oof[va, 0] = np.asarray(m.predict(X[va])).ravel()
            return oof
        # classification: align columns to `classes`
        C = len(classes)
        oof = np.zeros((n, C), dtype=float)
        for tr, va in splitter.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            P = m.predict_proba(X[va])
            oof[va] = _align_proba_to_global(P, m.classes_, classes)
        return oof

    def _single_pass(
        self,
        names: List[str],
        y: np.ndarray,
        oof: Dict[str, np.ndarray],
        eval_idx: np.ndarray,
        splitter: BaseCrossValidator,
        scoring: str,
    ) -> Tuple[List[str], float]:
        """
        Greedy add base models; at each step train+CV the meta-learner on eval_idx
        using the OOF meta-features of (selected + candidate).
        Returns (selected_names, best_loss_on_eval).
        """
        selected: List[str] = []
        best_loss = np.inf

        while True:
            improved = False
            best_cand = None
            cand_score = best_loss

            for n in names:
                if n in selected:
                    continue
                trial = selected + [n]
                X_meta = np.column_stack([oof[m][eval_idx] for m in trial])
                try:
                    scores = cross_val_score(
                        clone(self.meta_learner_),
                        X_meta,
                        y[eval_idx],
                        scoring=scoring,
                        cv=splitter,
                        n_jobs=1,
                    )
                    loss = -np.mean(scores)
                except Exception as e:
                    _logger.warning("CV failed for %s: %s", n, e)
                    continue
                if loss < cand_score * (1 - self.tol):
                    improved, best_cand, cand_score = True, n, loss

            if not improved or best_cand is None:
                break

            selected.append(best_cand)
            best_loss = cand_score
            if self.max_models and len(selected) >= self.max_models:
                break

        return selected, best_loss

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, multi_output=False)
        check_array(X_arr)
        y_arr = np.ravel(y_arr)

        classes = None
        if self.task in {"binary", "multiclass"}:
            classes = np.unique(y_arr)
        self.classes_ = classes

        self.meta_learner_ = (
            clone(self.meta_learner) if self.meta_learner else self._default_meta()
        )
        scoring = self.scoring or (
            "neg_log_loss" if self.task != "regression" else "neg_mean_squared_error"
        )

        names = list(self.base_estimators.keys())
        splitter = self._get_cv()

        _logger.info(
            "Computing class-aligned OOF predictions for %d candidates", len(names)
        )
        oof_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._aligned_oof)(
                self.base_estimators[n], X_arr, y_arr, classes, splitter
            )
            for n in names
        )
        self.oof_preds_ = dict(zip(names, oof_list))

        n = len(y_arr)
        rng = np.random.RandomState(self.random_state)
        if self.dev_size is not None:
            if not (0.0 < float(self.dev_size) < 1.0):
                raise ValueError("dev_size must be in (0,1) if provided.")
            m = int(np.ceil(n * float(self.dev_size)))
            eval_idx = rng.choice(n, size=m, replace=False)
        else:
            eval_idx = np.arange(n)

        # selection (bagged or single pass)
        if self.n_bags > 0:
            counts = np.zeros(len(names), dtype=float)
            for _ in range(self.n_bags):
                sz = max(2, int(len(eval_idx) * float(self.bag_frac)))
                bag_idx = rng.choice(eval_idx, size=sz, replace=True)
                s, _ = self._single_pass(
                    names, y_arr, self.oof_preds_, bag_idx, splitter, scoring
                )
                for nm in s:
                    counts[names.index(nm)] += 1.0

            mask = counts > 0
            if not np.any(mask):
                _logger.warning(
                    "Bagging produced no selections; falling back to single pass."
                )
                selected, best_loss = self._single_pass(
                    names, y_arr, self.oof_preds_, eval_idx, splitter, scoring
                )
            else:
                selected = [names[i] for i in np.where(mask)[0]]
                if self.max_models and len(selected) > self.max_models:
                    order = np.argsort(counts[np.where(mask)[0]])[::-1]
                    selected = [selected[i] for i in order[: self.max_models]]
                # recompute best_loss on eval_idx with the chosen set
                X_meta_eval = np.column_stack(
                    [self.oof_preds_[m][eval_idx] for m in selected]
                )
                scores = cross_val_score(
                    clone(self.meta_learner_),
                    X_meta_eval,
                    y_arr[eval_idx],
                    scoring=scoring,
                    cv=splitter,
                    n_jobs=1,
                )
                best_loss = -np.mean(scores)
        else:
            selected, best_loss = self._single_pass(
                names, y_arr, self.oof_preds_, eval_idx, splitter, scoring
            )

        if not selected:
            _logger.warning("No stacking gain; falling back to best single by CV.")
            raw = {}
            for n, est in self.base_estimators.items():
                try:
                    sc = cross_val_score(
                        clone(est),
                        X_arr,
                        y_arr,
                        scoring=scoring,
                        cv=splitter,
                        n_jobs=self.n_jobs,
                    )
                    raw[n] = -np.mean(sc)
                except Exception as e:
                    _logger.warning("CV failed for %s: %s", n, e)
            if not raw:
                raise RuntimeError("All base estimators failed CV.")
            selected, best_loss = [min(raw, key=raw.get)], min(raw.values())

        # Refit selected bases on full data
        self.selected_names_ = selected
        self.fitted_estimators_ = {
            m: clone(self.base_estimators[m]).fit(X_arr, y_arr) for m in selected
        }
        # Train meta-learner on full OOF meta-features (common stacking practice)
        X_meta_full = np.column_stack([self.oof_preds_[m] for m in selected])
        self.meta_learner_ = clone(self.meta_learner_).fit(X_meta_full, y_arr)
        self.best_score_ = best_loss
        return self

    def predict(self, X):
        check_is_fitted(
            self, ["selected_names_", "fitted_estimators_", "meta_learner_"]
        )
        X_arr = check_array(X)
        cols = [
            (
                m.predict_proba(X_arr)
                if self.task != "regression"
                else np.asarray(m.predict(X_arr)).reshape(-1, 1)
            )
            for m in self.fitted_estimators_.values()
        ]
        meta_X = np.hstack(cols)
        return self.meta_learner_.predict(meta_X)

    def predict_proba(self, X):
        if self.task == "regression":
            raise AttributeError("predict_proba not available for regression")
        check_is_fitted(
            self, ["selected_names_", "fitted_estimators_", "meta_learner_"]
        )
        X_arr = check_array(X)
        cols = [m.predict_proba(X_arr) for m in self.fitted_estimators_.values()]
        meta_X = np.hstack(cols)
        return self.meta_learner_.predict_proba(meta_X)


# ------------------------------- demo ------------------------------- #
if __name__ == "__main__":
    """
    Example: synthetic regression & classification.
    Demonstrates greedy stacking with and without bagging.
    """
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

    rng = 0

    # ——— Regression demo ———
    Xr, yr = make_regression(
        n_samples=800, n_features=35, n_informative=18, noise=0.5, random_state=rng
    )
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.25, random_state=rng
    )

    regressors = {
        "ridge": Ridge(alpha=1.0, random_state=rng),
        "rf": RandomForestRegressor(n_estimators=250, random_state=rng),
        "gbr": GradientBoostingRegressor(random_state=rng),
    }

    print("\n[Stacking] Regression – bagged selection")
    reg_stack = GreedyStackingEnsembleSelector(
        base_estimators=regressors,
        meta_learner=None,  # defaults to LinearRegression
        task="regression",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        n_bags=15,
        bag_frac=0.7,
        dev_size=0.2,
        tol=1e-6,
    ).fit(Xr_tr, yr_tr)
    # Build meta inputs on test from fitted bases:
    preds_te = [
        (
            m.predict_proba(Xr_te)
            if reg_stack.task != "regression"
            else np.asarray(m.predict(Xr_te)).reshape(-1, 1)
        )
        for m in reg_stack.fitted_estimators_.values()
    ]
    meta_X_te = np.hstack(preds_te)
    yr_pred = reg_stack.meta_learner_.predict(meta_X_te)
    print("Selected models:   ", reg_stack.selected_names_)
    print("OOF (eval) loss:   ", reg_stack.best_score_)
    print("Test    MSE:       ", mean_squared_error(yr_te, yr_pred))

    # ——— Classification demo ———
    Xc, yc = make_classification(
        n_samples=1200,
        n_features=45,
        n_informative=20,
        n_redundant=8,
        n_classes=3,
        weights=[0.5, 0.3, 0.2],
        random_state=rng,
    )
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.25, stratify=yc, random_state=rng
    )

    classifiers = {
        "lr": LogisticRegression(max_iter=2000, random_state=rng),
        "rf": RandomForestClassifier(n_estimators=350, random_state=rng),
    }

    print("\n[Stacking] Multiclass – single-pass greedy")
    clf_stack = GreedyStackingEnsembleSelector(
        base_estimators=classifiers,
        meta_learner=LogisticRegression(max_iter=2000, random_state=rng),
        task="multiclass",
        cv=5,
        n_jobs=-1,
        random_state=rng,
        n_bags=0,  # single pass
        dev_size=0.2,
        tol=1e-6,
    ).fit(Xc_tr, yc_tr)

    cols_te = [m.predict_proba(Xc_te) for m in clf_stack.fitted_estimators_.values()]
    meta_X_te = np.hstack(cols_te)
    Pc = clf_stack.meta_learner_.predict_proba(meta_X_te)
    yc_pred = np.argmax(
        Pc, axis=1
    )  # meta learner is LogisticRegression → also has predict
    print("Selected models:      ", clf_stack.selected_names_)
    print("OOF (eval) loss:      ", clf_stack.best_score_)
    print("Test    log_loss:     ", log_loss(yc_te, Pc, labels=np.unique(yc_tr)))
    print("Test    accuracy:     ", accuracy_score(yc_te, yc_pred))
