# File: greedy_weighted_ensemble.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_weighted_ensemble.py

Greedy forward selection for weighted‐average ensembles –
production‐level, scikit‐learn style.
"""

import logging
from typing import Dict, List, Optional, Union

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


class WeightOptimizer(BaseEstimator):
    """
    Solve for non‐negative, sum‐to‐one weights.

    Params
    ------
    task : {'binary','multiclass','regression'}
    tol  : float
    """

    def __init__(self, task: str = "binary", tol: float = 1e-9):
        self.task = task
        self.tol = tol

    def fit(self, preds: List[np.ndarray], y: np.ndarray) -> "WeightOptimizer":
        y = np.asarray(y)
        K = len(preds)

        if self.task == "regression":
            P = np.stack(preds, axis=1)

            def obj(w):
                return mean_squared_error(y, P.dot(w))

        else:
            P = np.stack(preds, axis=-1)

            def obj(w):
                Pwc = np.tensordot(P, w, axes=([2], [0]))
                return log_loss(y, Pwc)

        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: w},
        ]
        w0 = np.ones(K) / K
        res = minimize(
            obj, w0, method="SLSQP", constraints=cons, options={"ftol": self.tol}
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
        return np.tensordot(P, self.coef_, axes=([2], [0]))


class GreedyWeightedEnsembleSelector(BaseEstimator):
    """
    Greedy forward selection for weighted‐average ensembles.

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
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={task!r}")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric={metric!r}")
        if not base_estimators:
            raise ValueError("base_estimators must not be empty.")
        self.base_estimators = base_estimators
        self.task = task
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_models = max_models
        self.random_state = random_state
        self.tol = tol

    def _get_cv(self) -> BaseCrossValidator:
        if isinstance(self.cv, BaseCrossValidator):
            return self.cv
        if self.task == "regression":
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

    def _get_oof(
        self,
        name: str,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: Optional[int],
    ) -> np.ndarray:
        splitter = self._get_cv()
        n = X.shape[0]
        if self.task == "regression":
            oof = np.zeros(n, dtype=float)
        else:
            oof = np.zeros((n, n_classes), dtype=float)
        for tr, va in splitter.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            if self.task == "regression":
                oof[va] = m.predict(X[va])
            else:
                oof[va] = m.predict_proba(X[va])
        return oof

    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]):
        X_arr, y_arr = check_X_y(X, y, multi_output=False)
        check_array(X_arr)
        y_arr = np.ravel(y_arr)

        # determine n_classes
        if self.task in ("binary", "multiclass"):
            classes = np.unique(y_arr)
            n_classes = len(classes)
        else:
            n_classes = None

        outer_cv = self._get_cv()
        names = list(self.base_estimators.keys())

        # scoring mapping
        scoring = (
            "neg_log_loss" if self.metric == "log_loss" else "neg_mean_squared_error"
        )

        # 1) OOF preds
        _logger.info("Computing OOF predictions for %d candidates", len(names))
        oof = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_oof)(n, self.base_estimators[n], X_arr, y_arr, n_classes)
            for n in names
        )
        self.oof_preds_ = dict(zip(names, oof))

        # 2) greedy forward selection
        selected: List[str] = []
        best_score = np.inf
        best_weights: np.ndarray = np.array([])

        optimizer = WeightOptimizer(task=self.task)

        while True:
            improved = False
            candidate = None
            candidate_score = best_score
            candidate_w = None

            for n in names:
                if n in selected:
                    continue
                trial = [self.oof_preds_[m] for m in selected] + [self.oof_preds_[n]]
                optimizer.fit(trial, y_arr)
                loss = optimizer.best_loss_
                if loss < candidate_score * (1 - self.tol):
                    improved = True
                    candidate = n
                    candidate_score = loss
                    candidate_w = optimizer.coef_.copy()

            if not improved or candidate is None:
                break

            selected.append(candidate)
            best_score, best_weights = candidate_score, candidate_w  # type: ignore
            _logger.info(
                "Added %s → %s=%.6f; selected=%r",
                candidate,
                self.metric,
                best_score,
                selected,
            )
            if self.max_models and len(selected) >= self.max_models:
                break

        # fallback: best single
        if not selected:
            _logger.warning("No ensemble gain; selecting single best")
            raw = {}
            for n, est in self.base_estimators.items():
                sc = cross_val_score(
                    clone(est),
                    X_arr,
                    y_arr,
                    scoring=scoring,
                    cv=outer_cv,
                    n_jobs=self.n_jobs,
                )
                raw[n] = -np.mean(sc)
            best = min(raw, key=raw.get)
            selected, best_score, best_weights = [best], raw[best], np.array([1.0])

        # 3) re‐fit on full data
        self.selected_names_ = selected
        self.weights_ = best_weights
        self.fitted_estimators_ = {
            n: clone(self.base_estimators[n]).fit(X_arr, y_arr) for n in selected
        }
        self.best_score_ = best_score
        return self

    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        check_is_fitted(self, ["selected_names_", "weights_", "fitted_estimators_"])
        X_arr = check_array(X)
        if self.task == "regression":
            preds = [m.predict(X_arr).ravel() for m in self.fitted_estimators_.values()]
            return np.stack(preds, axis=1).dot(self.weights_)
        probss = [m.predict_proba(X_arr) for m in self.fitted_estimators_.values()]
        combo = np.tensordot(np.stack(probss, axis=-1), self.weights_, axes=([2], [0]))
        return combo.argmax(axis=1)

    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if self.task == "regression":
            raise AttributeError("predict_proba not available for regression")
        check_is_fitted(self, ["selected_names_", "weights_", "fitted_estimators_"])
        X_arr = check_array(X)
        probss = [m.predict_proba(X_arr) for m in self.fitted_estimators_.values()]
        return np.tensordot(np.stack(probss, axis=-1), self.weights_, axes=([2], [0]))


if __name__ == "__main__":
    """
    Example: synthetic regression & classification.
    Demonstrates fitting and evaluating a weighted‑average ensemble.
    """
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

    # ——— Regression demo ———
    Xr, yr = make_regression(
        n_samples=500, n_features=20, n_informative=15, noise=0.2, random_state=0
    )
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=0)

    regressors = {
        "ridge": Ridge(alpha=1.0, random_state=0),
        "rf": RandomForestRegressor(n_estimators=100, random_state=0),
    }
    reg_selector = GreedyWeightedEnsembleSelector(
        base_estimators=regressors,
        task="regression",
        metric="mse",
        cv=5,
        random_state=0,
        tol=1e-6,
    )
    reg_selector.fit(Xr_tr, yr_tr)
    yr_pred = reg_selector.predict(Xr_te)

    print("\n[WEIGHTED] Regression Results")
    print("Selected models:   ", reg_selector.selected_names_)
    print("Learned weights:   ", reg_selector.weights_)
    print("OOF CV loss (MSE): ", reg_selector.best_score_)
    print("Test    MSE:       ", mean_squared_error(yr_te, yr_pred))

    # ——— Classification demo ———
    Xc, yc = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=0,
    )
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.3, stratify=yc, random_state=0
    )

    classifiers = {
        "lr": LogisticRegression(max_iter=1000, random_state=0),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
    }
    clf_selector = GreedyWeightedEnsembleSelector(
        base_estimators=classifiers,
        task="multiclass",
        metric="log_loss",
        cv=5,
        random_state=0,
        tol=1e-6,
    )
    clf_selector.fit(Xc_tr, yc_tr)
    yc_proba = clf_selector.predict_proba(Xc_te)
    yc_pred = clf_selector.predict(Xc_te)

    print("\n[WEIGHTED] Classification Results")
    print("Selected models:      ", clf_selector.selected_names_)
    print("Learned weights:      ", clf_selector.weights_)
    print("OOF CV loss (logloss):", clf_selector.best_score_)
    print("Test    log_loss:     ", log_loss(yc_te, yc_proba))
    print("Test    accuracy:     ", accuracy_score(yc_te, yc_pred))
