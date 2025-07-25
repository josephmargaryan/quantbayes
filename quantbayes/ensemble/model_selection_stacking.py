# File: greedy_stacking_ensemble.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greedy_stacking_ensemble.py

Greedy forward selection for stacking ensembles – production‐level, scikit‐learn style.
"""

import logging
from typing import Dict, List, Optional, Union

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


class GreedyStackingEnsembleSelector(BaseEstimator):
    """
    Greedy forward selection of a subset of models for stacking.

    Parameters
    ----------
    base_estimators : Dict[str, BaseEstimator]
        Candidate, hyperparameter‐tuned estimators.
    meta_learner : BaseEstimator, optional
        If None, defaults to LogisticRegression (classification) or
        LinearRegression (regression).
    task : {'binary','multiclass','regression'}, default='binary'
    cv : int or BaseCrossValidator, default=5
    n_jobs : int, default=1
    max_models : int, optional
    random_state : int, optional
    scoring : str, optional
    tol : float, default=1e-6
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

    def _default_meta(self) -> BaseEstimator:
        return (
            LinearRegression()
            if self.task == "regression"
            else LogisticRegression(solver="lbfgs", max_iter=1000)
        )

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
            oof = np.zeros((n, 1), dtype=float)
        else:
            oof = np.zeros((n, n_classes), dtype=float)
        for tr, va in splitter.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            if self.task == "regression":
                oof[va, 0] = m.predict(X[va])
            else:
                oof[va, :] = m.predict_proba(X[va])
        return oof

    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]):
        X_arr, y_arr = check_X_y(X, y, multi_output=False)
        check_array(X_arr)
        y_arr = np.ravel(y_arr)

        # Determine n_classes
        if self.task in ("binary", "multiclass"):
            classes = np.unique(y_arr)
            n_classes = len(classes)
        else:
            n_classes = None

        # Meta‐learner & scoring
        self.meta_learner_ = (
            clone(self.meta_learner) if self.meta_learner else self._default_meta()
        )
        scoring = self.scoring or (
            "neg_log_loss"
            if self.task in ("binary", "multiclass")
            else "neg_mean_squared_error"
        )

        # 1) Compute OOF preds
        names = list(self.base_estimators.keys())
        _logger.info("Computing OOF predictions for %d candidates", len(names))
        oof_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_oof)(n, self.base_estimators[n], X_arr, y_arr, n_classes)
            for n in names
        )
        self.oof_preds_ = dict(zip(names, oof_list))

        # 2) Greedy selection
        selected: List[str] = []
        best_score = np.inf
        outer_cv = self._get_cv()

        while True:
            improved = False
            candidate: Optional[str] = None
            candidate_score = best_score

            for n in names:
                if n in selected:
                    continue
                trial = selected + [n]
                X_meta = np.column_stack([self.oof_preds_[m] for m in trial])
                try:
                    scores = cross_val_score(
                        clone(self.meta_learner_),
                        X_meta,
                        y_arr,
                        scoring=scoring,
                        cv=outer_cv,
                        n_jobs=1,
                    )
                    loss = -np.mean(scores)
                    _logger.debug("  %s → loss=%.6f", n, loss)
                except Exception as e:
                    _logger.warning("  CV failed for %s: %s", n, e)
                    continue
                if loss < candidate_score * (1 - self.tol):
                    candidate, candidate_score = n, loss
                    improved = True

            if not improved or candidate is None:
                break

            selected.append(candidate)
            best_score = candidate_score
            _logger.info(
                "Added %s → loss=%.6f; selected=%r", candidate, best_score, selected
            )
            if self.max_models and len(selected) >= self.max_models:
                break

        # Fallback: best single
        if not selected:
            _logger.warning("No stacking gain; falling back to single best model")
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
                except Exception:
                    continue
            if not raw:
                raise RuntimeError("All base estimators failed CV.")
            best = min(raw, key=raw.get)
            selected, best_score = [best], raw[best]

        # 3) Fit meta‐learner on full OOF matrix
        X_meta_full = np.column_stack([self.oof_preds_[m] for m in selected])
        self.meta_learner_ = clone(self.meta_learner_).fit(X_meta_full, y_arr)

        # 4) Refit base models
        self.selected_names_ = selected
        self.fitted_estimators_ = {
            m: clone(self.base_estimators[m]).fit(X_arr, y_arr) for m in selected
        }
        self.best_score_ = best_score
        return self

    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        check_is_fitted(
            self, ["selected_names_", "fitted_estimators_", "meta_learner_"]
        )
        X_arr = check_array(X)
        if self.task == "regression":
            cols = [m.predict(X_arr).ravel() for m in self.fitted_estimators_.values()]
            return self.meta_learner_.predict(np.column_stack(cols))
        else:
            cols = [m.predict_proba(X_arr) for m in self.fitted_estimators_.values()]
            return self.meta_learner_.predict(np.hstack(cols))

    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if self.task == "regression":
            raise AttributeError("predict_proba not available for regression")
        check_is_fitted(
            self, ["selected_names_", "fitted_estimators_", "meta_learner_"]
        )
        X_arr = check_array(X)
        cols = [m.predict_proba(X_arr) for m in self.fitted_estimators_.values()]
        return self.meta_learner_.predict_proba(np.hstack(cols))


if __name__ == "__main__":
    """
    Example: synthetic regression & classification.
    Shows train/test split, fitting, and simple evaluation.
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
    reg_selector = GreedyStackingEnsembleSelector(
        base_estimators=regressors,
        task="regression",
        cv=5,
        random_state=0,
        tol=1e-6,
    )
    reg_selector.fit(Xr_tr, yr_tr)
    yr_pred = reg_selector.predict(Xr_te)

    print("\n[STACKING] Regression Results")
    print("Selected models:   ", reg_selector.selected_names_)
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
    clf_selector = GreedyStackingEnsembleSelector(
        base_estimators=classifiers,
        task="multiclass",
        cv=5,
        random_state=0,
        tol=1e-6,
    )
    clf_selector.fit(Xc_tr, yc_tr)
    yc_proba = clf_selector.predict_proba(Xc_te)
    yc_pred = clf_selector.predict(Xc_te)

    print("\n[STACKING] Classification Results")
    print("Selected models:      ", clf_selector.selected_names_)
    print("OOF CV loss (logloss):", clf_selector.best_score_)
    print("Test    log_loss:     ", log_loss(yc_te, yc_proba))
    print("Test    accuracy:     ", accuracy_score(yc_te, yc_pred))
