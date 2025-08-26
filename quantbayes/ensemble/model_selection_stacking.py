# greedy_stacking_ensemble_v2.py

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

    def _default_meta(self):
        return (
            LinearRegression()
            if self.task == "regression"
            else LogisticRegression(solver="lbfgs", max_iter=1000)
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
                oof[va, 0] = m.predict(X[va])
            return oof

        # classification: align columns to `classes`
        C = len(classes)
        oof = np.zeros((n, C), dtype=float)
        for tr, va in splitter.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            P = m.predict_proba(X[va])
            cols = np.zeros((len(va), C), dtype=float)
            for i, cls in enumerate(m.classes_):
                j = int(np.where(classes == cls)[0][0])
                cols[:, j] = P[:, i]
            oof[va] = cols
        return oof

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

        # greedy forward selection
        selected: List[str] = []
        best_score = np.inf
        outer_cv = self._get_cv()

        while True:
            improved = False
            best_cand = None
            cand_score = best_score

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
                except Exception as e:
                    _logger.warning("CV failed for %s: %s", n, e)
                    continue
                if loss < cand_score * (1 - self.tol):
                    improved, best_cand, cand_score = True, n, loss

            if not improved or best_cand is None:
                break
            selected.append(best_cand)
            best_score = cand_score
            _logger.info(
                "Added %s → loss=%.6f; selected=%r", best_cand, best_score, selected
            )
            if self.max_models and len(selected) >= self.max_models:
                break

        # fallback to best single if needed
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
                        cv=outer_cv,
                        n_jobs=self.n_jobs,
                    )
                    raw[n] = -np.mean(sc)
                except Exception:
                    continue
            if not raw:
                raise RuntimeError("All base estimators failed CV.")
            selected, best_score = [min(raw, key=raw.get)], min(raw.values())

        self.selected_names_ = selected
        X_meta_full = np.column_stack([self.oof_preds_[m] for m in selected])
        self.meta_learner_ = clone(self.meta_learner_).fit(X_meta_full, y_arr)

        self.fitted_estimators_ = {
            m: clone(self.base_estimators[m]).fit(X_arr, y_arr) for m in selected
        }
        self.best_score_ = best_score
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
                else m.predict(X_arr)[:, None]
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
        return self.meta_learner_.predict_proba(np.hstack(cols))
