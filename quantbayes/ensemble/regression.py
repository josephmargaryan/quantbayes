# ensemble_regression_v2.py

import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class EnsembleRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        n_splits: int = 5,
        ensemble_method: str = "weighted_average",  # or "stacking"
        weights: Optional[Dict[str, float]] = None,
        meta_learner: Optional[BaseEstimator] = None,
        cv: Optional[KFold] = None,
        random_state: Optional[int] = 42,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.cv = cv
        self.random_state = random_state
        if ensemble_method not in {"weighted_average", "stacking"}:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        names = list(self.models.keys())
        self.fitted_models_ = {}

        if self.ensemble_method == "stacking":
            splitter = self.cv or KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            oof = np.zeros((X.shape[0], len(names)))
            for j, name in enumerate(names):
                base = self.models[name]
                col = np.zeros(X.shape[0], dtype=float)
                for tr, va in splitter.split(X, y):
                    m = clone(base)
                    if sample_weight is None:
                        m.fit(X[tr], y[tr])
                    else:
                        try:
                            m.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
                        except TypeError:
                            m.fit(X[tr], y[tr])
                    col[va] = m.predict(X[va])
                oof[:, j] = col
                fm = clone(base)
                if sample_weight is None:
                    fm.fit(X, y)
                else:
                    try:
                        fm.fit(X, y, sample_weight=sample_weight)
                    except TypeError:
                        fm.fit(X, y)
                self.fitted_models_[name] = fm
            self.meta_fitted_ = clone(self.meta_learner or LinearRegression()).fit(
                oof, y
            )
            self.oof_predictions_ = oof
            self.train_predictions_ = self.meta_fitted_.predict(oof)
        else:
            cols = []
            for name, base in self.models.items():
                m = clone(base)
                if sample_weight is None:
                    m.fit(X, y)
                else:
                    try:
                        m.fit(X, y, sample_weight=sample_weight)
                    except TypeError:
                        m.fit(X, y)
                self.fitted_models_[name] = m
                cols.append(m.predict(X)[:, None])
            P = np.hstack(cols)  # (N, M)

            if self.weights is None:
                w = np.ones(P.shape[1], dtype=float) / P.shape[1]
            else:
                w = np.array([self.weights.get(nm, 0.0) for nm in names], dtype=float)
                if not np.any(w > 0):
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()

            self.train_predictions_ = P @ w
            self._weights_arr_ = w

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        if self.ensemble_method == "stacking":
            base = [self.fitted_models_[nm].predict(X)[:, None] for nm in self.models]
            return self.meta_fitted_.predict(np.hstack(base))
        P = np.hstack([m.predict(X)[:, None] for m in self.fitted_models_.values()])
        w = getattr(self, "_weights_arr_", None)
        if w is None:
            w = np.ones(P.shape[1], dtype=float) / P.shape[1]
        return P @ w
