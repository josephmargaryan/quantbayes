# ensemble_binary_v2.py

import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

"""
Example Usage;
    ```python 
        models = {
        "lr": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    ens = EnsembleBinary(
        models=models,
        ensemble_method="weighted_average",
        combine="geom",
        weights={"lr": 0.4, "rf": 0.6},
        random_state=42,
    )

    ens.fit(X_train, y_train)

    # save
    joblib.dump(ens, "ensemble_binary_v2.joblib", compress=3)

    # later: load and use
    ens2 = joblib.load("ensemble_binary_v2.joblib")
    y_proba = ens2.predict_proba(X_test)
    y_pred  = ens2.predict(X_test)
```
"""

EPS = 1e-12


def _pos_index(model_classes: np.ndarray, pos_label) -> int:
    idx = np.where(model_classes == pos_label)[0]
    if idx.size == 0:
        raise ValueError(
            f"pos_label={pos_label!r} not found in model.classes_={model_classes!r}"
        )
    return int(idx[0])


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log(1 - p)


class EnsembleBinary(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        n_splits: int = 5,
        ensemble_method: str = "weighted_average",  # or "stacking"
        weights: Optional[Dict[str, float]] = None,  # for weighted_average
        meta_learner: Optional[BaseEstimator] = None,
        cv: Optional[StratifiedKFold] = None,
        combine: str = "proba",  # "proba" | "geom" | "logit"
        pos_label=None,  # if None → max(self.classes_)
        random_state: Optional[int] = 42,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_learner = meta_learner
        self.cv = cv
        self.combine = combine
        self.pos_label = pos_label
        self.random_state = random_state

        if ensemble_method not in {"weighted_average", "stacking"}:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")
        if combine not in {"proba", "geom", "logit"}:
            raise ValueError("combine must be one of {'proba','geom','logit'}")

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Binary ensemble requires exactly 2 classes.")
        self.n_features_in_ = X.shape[1]

        self.pos_label_ = (
            self.pos_label if self.pos_label is not None else np.max(self.classes_)
        )
        self.neg_label_ = (
            self.classes_[0]
            if self.classes_[1] == self.pos_label_
            else self.classes_[1]
        )

        splitter = self.cv or StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        names = list(self.models.keys())
        self.fitted_models_ = {}

        if self.ensemble_method == "stacking":
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
                    pos_idx = _pos_index(m.classes_, self.pos_label_)
                    col[va] = m.predict_proba(X[va])[:, pos_idx]
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
            self.meta_fitted_ = clone(
                self.meta_learner or LogisticRegression(max_iter=1000)
            ).fit(oof, y)
            self.oof_predictions_ = oof
            self.train_predictions_proba_ = self.meta_fitted_.predict_proba(oof)[
                :, _pos_index(self.meta_fitted_.classes_, self.pos_label_)
            ]
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
                pos_idx = _pos_index(m.classes_, self.pos_label_)
                cols.append(m.predict_proba(X)[:, pos_idx][:, None])
            P = np.hstack(cols)  # (N, M)

            if self.weights is None:
                w = np.ones(P.shape[1], dtype=float) / P.shape[1]
            else:
                w = np.array([self.weights.get(nm, 0.0) for nm in names], dtype=float)
                if not np.any(w > 0):
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()

            if self.combine == "proba":
                self.train_predictions_proba_ = P @ w
            elif self.combine == "geom":
                P_ = np.clip(P, EPS, 1 - EPS)
                pos_geom = np.exp((np.log(P_) * w).sum(axis=1))
                neg_geom = np.exp((np.log(1.0 - P_) * w).sum(axis=1))
                self.train_predictions_proba_ = pos_geom / (pos_geom + neg_geom)
            else:  # logit averaging
                z = (_logit(P) * w).sum(axis=1)
                self.train_predictions_proba_ = 1.0 / (1.0 + np.exp(-z))

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        names = list(self.models.keys())
        if self.ensemble_method == "stacking":
            M = np.column_stack(
                [
                    self.fitted_models_[nm].predict_proba(X)[
                        :, _pos_index(self.fitted_models_[nm].classes_, self.pos_label_)
                    ]
                    for nm in names
                ]
            )
            pos = self.meta_fitted_.predict_proba(M)[
                :, _pos_index(self.meta_fitted_.classes_, self.pos_label_)
            ]
        else:
            P = np.column_stack(
                [
                    self.fitted_models_[nm].predict_proba(X)[
                        :, _pos_index(self.fitted_models_[nm].classes_, self.pos_label_)
                    ]
                    for nm in names
                ]
            )
            if self.weights is None:
                w = np.ones(P.shape[1], dtype=float) / P.shape[1]
            else:
                w = np.array([self.weights.get(nm, 0.0) for nm in names], dtype=float)
                if not np.any(w > 0):
                    raise ValueError("Sum of weights cannot be zero.")
                w /= w.sum()

            if self.combine == "proba":
                pos = P @ w
            elif self.combine == "geom":
                P_ = np.clip(P, EPS, 1 - EPS)
                pos_geom = np.exp((np.log(P_) * w).sum(axis=1))
                neg_geom = np.exp((np.log(1.0 - P_) * w).sum(axis=1))
                pos = pos_geom / (pos_geom + neg_geom)
            else:
                z = (_logit(P) * w).sum(axis=1)
                pos = 1.0 / (1.0 + np.exp(-z))

        # assemble columns ordered by self.classes_ (neg first/wherever it is)
        proba = np.zeros((X.shape[0], 2), dtype=float)
        neg_idx = int(np.where(self.classes_ == self.neg_label_)[0][0])
        pos_idx = int(np.where(self.classes_ == self.pos_label_)[0][0])
        proba[:, pos_idx] = pos
        proba[:, neg_idx] = 1.0 - pos
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        pos_idx = int(np.where(self.classes_ == self.pos_label_)[0][0])
        return (proba[:, pos_idx] >= 0.5).astype(self.classes_.dtype)
