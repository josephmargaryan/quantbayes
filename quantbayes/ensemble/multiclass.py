# ensemble_multiclass_v2.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Optional, Sequence

from .utils_ensemble import (
    align_proba_to_global,
    geometric_mean_ensemble,
    binary_logit_average,
)


class EnsembleMulticlass(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        n_splits: int = 5,
        ensemble_method: str = "weighted_average",  # or "stacking"
        weights: Optional[Dict[str, float]] = None,
        meta_learner: Optional[BaseEstimator] = None,
        cv: Optional[StratifiedKFold] = None,
        combine: str = "proba",  # "proba" or "geom" (geometric mean)
        calibration: Optional[
            str
        ] = None,  # None | "sigmoid" | "isotonic" (applied to meta_learner via CalibratedClassifierCV)
        random_state: Optional[int] = 42,
    ):
        self.models = models
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.cv = cv
        self.combine = combine
        self.calibration = calibration
        self.random_state = random_state
        if ensemble_method not in {"weighted_average", "stacking"}:
            raise ValueError("ensemble_method must be 'weighted_average' or 'stacking'")
        self.meta_learner = meta_learner or LogisticRegression(
            solver="lbfgs", max_iter=1000, multi_class="auto"
        )

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        names = list(self.models)

        # default CV: stratified for classification
        splitter = self.cv or StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        if self.ensemble_method == "stacking":
            n = X.shape[0]
            oof = np.zeros((n, len(names) * C))
            self.fitted_models_ = {}

            for idx, name in enumerate(names):
                base = self.models[name]
                tmp = np.zeros((n, C))
                for tr, va in splitter.split(X, y):
                    m = clone(base)
                    if sample_weight is None:
                        m.fit(X[tr], y[tr])
                    else:
                        # pass weights if supported
                        try:
                            m.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
                        except TypeError:
                            m.fit(X[tr], y[tr])
                    P = m.predict_proba(X[va])
                    P_aligned = align_proba_to_global(P, m.classes_, self.classes_)
                    tmp[va] = P_aligned
                oof[:, idx * C : (idx + 1) * C] = tmp

                # fit final base on full data
                fm = clone(base)
                if sample_weight is None:
                    fm.fit(X, y)
                else:
                    try:
                        fm.fit(X, y, sample_weight=sample_weight)
                    except TypeError:
                        fm.fit(X, y)
                self.fitted_models_[name] = fm

            meta = clone(self.meta_learner)
            if self.calibration is not None:
                from sklearn.calibration import CalibratedClassifierCV

                # CV-calibrate the meta learner on OOF to reduce leakage
                meta = CalibratedClassifierCV(meta, method=self.calibration, cv=3)
            meta.fit(oof, y)
            self.meta_fitted_ = meta
            self.oof_predictions_ = oof

        else:  # weighted_average
            self.fitted_models_ = {}
            prob_list = []
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
                P = m.predict_proba(X)
                P_aligned = align_proba_to_global(P, m.classes_, self.classes_)
                prob_list.append(P_aligned)

            A = np.stack(prob_list, axis=0)  # (M, N, C)
            if self.weights is None:
                w = np.ones(len(names)) / len(names)
            else:
                w = np.array([self.weights.get(nm, 0.0) for nm in names], dtype=float)
                if not np.any(w > 0):
                    raise ValueError("All provided weights are zero.")
                w /= w.sum()

            if self.combine == "proba":
                self.train_predictions_proba_ = np.tensordot(w, A, axes=1)  # (N,C)
            elif self.combine == "geom":
                self.train_predictions_proba_ = geometric_mean_ensemble(A, w)
            else:
                raise ValueError("combine must be 'proba' or 'geom'")

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X)
        names = list(self.models)

        if self.ensemble_method == "stacking":
            blocks = []
            for n in names:
                P = self.fitted_models_[n].predict_proba(X)
                blocks.append(
                    align_proba_to_global(
                        P, self.fitted_models_[n].classes_, self.classes_
                    )
                )
            M = np.hstack(blocks)
            return self.meta_fitted_.predict_proba(M)

        # weighted_average path
        mats = []
        for n in names:
            P = self.fitted_models_[n].predict_proba(X)
            mats.append(
                align_proba_to_global(P, self.fitted_models_[n].classes_, self.classes_)
            )
        A = np.stack(mats, axis=0)  # (M,N,C)

        if self.weights is None:
            w = np.ones(A.shape[0], dtype=float) / A.shape[0]
        else:
            w = np.array([self.weights.get(nm, 0.0) for nm in names], dtype=float)
            if not np.any(w > 0):
                raise ValueError("All provided weights are zero.")
            w /= w.sum()

        if self.combine == "proba":
            return np.tensordot(w, A, axes=1)
        elif self.combine == "geom":
            return geometric_mean_ensemble(A, w)
        else:
            raise ValueError("combine must be 'proba' or 'geom'")

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
