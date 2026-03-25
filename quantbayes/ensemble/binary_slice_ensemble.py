# binary_slice_ensemble.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

__all__ = ["SliceWiseEnsembleClassifier"]


class SliceWiseEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Slice-wise binary ensemble with stacking or weighted-average, plus predict_proba.
    """

    def __init__(
        self,
        base_models: dict,
        ensemble_method: str = "stacking",  # "stacking" or "weighted_average"
        slice_fractions: list = None,
        cv: TimeSeriesSplit = None,
        meta_learner=None,
    ):
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.slice_fractions = sorted(slice_fractions or [1.0, 0.75, 0.5], reverse=True)
        self.cv = cv or TimeSeriesSplit(n_splits=3, gap=1)
        self.meta_learner = meta_learner or LogisticRegression(max_iter=200)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Binary classification requires exactly 2 classes.")
        n = X.shape[0]
        cuts = [int((1 - f) * n) for f in self.slice_fractions]

        self._slices = []
        scores = []
        for cut in cuts:
            Xs, ys = X[cut:], y[cut:]
            oof_pred, models, meta, bw = self._fit_slice(Xs, ys)
            self._slices.append((models, meta, bw))
            scores.append(accuracy_score(ys, oof_pred))

        total = sum(scores)
        if total <= 0:
            raise ValueError("All slice accuracies zero.")
        self.slice_weights_ = [s / total for s in scores]
        self.is_fitted_ = True
        return self

    def _fit_slice(self, Xs, ys):
        n = Xs.shape[0]
        # collect OOF probability for positive class
        oof_proba = {name: np.zeros(n) for name in self.base_models}
        for name, mdl in self.base_models.items():
            for tr, val in self.cv.split(Xs):
                m = clone(mdl)
                m.fit(Xs[tr], ys[tr])
                oof_proba[name][val] = m.predict_proba(Xs[val])[:, 1]

        # fit each base on full slice
        fitted = {
            name: clone(mdl).fit(Xs, ys) for name, mdl in self.base_models.items()
        }

        if self.ensemble_method == "stacking":
            # meta-learner on OOF probs
            X_meta = np.column_stack([oof_proba[n] for n in self.base_models])
            meta = clone(self.meta_learner).fit(X_meta, ys)
            oof_pred = meta.predict(X_meta)
            bw = None
        else:
            # weighted-average by OOF accuracy
            scores = np.array(
                [
                    accuracy_score(ys, (oof_proba[n] >= 0.5).astype(int))
                    for n in self.base_models
                ]
            )
            if scores.sum() > 0:
                ws = scores / scores.sum()
            else:
                ws = np.ones_like(scores) / len(scores)
            bw = dict(zip(self.base_models, ws))
            ens_proba = sum(bw[n] * oof_proba[n] for n in self.base_models)
            oof_pred = (ens_proba >= 0.5).astype(int)
            meta = None

        return oof_pred, fitted, meta, bw

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        n = X.shape[0]
        agg = np.zeros((n, 2))
        for w, (models, meta, bw) in zip(self.slice_weights_, self._slices):
            if self.ensemble_method == "stacking":
                feats = np.column_stack(
                    [models[n].predict_proba(X)[:, 1] for n in self.base_models]
                )
                slice_p = meta.predict_proba(feats)
            else:
                slice_pos = sum(
                    bw[n] * models[n].predict_proba(X)[:, 1] for n in self.base_models
                )
                slice_p = np.vstack([1 - slice_pos, slice_pos]).T
            agg += w * slice_p
        return agg


if __name__ == "__main__":
    # synthetic binary classification test
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    np.random.seed(0)
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=0
    )

    base_models = {
        "logreg": LogisticRegression(max_iter=200),
        "tree": DecisionTreeClassifier(max_depth=5),
    }

    for method in ["stacking", "weighted_average"]:
        clf = SliceWiseEnsembleClassifier(
            base_models=base_models,
            ensemble_method=method,
            slice_fractions=[1.0, 0.75, 0.5],
            cv=TimeSeriesSplit(n_splits=4, gap=1),
            meta_learner=LogisticRegression(max_iter=200),
        ).fit(X, y)

        p = clf.predict(X)
        pr = clf.predict_proba(X)
        print(f"{method:>15} accuracy:", accuracy_score(y, p))
        print(f"{method:>15} slice weights:", clf.slice_weights_)
        print(f"{method:>15} prob[0:5]:\n", pr[:5])
        print("-" * 50)
