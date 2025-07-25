# multiclass_slice_ensemble.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

__all__ = ["SliceWiseEnsembleMulticlass"]


class SliceWiseEnsembleMulticlass(BaseEstimator, ClassifierMixin):
    """
    Slice-wise ensemble for multiclass classification, stacking or weighted-average,
    with fallback for folds containing only one class.
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
        # record the global classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes for multiclass ensemble.")

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
        # 1) collect OOF probability arrays for each base
        oof_b = {name: np.zeros((n, self.n_classes_)) for name in self.base_models}

        for name, mdl in self.base_models.items():
            for tr, val in self.cv.split(Xs):
                y_tr = ys[tr]
                if len(np.unique(y_tr)) < 2:
                    # fallback: all probability mass on the single class
                    cls = int(y_tr[0])
                    j = np.where(self.classes_ == cls)[0][0]
                    oof_b[name][val, j] = 1.0
                else:
                    m = clone(mdl)
                    m.fit(Xs[tr], y_tr)
                    probs = m.predict_proba(Xs[val])
                    # map model.classes_ → global columns
                    for i, cls in enumerate(m.classes_):
                        j = np.where(self.classes_ == cls)[0][0]
                        oof_b[name][val, j] = probs[:, i]

        # 2) fit each base on full slice
        fitted = {
            name: clone(mdl).fit(Xs, ys) for name, mdl in self.base_models.items()
        }

        # 3) build the ensemble for this slice
        if self.ensemble_method == "stacking":
            # stack all base‐probs as meta features
            meta_X = np.hstack([oof_b[name] for name in self.base_models])
            meta = clone(self.meta_learner).fit(meta_X, ys)

            # map meta.predict back into global classes
            raw_proba = meta.predict_proba(meta_X)
            slice_proba = np.zeros((n, self.n_classes_))
            for i, cls in enumerate(meta.classes_):
                j = np.where(self.classes_ == cls)[0][0]
                slice_proba[:, j] = raw_proba[:, i]

            oof_pred = slice_proba.argmax(axis=1)
            bw = None

        else:  # weighted_average
            # weight each base by its OOF accuracy
            accs = np.array(
                [
                    accuracy_score(ys, oof_b[name].argmax(axis=1))
                    for name in self.base_models
                ]
            )
            if accs.sum() > 0:
                ws = accs / accs.sum()
            else:
                ws = np.ones_like(accs) / len(accs)
            bw = dict(zip(self.base_models, ws))

            # compute combined OOF proba
            slice_proba = sum(
                ws[i] * oof_b[name] for i, name in enumerate(self.base_models)
            )
            oof_pred = slice_proba.argmax(axis=1)
            meta = None

        return oof_pred, fitted, meta, bw

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        n = X.shape[0]
        agg = np.zeros((n, self.n_classes_))

        for w, (models, meta, bw) in zip(self.slice_weights_, self._slices):
            if self.ensemble_method == "stacking":
                # build the same meta‐features at prediction time
                mats = []
                for name in self.base_models:
                    probs = models[name].predict_proba(X)
                    mat = np.zeros((n, self.n_classes_))
                    for i, cls in enumerate(models[name].classes_):
                        j = np.where(self.classes_ == cls)[0][0]
                        mat[:, j] = probs[:, i]
                    mats.append(mat)
                meta_X = np.hstack(mats)

                raw_proba = meta.predict_proba(meta_X)
                slice_proba = np.zeros((n, self.n_classes_))
                for i, cls in enumerate(meta.classes_):
                    j = np.where(self.classes_ == cls)[0][0]
                    slice_proba[:, j] = raw_proba[:, i]

            else:  # weighted_average
                mats = []
                for name in self.base_models:
                    probs = models[name].predict_proba(X)
                    mat = np.zeros((n, self.n_classes_))
                    for i, cls in enumerate(models[name].classes_):
                        j = np.where(self.classes_ == cls)[0][0]
                        mat[:, j] = probs[:, i]
                    mats.append(mat)
                slice_proba = sum(
                    bw[name] * mats[i] for i, name in enumerate(self.base_models)
                )

            agg += w * slice_proba

        return agg


if __name__ == "__main__":
    # synthetic multiclass test
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    np.random.seed(0)
    t = np.arange(300)
    X = (t % 50).reshape(-1, 1)
    y = t // 50  # six classes: 0..5

    base_models = {
        "logreg": LogisticRegression(max_iter=200),
        "tree": DecisionTreeClassifier(max_depth=5),
    }

    for method in ["stacking", "weighted_average"]:
        clf = SliceWiseEnsembleMulticlass(
            base_models=base_models,
            ensemble_method=method,
            slice_fractions=[1.0, 0.75, 0.5],
            cv=TimeSeriesSplit(n_splits=4, gap=1),
            meta_learner=LogisticRegression(max_iter=200),
        ).fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        print(f"{method:>15} accuracy:", accuracy_score(y, preds))
        print(f"{method:>15} slice weights:", clf.slice_weights_)
        print(f"{method:>15} proba shape:", proba.shape)
        print("-" * 50)
