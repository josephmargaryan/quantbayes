# regression_slice_ensemble.py

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import pearsonr

__all__ = ["SliceWiseEnsembleRegressor"]


class SliceWiseEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Slice-wise ensemble for regression with stacking or weighted-average.
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
        self.meta_learner = meta_learner or Ridge()

    def _fit_slice(self, Xs, ys):
        n = Xs.shape[0]
        # 1) generate OOF preds
        oof_b = {name: np.zeros(n) for name in self.base_models}
        for name, mdl in self.base_models.items():
            for tr, val in self.cv.split(Xs):
                m = clone(mdl)
                m.fit(Xs[tr], ys[tr])
                oof_b[name][val] = m.predict(Xs[val])
        # 2) fit each base on full slice
        fitted = {
            name: clone(mdl).fit(Xs, ys) for name, mdl in self.base_models.items()
        }

        if self.ensemble_method == "stacking":
            # 2-stage stacking
            meta_X = np.column_stack([oof_b[name] for name in self.base_models])
            meta = clone(self.meta_learner).fit(meta_X, ys)
            oof_pred = meta.predict(meta_X)
            weights_b = None
        else:
            # weighted-average by Pearson corr on OOF
            scores = np.array(
                [abs(pearsonr(ys, oof_b[name])[0]) for name in self.base_models]
            )
            if scores.sum() == 0:
                w = np.ones_like(scores) / len(scores)
            else:
                w = scores / scores.sum()
            weights_b = dict(zip(self.base_models, w))
            oof_pred = sum(weights_b[name] * oof_b[name] for name in self.base_models)
            meta = None

        return oof_pred, fitted, meta, weights_b

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n = X.shape[0]
        cuts = [int((1 - f) * n) for f in self.slice_fractions]

        slice_info, slice_scores = [], []
        for cut in cuts:
            Xs, ys = X[cut:], y[cut:]
            oof, models, meta, bw = self._fit_slice(Xs, ys)
            slice_info.append((models, meta, bw))
            slice_scores.append(abs(pearsonr(ys, oof)[0]))

        total = sum(slice_scores)
        if total == 0:
            raise ValueError("All slice correlations zero.")
        self.slice_weights_ = [s / total for s in slice_scores]
        self._slices = slice_info
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        preds = np.zeros(X.shape[0])
        for w, (models, meta, bw) in zip(self.slice_weights_, self._slices):
            if self.ensemble_method == "stacking":
                mat = np.column_stack([models[n].predict(X) for n in self.base_models])
                preds += w * meta.predict(mat)
            else:
                tmp = sum(bw[n] * models[n].predict(X) for n in self.base_models)
                preds += w * tmp
        return preds


if __name__ == "__main__":
    # synthetic regression test
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score

    np.random.seed(0)
    t = np.arange(200)
    X = t.reshape(-1, 1)
    y = np.sin(t / 20) + 0.1 * np.random.randn(200)

    base_models = {
        "ridge": Ridge(alpha=1.0),
        "tree": DecisionTreeRegressor(max_depth=5),
    }

    for method in ["stacking", "weighted_average"]:
        mdl = SliceWiseEnsembleRegressor(
            base_models=base_models,
            ensemble_method=method,
            slice_fractions=[1.0, 0.75, 0.5],
            cv=TimeSeriesSplit(n_splits=4, gap=1),
            meta_learner=Ridge(alpha=0.5),
        ).fit(X, y)

        preds = mdl.predict(X)
        print(f"{method:>15} R²:", r2_score(y, preds))
        print(f"{method:>15} slice weights:", mdl.slice_weights_)
        print("-" * 50)
